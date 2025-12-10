"""
高度图订阅器模块
功能：订阅 ROS1 点云话题，处理点云数据，生成高度图

工作流程：
1. 订阅 /elevation_map_fused_visualization/elevation_cloud 话题
2. 通过 TF 获取机器人位置
3. 转换点云坐标系
4. 处理点云，生成高度图网格
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import threading
from typing import Optional

from utils.heightmap_processor import HeightMapProcessor, KNN_DEFAULT_HEIGHT_OFFSET

# ROS1 话题配置
ROS_HEIGHTMAP_TOPIC = "/elevation_map_fused_visualization/elevation_cloud"
ROS_BASE_FRAME = "odom_corrected"      # 全局坐标系
ROS_ROBOT_FRAME = "torso_link"        # 机器人本体坐标系


class HeightMapSubscriber:
    """
    高度图订阅器类
    功能：订阅 ROS1 点云话题，处理点云数据，生成高度图
    """
    
    def __init__(self, topic_name: str = ROS_HEIGHTMAP_TOPIC,
                 base_frame: str = ROS_BASE_FRAME,
                 robot_frame: str = ROS_ROBOT_FRAME):
        """
        初始化高度图订阅器
        
        @param topic_name: ROS1 点云话题名称
        @param base_frame: 全局坐标系名称（odom_corrected）
        @param robot_frame: 机器人本体坐标系名称（torso_link）
        """
        self.topic_name = topic_name
        self.base_frame = base_frame
        self.robot_frame = robot_frame
        
        # TF 缓冲区和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 高度图处理器
        self.processor = HeightMapProcessor()
        
        # 订阅点云话题
        self.cloud_sub = rospy.Subscriber(
            topic_name,
            PointCloud2,
            self._pointcloud_callback,
            queue_size=1
        )
        
        # 状态标志（线程安全）
        self.has_data = False
        self.last_update_time = rospy.Time(0)
        # 线程锁：保护 has_data 和 last_update_time 在多线程环境下的安全访问
        # 因为 ROS 回调函数在独立线程中运行，而 get_heightmap() 在主线程中调用
        self.lock = threading.Lock()
        
        rospy.loginfo(f"HeightMapSubscriber initialized. Subscribed to {topic_name}")
    
    def _pointcloud_callback(self, msg: PointCloud2):
        """
        点云回调函数
        
        @param msg: PointCloud2 消息
        """
        try:
            # 获取 TF 变换
            # lookup_transform(target_frame, source_frame, time, timeout)
            # - target_frame: 目标坐标系（要转换到的坐标系）
            # - source_frame: 源坐标系（要转换的坐标系）
            # - time: 查询的时间戳（rospy.Time(0) 表示最新可用时间）
            # - timeout: 超时时间（如果在这个时间内找不到变换，抛出异常）
            # 返回：TransformStamped 对象，包含从 source_frame 到 target_frame 的变换
            try:
                # 获取点云坐标系到 base 坐标系的变换
                transform_cloud_to_base = self.tf_buffer.lookup_transform(
                    self.base_frame,        # 目标：odom_corrected
                    msg.header.frame_id,     # 源：点云的坐标系（通常是 odom）
                    rospy.Time(0),           # 查询最新可用时间
                    rospy.Duration(0.1)      # 超时 0.1 秒
                )
                
                # 获取机器人本体到 base 坐标系的变换
                transform_torso_to_base = self.tf_buffer.lookup_transform(
                    self.base_frame,        # 目标：odom_corrected
                    self.robot_frame,       # 源：torso_link
                    rospy.Time(0),          # 查询最新可用时间
                    rospy.Duration(0.1)     # 超时 0.1 秒
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(1.0, f"TF lookup failed: {e}. Skipping heightmap update.")
                return
            
            # 提取机器人位置和姿态
            torso_pos = np.array([
                transform_torso_to_base.transform.translation.x,
                transform_torso_to_base.transform.translation.y,
                transform_torso_to_base.transform.translation.z
            ], dtype=np.float32)
            
            # 提取机器人 yaw 角度（从四元数）
            q = transform_torso_to_base.transform.rotation
            # 四元数转 yaw
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            
            # 转换点云到 base 坐标系
            transformed_msg = do_transform_cloud(msg, transform_cloud_to_base)
            
            # 提取点云数据
            # 注意：点云的 z 坐标表示地面高度（在 odom_corrected 坐标系中）
            cloud_points_list = []
            for point in pc2.read_points(transformed_msg, field_names=("x", "y", "z"), skip_nans=True):
                cloud_points_list.append([point[0], point[1], point[2]])  # z 是地面高度
            
            if len(cloud_points_list) == 0:
                rospy.logwarn_throttle(1.0, "Point cloud is empty after transformation.")
                return
            
            cloud_points = np.array(cloud_points_list, dtype=np.float32)
            
            # 更新高度图
            # 高度计算：机器人 Z - 地面高度（点云的 z），与训练代码一致
            self.processor.update_heightmap(cloud_points, torso_pos, yaw)
            
            # 更新状态
            with self.lock:
                self.has_data = True
                self.last_update_time = rospy.Time.now()
                
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Error processing point cloud: {e}")
    
    def get_heightmap(self) -> Optional[np.ndarray]:
        """
        获取当前高度图（225 维向量，15×15 网格）
        
        高度值含义：机器人 Z - 地面高度（与训练代码一致）
        
        @return: numpy array, shape (225,), dtype=np.float32，如果数据未准备好则返回 None
        """
        with self.lock:
            if not self.has_data:
                return None
            return self.processor.get_heightmap()
    
    def get_robot_position(self) -> Optional[np.ndarray]:
        """
        获取机器人位置（通过 TF）
        
        @return: numpy array, shape (3,), [x, y, z] 在 odom_corrected 坐标系中，如果失败则返回 None
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )
            return np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=np.float32)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"TF lookup failed: {e}")
            return None
    
    def is_ready(self) -> bool:
        """
        检查高度图数据是否准备好
        
        @return: True 如果数据已准备好，False 否则
        """
        with self.lock:
            return self.has_data

