"""
高度图处理模块
功能：处理点云数据，生成 15×15 = 225 维高度图网格

处理流程：
1. 接收点云数据（PointCloud2）
2. 转换坐标系（从点云坐标系到 odom_corrected）
3. 获取机器人位置（通过 TF）
4. 计算网格点全局坐标（相对于机器人）
5. KNN 搜索最近邻点
6. IDW 插值计算高度
7. 生成 15×15 高度图网格并展平为 225 维向量

高度计算方式（与训练代码一致）：
- 高度值 = 机器人 Z - 地面高度（点云的 Z 坐标）
- 如果地面高度为 0，机器人 Z 为 0.85，则高度 = 0.85 - 0 = 0.85
"""

import numpy as np
import threading

# 高度图网格配置（与训练代码一致：15×15 = 225）
GRID_X_SIDE_LENGTH = 15      # X 方向网格点数
GRID_Y_SIDE_LENGTH = 15      # Y 方向网格点数
GRID_X_RESOLUTION = 0.1      # X 方向分辨率（米）
GRID_Y_RESOLUTION = 0.1      # Y 方向分辨率（米）
KNN_K = 3                    # K 近邻数量
KNN_MAX_DISTANCE = 0.15      # 最大搜索距离（米）
KNN_DEFAULT_HEIGHT_OFFSET = 0.85  # 默认高度偏移（相对于机器人 Z）

GRID_POINTS_COUNT = GRID_X_SIDE_LENGTH * GRID_Y_SIDE_LENGTH  # 225 个网格点


class HeightMapProcessor:
    """
    高度图处理类
    功能：处理点云数据，生成高度图网格
    """
    
    def __init__(self):
        """初始化高度图处理器"""
        # 创建本地查询点网格（相对于机器人本体的局部坐标）
        self.local_query_points = self._create_elevation_grid()
        
        # 当前高度图数据（15×15 网格）
        self.heights = np.full((GRID_Y_SIDE_LENGTH, GRID_X_SIDE_LENGTH), 
                              KNN_DEFAULT_HEIGHT_OFFSET, dtype=np.float32)
        
        # 线程锁：保护高度图数据在多线程环境下的安全访问
        # 原因：update_heightmap() 在 ROS 回调线程中调用，get_heightmap() 在主控制线程中调用
        # 使用锁确保读取和写入操作不会同时进行，避免数据竞争
        self.lock = threading.Lock()
        
    def _create_elevation_grid(self) -> np.ndarray:
        """
        创建高度图网格的局部查询点（相对于机器人本体）
        
        @return: numpy array, shape (225, 2), 每行是 [x, y] 局部坐标
        """
        local_points = np.zeros((GRID_POINTS_COUNT, 2), dtype=np.float32)
        
        # 计算网格范围的一半
        half_x = GRID_X_SIDE_LENGTH / 2.0 * GRID_X_RESOLUTION
        half_y = GRID_Y_SIDE_LENGTH / 2.0 * GRID_Y_RESOLUTION
        
        point_idx = 0
        for j in range(GRID_Y_SIDE_LENGTH):  # Y 方向（行）
            y = -half_y + (j + 0.5) * GRID_Y_RESOLUTION
            for i in range(GRID_X_SIDE_LENGTH):  # X 方向（列）
                x = -half_x + (i + 0.5) * GRID_X_RESOLUTION
                local_points[point_idx, 0] = x
                local_points[point_idx, 1] = y
                point_idx += 1
        
        return local_points
    
    def _get_global_query_points(self, torso_pos: np.ndarray, 
                                  torso_yaw: float) -> np.ndarray:
        """
        将局部查询点转换为全局坐标（在 odom_corrected 坐标系中）
        
        @param torso_pos: 机器人位置 [x, y, z]，在 odom_corrected 坐标系中
        @param torso_yaw: 机器人 yaw 角度（弧度）
        @return: numpy array, shape (225, 2), 每行是 [x, y] 全局坐标
        """
        # 创建 yaw 旋转矩阵（2D）
        cos_yaw = np.cos(torso_yaw)
        sin_yaw = np.sin(torso_yaw)
        rot_matrix = np.array([[cos_yaw, -sin_yaw],
                               [sin_yaw, cos_yaw]], dtype=np.float32)
        
        # 旋转局部点
        local_points_T = self.local_query_points.T  # (2, 225)
        rotated_points = rot_matrix @ local_points_T  # (2, 225)
        
        # 平移到机器人位置
        global_points = rotated_points.T + torso_pos[:2]  # (225, 2)
        
        return global_points
    
    def _interpolate_height_idw(self, query_point_xy: np.ndarray,
                                 cloud_points: np.ndarray,
                                 torso_z: float) -> float:
        """
        使用逆距离加权（IDW）插值计算查询点的高度
        
        实现与 C++ 代码（videomimic_inference_real.cpp）保持一致
        
        高度计算方式（与训练代码一致）：
        - 高度值 = 机器人 Z - 地面高度（点云的 Z 坐标）
        - 如果地面高度为 0，机器人 Z 为 0.85，则高度 = 0.85 - 0 = 0.85
        
        @param query_point_xy: 查询点 [x, y] 坐标
        @param cloud_points: 点云数据，shape (N, 3)，每行是 [x, y, z]（z 是地面高度）
        @param torso_z: 机器人 Z 坐标（在 odom_corrected 坐标系中）
        @return: 高度值 = torso_z - 地面高度（与训练代码一致）
        """
        if cloud_points.shape[0] == 0:
            return KNN_DEFAULT_HEIGHT_OFFSET
        
        # 计算所有点到查询点的距离（平方距离）
        distances_sq = np.sum((cloud_points[:, :2] - query_point_xy) ** 2, axis=1)
        
        # 找到 K 个最近邻（模拟 KD 树搜索）
        if cloud_points.shape[0] <= KNN_K:
            nearest_indices = np.arange(cloud_points.shape[0])
        else:
            # 使用 argpartition 找到前 K 个最小值的索引
            nearest_indices = np.argpartition(distances_sq, KNN_K)[:KNN_K]
            # 对前 K 个索引按距离排序（确保顺序正确，与 C++ 的 KD 树返回顺序一致）
            nearest_indices = nearest_indices[np.argsort(distances_sq[nearest_indices])]
        
        # 检查最近邻距离（与 C++ 代码一致：使用 sqrt 检查第一个最近邻的距离）
        min_dist = np.sqrt(distances_sq[nearest_indices[0]])
        if min_dist > KNN_MAX_DISTANCE:
            return KNN_DEFAULT_HEIGHT_OFFSET
        
        # IDW 权重计算（与 C++ 代码逻辑一致）
        epsilon = 1e-6
        epsilon_sq = epsilon * epsilon
        total_weight = 0.0
        weighted_sum_z = 0.0
        
        # 遍历所有找到的最近邻（与 C++ 代码的循环逻辑一致）
        for i in range(len(nearest_indices)):
            idx = nearest_indices[i]
            dist_sq = distances_sq[idx]
            
            # 如果查询点正好是某个点（与 C++ 代码一致）
            if dist_sq < epsilon_sq:
                # 返回高度相对于机器人 Z
                return torso_z - cloud_points[idx, 2]
            
            # 计算 IDW 权重（与 C++ 代码一致）
            dist = np.sqrt(dist_sq)
            weight = 1.0 / (dist + epsilon)
            weighted_sum_z += cloud_points[idx, 2] * weight
            total_weight += weight
        
        # 计算插值高度（与 C++ 代码一致）
        if total_weight > epsilon:
            interpolated_z_world = weighted_sum_z / total_weight
            # 返回高度相对于机器人 Z 的位置
            return torso_z - interpolated_z_world
        else:
            return KNN_DEFAULT_HEIGHT_OFFSET
    
    def update_heightmap(self, cloud_points: np.ndarray,
                         torso_pos: np.ndarray,
                         torso_yaw: float) -> None:
        """
        更新高度图网格
        
        高度计算方式（与训练代码一致）：
        - 高度值 = 机器人 Z - 地面高度（点云的 Z 坐标）
        - 点云的 Z 坐标表示地面高度（在 odom_corrected 坐标系中）
        
        @param cloud_points: 点云数据，shape (N, 3)，每行是 [x, y, z]（z 是地面高度）
        @param torso_pos: 机器人位置 [x, y, z]（在 odom_corrected 坐标系中）
        @param torso_yaw: 机器人 yaw 角度（弧度）
        """
        if cloud_points.shape[0] == 0:
            return  # 保持之前的高度图
        
        # 获取全局查询点
        global_query_points = self._get_global_query_points(torso_pos, torso_yaw)
        
        # 更新每个网格点的高度
        # 高度值 = 机器人 Z - 地面高度（点云的 Z）
        new_heights = np.zeros((GRID_Y_SIDE_LENGTH, GRID_X_SIDE_LENGTH), dtype=np.float32)
        point_idx = 0
        
        for j in range(GRID_Y_SIDE_LENGTH):  # Y 方向（行）
            for i in range(GRID_X_SIDE_LENGTH):  # X 方向（列）
                query_xy = global_query_points[point_idx]
                # 计算高度：机器人 Z - 地面高度（与训练代码一致）
                height = self._interpolate_height_idw(query_xy, cloud_points, torso_pos[2])
                new_heights[j, i] = height
                point_idx += 1
        
        # 线程安全地更新高度图
        with self.lock:
            self.heights = new_heights
    
    def get_heightmap(self) -> np.ndarray:
        """
        获取当前高度图（225 维向量）
        
        @return: numpy array, shape (225,), dtype=np.float32
        """
        with self.lock:
            # 展平为 225 维向量（按行优先顺序）
            return self.heights.flatten().copy()
    
    def get_heightmap_2d(self) -> np.ndarray:
        """
        获取当前高度图（15×15 网格）
        
        @return: numpy array, shape (15, 15), dtype=np.float32
        """
        with self.lock:
            return self.heights.copy()

