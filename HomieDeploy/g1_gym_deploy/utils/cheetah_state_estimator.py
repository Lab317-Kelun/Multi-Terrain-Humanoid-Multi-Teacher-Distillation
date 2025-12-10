"""
状态估计器模块
功能：从 LCM 接收机器人状态数据（IMU、关节状态、遥控器命令），并提供给策略网络使用

LCM 消息订阅：
- "state_estimator_data": IMU 数据（欧拉角、角速度、加速度）
- "body_control_data": 关节状态（位置、速度）
- "rc_command": 遥控器命令（摇杆、按钮）
- "pedal_command": 踏板命令（速度指令）
"""

import math
import select
import threading
import time

import numpy as np

from lcm_types.body_control_data_lcmt import body_control_data_lcmt
from lcm_types.rc_command_lcmt import rc_command_lcmt
from lcm_types.state_estimator_lcmt import state_estimator_lcmt
from lcm_types.arm_action_lcmt import arm_action_lcmt
from lcm_types.command_lcmt import command_lcmt
import lcm
import os

def get_rpy_from_quaternion(q):
    """
    从四元数计算欧拉角（roll, pitch, yaw）
    
    @param q 四元数 [w, x, y, z]
    @return 欧拉角 [roll, pitch, yaw]（单位：弧度）
    """
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))  # roll（横滚）
    p = np.arcsin(2 * (w * y - z * x))                                # pitch（俯仰）
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))  # yaw（偏航）
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    从欧拉角计算旋转矩阵
    
    @param rpy 欧拉角 [roll, pitch, yaw]（单位：弧度）
    @return 旋转矩阵（3x3），从世界坐标系到身体坐标系
    """
    r, p, y = rpy
    # X 轴旋转矩阵（绕 X 轴旋转 roll）
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    # Y 轴旋转矩阵（绕 Y 轴旋转 pitch）
    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    # Z 轴旋转矩阵（绕 Z 轴旋转 yaw）
    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    # 组合旋转矩阵：R = R_z * R_y * R_x（ZYX 顺序）
    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot


class StateEstimator:
    """
    状态估计器类
    功能：从 LCM 接收机器人状态数据，并提供给策略网络使用
    
    订阅的 LCM 通道：
    - "state_estimator_data": IMU 数据（来自 C++ 程序）
    - "body_control_data": 关节状态（来自 C++ 程序）
    - "rc_command": 遥控器命令（来自 C++ 程序）
    - "pedal_command": 踏板命令（速度指令）
    """
    def __init__(self, lc):
        """
        初始化状态估计器
        
        @param lc LCM 实例，用于订阅和接收消息
        """
        # 关节索引映射：从 C++ 顺序转换为 Isaac Gym 顺序
        # C++ 顺序：29 个关节（腿12 + 腰3 + 臂14）
        # Isaac Gym 顺序：27 个关节（跳过索引 13, 14，因为腰部可能被锁定）
        self.joint_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

        self.lc = lc
        self.num_dofs = 27  # 关节数量（27 个：腿12 + 腰3 + 臂12）
        
        # 状态变量
        self.joint_pos = np.zeros(self.num_dofs+2)  # 关节位置（29 个，包含 C++ 的完整索引）
        self.joint_vel = np.zeros(self.num_dofs+2)  # 关节速度（29 个）
        self.arm_actions = np.zeros(14)             # 手臂动作（14 个：左臂7 + 右臂7）
        self.euler = np.zeros(3)                    # 欧拉角 [roll, pitch, yaw]
        self.R = np.eye(3)                          # 旋转矩阵（从世界坐标系到身体坐标系）
        self.buf_idx = 0                            # 缓冲区索引（用于平滑）
        self.imu_ang_vel = np.zeros(3)              # IMU 角速度（未使用，直接使用 body_ang_vel）
        self.base_pos = np.zeros(3)                 # 机器人位置 [x, y, z]（世界坐标系）
        self.use_tf_for_position = False           # 是否使用 TF 获取位置（需要 ROS1）
        self.tf_buffer = None                       # TF 缓冲区（如果使用 ROS1）
        self.tf_listener = None                     # TF 监听器（如果使用 ROS1）
        
        # 遥控器状态
        self.left_stick = [0, 0]                    # 左摇杆 [x, y]
        self.right_stick = [0, 0]                   # 右摇杆 [x, y]
        self.right_lower_right_switch = 0           # R2 按钮当前状态
        self.right_lower_right_switch_pressed = 0   # R2 按钮是否被按下（边沿检测）

        # 时间相关
        self.init_time = time.time()                # 初始化时间
        self.received_first_bodydate = False        # 是否已收到第一个身体数据

        # 订阅 LCM 消息
        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)      # IMU 数据
        self.bodydate_state_subscription = self.lc.subscribe("body_control_data", self._bodydata_cb)  # 关节状态
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)  # 遥控器命令
        self.pedal_command_subscription = self.lc.subscribe("pedal_command", self._pedal_command_cb)  # 踏板命令

        # 平滑相关（当前未使用）
        self.body_quat = np.array([0, 0, 0, 1])     # 身体四元数（未使用）
        self.smoothing_ratio = 0.2                  # 平滑系数
        self.body_ang_vel = np.zeros(3)             # 身体角速度 [roll_rate, pitch_rate, yaw_rate]
        self.smoothing_length = 12                  # 平滑历史长度
        self.dt_history = np.zeros((self.smoothing_length, 1))  # 时间间隔历史
        self.euler_prev = np.zeros(3)               # 上一时刻的欧拉角
        self.deuler_history = np.zeros((self.smoothing_length, 3))  # 欧拉角变化历史
        self.timeuprev = time.time()                # 上一时刻的时间
        
        # 速度命令 [vx, vy, vyaw, height]
        self.command = np.zeros(4)
        self.command[3] = 0.74  # 默认高度（单位：米）
        
        # 目标点命令队列系统
        self.target_point_queue = []  # 目标点列表（队列）
        self.current_target_index = 0  # 当前目标点在队列中的索引
        self.target_point = None  # 当前使用的目标点 [x, y]
        self.received_target_command = False  # 是否收到过目标点命令（pedal_command）
        
        # 目标点队列参数
        self.target_reach_threshold = 0.3  # 目标到达阈值（m），距离小于此值认为到达目标
        
        # 用于记录"先转向"状态
        self.need_turn_first = False  # 是否需要先转向（当第一次检测到 heading_error > 30度时设置）
        self.last_target_point = None  # 上一次的目标点（用于检测目标点是否改变）
        

    def get_gravity_vector(self):
        """
        获取重力向量（在身体坐标系中）
        
        @return 重力向量 [gx, gy, gz]（单位：m/s²，归一化后方向）
        世界坐标系中重力为 [0, 0, -1]，通过旋转矩阵转换到身体坐标系
        """
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

    def get_rpy(self):
        """
        获取欧拉角
        
        @return 欧拉角 [roll, pitch, yaw]（单位：弧度）
        """
        return self.euler

    def get_command(self):
        """
        获取速度命令
        
        @return 速度命令 [vx, vy, vyaw, height]
          - vx: 前进速度（m/s）
          - vy: 侧向速度（m/s）
          - vyaw: 偏航角速度（rad/s）
          - height: 目标高度（m）
        
        优先级：
        1. 如果收到过目标点命令（pedal_command），根据目标点和当前位置计算速度命令
        2. 否则，从摇杆计算命令
        
        注意：如果收到目标点命令，会启用 delta_yaw, delta_pose_x, delta_pose_y 三个观测
        """
        # 如果收到过目标点命令，根据目标点和当前位置计算速度命令
        if self.received_target_command:
            # 更新当前目标点
            self.target_point = np.array(self.target_point_queue[self.current_target_index], dtype=np.float32)
            
            # 获取当前位置和朝向
            current_pos = self.base_pos[:2]
            current_yaw = self.euler[2]
            
            # 计算到目标点的距离
            target_vec = self.target_point - current_pos
            distance = np.linalg.norm(target_vec)
            
            # 如果到达当前目标点，切换到下一个
            if distance < self.target_reach_threshold:
                self.current_target_index += 1
                # 限制索引在有效范围内（越界时使用最后一个目标点）
                if self.current_target_index >= len(self.target_point_queue):
                    self.current_target_index = len(self.target_point_queue) - 1

            target_vec_norm = target_vec / (distance + 1e-5)
            target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])
            heading_error = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
            
            cmd_yaw = np.clip(0.8 * heading_error, -0.5, 0.5)
            
            if np.abs(heading_error) > np.pi/6 and not self.need_turn_first:
                self.need_turn_first = True
            
            if self.need_turn_first:
                if np.abs(heading_error) < 0.15:
                    cmd_x, cmd_y = 0.5, 0.0
                    self.need_turn_first = False
                else:
                    cmd_x, cmd_y = 0.0, 0.0
            else:
                cmd_x, cmd_y = 0.5, 0.0
            
            cmd_height = 0.74
            
            return np.array([cmd_x, cmd_y, cmd_yaw, cmd_height], dtype=np.float32)
        
        # 如果没有目标点命令，从摇杆计算速度命令
        return self._get_joystick_command()
    
    def _get_joystick_command(self):
        """
        从摇杆计算速度命令（内部方法）
        
        @return 速度命令 [vx, vy, vyaw, height]
        """
        # 系数确定方式：
        # 1. 根据训练时的命令范围（训练代码中的 lin_vel_x, lin_vel_y, ang_vel_yaw 范围）
        # 2. 根据机器人的实际运动能力（最大安全速度）
        # 3. 根据操作体验（摇杆满量程对应合理的最大速度）
        # 摇杆输入范围：[-1, 1]，映射到实际速度命令
        cmd_x = 0.8 * self.left_stick[1]      # 前进速度：左摇杆 Y 轴（向上推为正，向下推为负）
                                              # 系数 0.8：最大前进速度 0.8 m/s（摇杆满量程时）
        cmd_y = -0.4 * self.left_stick[0]     # 侧向速度：左摇杆 X 轴（向右推为正）
                                              # 系数 -0.4：最大侧向速度 0.4 m/s（负号用于方向映射）
        cmd_yaw = -0.4 * self.right_stick[0]  # 偏航角速度：右摇杆 X 轴（向右推为正）
                                              # 系数 -0.4：最大偏航角速度 0.4 rad/s（约 23°/s）
        cmd_height = 0.74                      # 目标高度：固定为 0.74m（不再使用右摇杆 Y 轴）
                                              # 固定值：与训练时的 base_height_target 一致
        
        return np.array([cmd_x, cmd_y, cmd_yaw, cmd_height], dtype=np.float32)

    def get_buttons(self):
        """
        获取按钮状态
        
        @return R2 按钮当前状态（0 或 1）
        """
        return self.right_lower_right_switch

    def get_dof_pos(self):
        """
        获取关节位置（按 Isaac Gym 顺序）
        
        @return 关节位置数组（27 个），已通过 joint_idxs 重新排序
        """
        return self.joint_pos[self.joint_idxs]

    def get_dof_vel(self):
        """
        获取关节速度（按 Isaac Gym 顺序）
        
        @return 关节速度数组（27 个），已通过 joint_idxs 重新排序
        """
        return self.joint_vel[self.joint_idxs]

    def get_yaw(self):
        """
        获取偏航角
        
        @return 偏航角 yaw（单位：弧度）
        """
        return self.euler[2]

    def get_body_angular_vel(self):
        """
        获取身体角速度
        
        @return 身体角速度 [roll_rate, pitch_rate, yaw_rate]（单位：rad/s）
        
        注意：当前直接返回从 IMU 接收的值，注释掉的代码是平滑处理（未使用）
        """
        # 注释掉的代码：平滑处理（未使用）
        # self.body_ang_vel = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (1 - self.smoothing_ratio) * self.body_ang_vel
        return self.body_ang_vel

    def get_base_pos(self):
        """
        获取机器人位置（世界坐标系）
        
        如果 use_tf_for_position=True，则通过 TF 获取真实位置
        否则返回 LCM 消息中的位置（可能为 0）
        
        @return 机器人位置 [x, y, z]（单位：米）
        """
        # 如果使用 TF 获取位置，且 TF 已初始化
        if self.use_tf_for_position and self.tf_buffer is not None:
            try:
                import rospy
                transform = self.tf_buffer.lookup_transform(
                    'odom_corrected',  # 目标坐标系
                    'torso_link',      # 源坐标系
                    rospy.Time(0),
                    rospy.Duration(0.1)
                )
                self.base_pos = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ], dtype=np.float32)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                # TF 查询失败，使用默认值或 LCM 消息中的值
                # 不打印警告，因为可能频繁失败（TF 未初始化等）
                pass
            except Exception as e:
                # 其他异常（如 rospy 未导入）
                pass
        
        return self.base_pos
    
    def get_target_point(self):
        """获取当前目标点 [target_x, target_y]，如果没有则返回 None"""
        return self.target_point

    
    def get_target_queue_info(self):
        """
        获取目标点队列信息（用于调试）
        
        @return 字典，包含队列信息
        """
        return {
            'queue_length': len(self.target_point_queue),
            'current_index': self.current_target_index,
            'current_target': self.target_point.tolist() if self.target_point is not None else None,
            'all_targets': self.target_point_queue.copy()
        }
    
    def enable_tf_position(self, tf_buffer, tf_listener):
        """
        启用通过 TF 获取机器人位置
        
        @param tf_buffer: tf2_ros.Buffer 对象
        @param tf_listener: tf2_ros.TransformListener 对象
        """
        self.use_tf_for_position = True
        self.tf_buffer = tf_buffer
        self.tf_listener = tf_listener

    def get_arm_action(self):
        """
        获取手臂动作
        
        @return 手臂动作数组（14 个：左臂7 + 右臂7）
        """
        return self.arm_actions

    def _bodydata_cb(self, channel, data):
        """
        LCM 回调函数：处理关节状态数据
        
        @param channel LCM 通道名称
        @param data 消息数据（二进制）
        
        消息来源：C++ 程序发布的 "body_control_data"
        消息内容：关节位置和速度（29 个关节）
        """
        # 标记已收到第一个身体数据
        if not self.received_first_bodydate:
            self.received_first_bodydate = True
            print(f"First body data: {time.time() - self.init_time}")

        # 解码 LCM 消息
        msg = body_control_data_lcmt.decode(data)
        self.joint_pos = np.array(msg.q)   # 关节位置（29 个）
        self.joint_vel = np.array(msg.qd)  # 关节速度（29 个）

    def _arm_action_cb(self, channel, data):
        """
        LCM 回调函数：处理手臂动作数据（当前未使用）
        
        @param channel LCM 通道名称
        @param data 消息数据（二进制）
        """
        msg = arm_action_lcmt.decode(data)
        self.arm_actions = np.array(msg.act)  # 手臂动作（14 个）

    def _imu_cb(self, channel, data):
        """
        LCM 回调函数：处理 IMU 数据
        
        @param channel LCM 通道名称
        @param data 消息数据（二进制）
        
        消息来源：C++ 程序发布的 "state_estimator_data"
        消息内容：欧拉角、角速度、加速度
        """
        # 解码 LCM 消息
        msg = state_estimator_lcmt.decode(data)

        # 更新欧拉角
        self.euler = np.array(msg.rpy)  # [roll, pitch, yaw]

        # 更新旋转矩阵（从世界坐标系到身体坐标系）
        self.R = get_rotation_matrix_from_rpy(self.euler)
        
        # 更新身体角速度
        self.body_ang_vel = np.array(msg.omegaBody)  # [roll_rate, pitch_rate, yaw_rate]
        
        # 更新机器人位置
        self.base_pos = np.array(msg.p)  # [x, y, z]
        
        # 注释掉的代码：平滑处理（未使用）
        # self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        # self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timeuprev
        
        # 更新时间戳和索引
        self.timeuprev = time.time()
        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)  # 保存当前欧拉角，用于下次计算变化量
        
    def _rc_command_cb(self, channel, data):
        """
        LCM 回调函数：处理遥控器命令
        
        @param channel LCM 通道名称
        @param data 消息数据（二进制）
        
        消息来源：C++ 程序发布的 "rc_command"
        消息内容：摇杆位置和按钮状态
        """
        # 解码 LCM 消息
        msg = rc_command_lcmt.decode(data)

        # 边沿检测：检测 R2 按钮是否从 0 变为 1（按下事件）
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        # 更新摇杆和按钮状态
        self.right_stick = msg.right_stick      # 右摇杆 [x, y]
        self.left_stick = msg.left_stick        # 左摇杆 [x, y]
        self.right_lower_right_switch = msg.right_lower_right_switch  # R2 按钮状态

    def _pedal_command_cb(self, channel, data):
        """
        LCM 回调函数：处理目标点命令
        
        @param channel LCM 通道名称
        @param data 消息数据（二进制）
        
        消息来源：外部发布的 "pedal_command"
        消息内容：目标点 [target_x, target_y]（只包含位置，不包含朝向）
        
        功能：
        - 维护目标点队列：收到新目标点直接添加到队列（外部已确保距离 > 0.6m）
        - 机器人到达当前目标点（距离 < 0.3m）后，自动切换到下一个目标点
        - 如果收到此消息，会启用 delta_yaw, delta_pose_x, delta_pose_y 三个观测
        - delta_yaw 和速度命令会根据目标点和当前位置自动计算
        """
        msg = command_lcmt.decode(data)
        new_target = np.array(msg.target, dtype=np.float32)
        self.target_point_queue.append(new_target.tolist())
        self.received_target_command = True

    def poll(self, cb=None):
        """
        LCM 消息轮询函数（在独立线程中运行）
        
        功能：持续监听 LCM 消息，当有消息到达时调用相应的回调函数
        
        @param cb 可选的回调函数（未使用）
        """
        t = time.time()
        try:
            while True:
                timeout = 0.01  # 超时时间：10ms
                # 使用 select 监听 LCM 文件描述符，检查是否有可读数据
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                
                if rfds:
                    # 有消息到达，处理所有待处理的 LCM 消息
                    self.lc.handle()  # 这会触发所有已订阅通道的回调函数
                else:
                    # 没有消息，继续等待
                    continue

        except KeyboardInterrupt:
            # 捕获键盘中断（Ctrl+C），优雅退出
            pass

    def spin(self):
        """
        启动 LCM 消息处理线程
        
        功能：在独立线程中运行 poll() 函数，持续接收和处理 LCM 消息
        这个线程在后台运行，不会阻塞主线程
        """
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()  # 启动线程

    def close(self):
        """
        关闭状态估计器
        
        功能：取消订阅 LCM 消息（当前只取消订阅 body_control_data）
        """
        self.lc.unsubscribe(self.bodydate_state_subscription)


if __name__ == "__main__":
    import lcm

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    se = StateEstimator(lc)
    se.poll()
