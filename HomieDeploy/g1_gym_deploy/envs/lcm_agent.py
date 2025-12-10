"""
LCM 代理类
功能：通过 LCM 与 C++ 控制程序（g1_control.cpp）通信
- 接收机器人状态（关节位置、速度、身体姿态等）
- 构建策略网络的观测向量
- 发布策略网络的关节命令给 C++ 程序

LCM 消息发布：
- "pd_plustau_targets"：发布 29 个关节的目标位置和力矩
  * 前 12 个关节（索引 0-11）：来自策略网络输出（左右腿各 6 个）
  * 索引 12-28：当前为 0（腰部 3 个 + 手臂 14 个未使用）
"""

import time

import lcm
import numpy as np
import torch
from utils.cheetah_state_estimator import StateEstimator
from lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from lcm_types.arm_action_lcmt import arm_action_lcmt
from utils.command_profile import RCControllerProfile
from lcm_types.body_record_lcmt import body_record_lcmt

# ROS1 相关导入（可选，如果未安装 ROS1 则跳过）
try:
    import rospy
    from utils.heightmap_subscriber import HeightMapSubscriber
    ROS1_AVAILABLE = True
except ImportError:
    ROS1_AVAILABLE = False
    rospy = None
    HeightMapSubscriber = None

# 初始化 LCM 通信（UDP 多播，与 C++ 程序通信）
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


class LCMAgent():
    def __init__(self, se:StateEstimator, command_profile: RCControllerProfile):
        """
        初始化 LCM 代理
        
        @param se 状态估计器（从 LCM 接收机器人状态）
        @param command_profile 命令配置文件（处理速度指令等）
        """
        self.se = se
        self.command_profile = command_profile

        # 控制周期：50Hz（20ms）
        self.dt = 1/50
        self.timestep = 0  # 当前时间步

        # 环境配置
        self.num_envs = 1              # 环境数量（单机器人）
        self.num_dofs = 27             # 总关节数（27个：腿12 + 腰3 + 臂12）
        # 观测维度：动态调整（45 或 48 维 proprio）
        # - 如果收到目标点命令：48 维（包含 delta_yaw, delta_pose_x, delta_pose_y）
        # - 否则：45 维（不包含这三个观测）
        # 完整观测（包含历史）：302 + 10*45 = 752 或 305 + 10*48 = 785
        self.use_target_command = False  # 是否使用目标点命令（动态启用三个观测）
        self.num_obs = 45 + 225 + 3 + 29  # 单步观测维度：302（高度图是 15×15 = 225），如果启用目标点则为 305
        self.num_history_length = 10   # 历史长度（用于 HistoryWrapper，与 MuJoCo 一致）
        self.num_lower_dofs = 12       # 下肢关节数（左右腿各 6 个）
        self.num_commands = 4          # 命令维度：[vx, vy, vyaw, height]
        self.device = 'cuda:0'        # 计算设备（GPU）

        # 默认关节位置（27个关节的标称位置，单位：rad）
        # 顺序：左腿(6) + 右腿(6) + 腰部(3) + 左臂(6) + 右臂(6)
        self.default_dof_pos = np.array([-0.1000,  0.0000,  0.0000,  0.3000, -0.2000,  0.0000, -0.1000,  0.0000,
         0.0000,  0.3000, -0.2000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000], dtype=np.float32)
        
        # PD 控制增益：位置增益（刚度，单位：N·m/rad）
        # 顺序：左腿(6) + 右腿(6) + 腰部(3) + 左臂(6) + 右臂(6) = 27
        self.p_gains = np.array([150., 150., 150., 300.,  40.,  40., 150., 150., 150., 300.,  40.,  40., 300., 200., 200., 200., 100.,  20.,  20.,  20., 200., 200., 200., 100., 20.,  20.,  20.], dtype=np.float)
        
        # PD 控制增益：速度增益（阻尼，单位：N·m·s/rad）
        self.d_gains = np.array([2.0000, 2.0000, 2.0000, 4.0000, 4.0000, 4.0000, 2.0000, 2.0000, 2.0000, 4.0000, 4.0000, 4.0000, 5.0000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000,0.5000, 0.5000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000], dtype=np.float)

        # 力矩限制（单位：N·m）
        # 顺序：左腿(6) + 右腿(6) + 腰部(3) + 左臂(6) + 右臂(6) = 27
        self.torque_limit = np.array([ 88.,  88.,  88., 139.,  50.,  50.,  88.,  88.,  88., 139.,  50.,  50.,
         88.,  25.,  25.,  25.,  25.,  25.,   5.,   5.,  25.,  25.,  25.,  25.,
         25.,   5.,   5.])
        
        # 状态变量
        self.commands = np.zeros((1, self.num_commands), dtype=np.float32)      # 速度命令 [vx, vy, vyaw, height]
        self.actions = torch.zeros(self.num_lower_dofs)      # 当前动作（12个，来自策略网络）
        self.last_actions = torch.zeros(self.num_lower_dofs) # 上一时刻动作（用于历史）
        self.gravity_vector = np.zeros(3, dtype=np.float32)                     # 重力向量（在身体坐标系中）
        self.dof_pos = np.zeros(self.num_dofs, dtype=np.float32)                # 当前关节位置（27个）
        self.dof_vel = np.zeros(self.num_dofs, dtype=np.float32)                # 当前关节速度（27个）
        self.body_angular_vel = np.zeros(3, dtype=np.float32)                  # 身体角速度（roll, pitch, yaw）
        self.joint_pos_target = np.zeros(self.num_dofs + 2, dtype=np.float32)  # 目标关节位置（29个，用于 LCM 消息）
        self.torques = np.zeros(self.num_dofs + 2, dtype=np.float32)            # 前馈力矩（29个，当前全为 0）

        self.joint_idxs = self.se.joint_idxs  # 关节索引映射
        
        # 高度图订阅器（ROS1，可选）
        # 注意：rospy 节点应该在主程序入口（deploy_policy.py）初始化，这里不再重复初始化
        self.heightmap_subscriber = None
        if ROS1_AVAILABLE:
            try:
                # 创建高度图订阅器（假设 rospy 节点已在主程序中初始化）
                self.heightmap_subscriber = HeightMapSubscriber()
                
                # 启用 StateEstimator 使用 TF 获取位置
                self.se.enable_tf_position(
                    self.heightmap_subscriber.tf_buffer,
                    self.heightmap_subscriber.tf_listener
                )
                
                print("HeightMapSubscriber initialized successfully.")
            except Exception as e:
                print(f"Warning: Failed to initialize HeightMapSubscriber: {e}")
                print("Heightmap will use default values (robot Z coordinate).")
                self.heightmap_subscriber = None
        else:
            print("Warning: ROS1 not available. Heightmap will use default values.")


    def get_obs(self):
        """
        获取当前观测向量（供策略网络使用）
        
        观测向量组成：
        1. 本体感受观测（45维）：
           - commands(3) * [2.0, 2.0, 0.5]
           - ang_vel(3) * 0.5
           - [已注释] delta_yaw(1) * 0.5
           - [已注释] delta_pose_x(1) * 0.5
           - [已注释] delta_pose_y(1) * 0.5
           - gravity(3)
           - (dof_pos[:12] - default_angles) * 1.0
           - dof_vel[:12] * 0.05
           - action_history(12)
        2. 高度图（225维，15×15）：机器人 Z - 地面高度（与训练代码一致）
        3. 特权信息（3+29=32维）：全为 0
        
        总维度：45 + 225 + 3 + 29 = 302
        
        @return 观测张量（形状：[1, 302]），在 GPU 上
        """
        # 从状态估计器获取数据
        self.gravity_vector = self.se.get_gravity_vector()
        cmds = self.command_profile.get_command(self.timestep * self.dt)
        self.commands[:, :] = cmds[:self.num_commands]
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()
        base_pos = self.se.get_base_pos()
        
        # 检查是否收到目标点命令
        self.use_target_command = self.se.has_target_command()
        
        # 根据是否使用目标点命令动态调整观测维度
        if self.use_target_command:
            proprio_dim = 48  # 包含 delta_yaw, delta_pose_x, delta_pose_y
        else:
            proprio_dim = 45  # 不包含这三个观测
        
        # 构建本体感受观测（动态维度：45 或 48 维）
        proprio_obs = np.zeros(proprio_dim, dtype=np.float32)
        idx = 0
        
        # commands(3) * cmd_scale [2.0, 2.0, 0.5]
        proprio_obs[idx:idx+3] = self.commands[0, :3] * np.array([2.0, 2.0, 0.5])
        idx += 3
        
        # ang_vel(3) * ang_vel_scale (0.5)
        proprio_obs[idx:idx+3] = self.body_angular_vel * 0.5
        idx += 3
        
        # ========== 动态观测：delta_yaw, delta_pose_x, delta_pose_y ==========
        # 如果收到目标点命令（pedal_command），启用这三个观测
        # 注意：观测顺序与训练代码一致：delta_yaw, delta_pose_x, delta_pose_y
        # 参考训练代码：humanoid_robot.py 的 compute_observations() 方法（第745-850行）
        if self.use_target_command:
            # 获取目标点和当前位置
            target_point = self.se.get_target_point()  # [target_x, target_y]（只包含位置）
            current_pos = base_pos[:2]  # [x, y]
            current_yaw = self.se.get_rpy()[2]  # yaw
            
            # 计算到目标点的方向向量
            target_vec = target_point - current_pos
            distance = np.linalg.norm(target_vec)
            
            # 归一化目标向量（参考训练代码第384行）
            if distance > 1e-5:
                target_vec_norm = target_vec / distance
            else:
                target_vec_norm = np.array([1.0, 0.0])  # 默认方向
            
            # 计算目标朝向（参考训练代码第385行）
            target_yaw = np.arctan2(target_vec_norm[1], target_vec_norm[0])
            
            # delta_yaw(1) * delta_yaw_scale (0.5)
            # 参考训练代码第747行：delta_yaw = wrap_to_pi(self.commands[:, 3] - self.yaw)
            # 其中 commands[:, 3] 是目标朝向（target_yaw）
            # wrap_to_pi 实现：np.arctan2(np.sin(angle), np.cos(angle))
            delta_yaw = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
            proprio_obs[idx] = delta_yaw * 0.5  # 参考训练代码第783行：noisy_delta_yaw * obs_scales.delta_yaw
            idx += 1
            
            # delta_pose_x(1) * delta_pose_scale (0.5)
            # 参考训练代码第749行：delta_pose_x = self.cur_goals[:, 0] - self.root_states[:, 0]
            delta_pose_x = target_point[0] - current_pos[0]
            proprio_obs[idx] = delta_pose_x * 0.5  # 参考训练代码第784行：noisy_delta_pose_x * obs_scales.delta_pose_x
            idx += 1
            
            # delta_pose_y(1) * delta_pose_scale (0.5)
            # 参考训练代码第750行：delta_pose_y = self.cur_goals[:, 1] - self.root_states[:, 1]
            delta_pose_y = target_point[1] - current_pos[1]
            proprio_obs[idx] = delta_pose_y * 0.5  # 参考训练代码第785行：noisy_delta_pose_y * obs_scales.delta_pose_y
            idx += 1
        
        # gravity(3) - 无 scale（训练代码中直接使用）
        proprio_obs[idx:idx+3] = self.gravity_vector
        idx += 3
        
        # (dof_pos[:12] - default_dof_pos[:12]) * dof_pos_scale (1.0)
        proprio_obs[idx:idx+12] = (self.dof_pos[:12] - self.default_dof_pos[:12]) * 1.0
        idx += 12
        
        # dof_vel[:12] * dof_vel_scale (0.05)
        proprio_obs[idx:idx+12] = self.dof_vel[:12] * 0.05
        idx += 12
        
        # action_history(12)
        proprio_obs[idx:idx+12] = self.actions.cpu().numpy()
        idx += 12
        
        # 高度图（225维，15×15）：从 ROS1 话题获取真实高度图，如果失败则使用默认值
        # 高度值 = 机器人 Z - 地面高度（与训练代码一致）
        if self.heightmap_subscriber is not None:
            try:
                heights = self.heightmap_subscriber.get_heightmap()
                if heights is None or heights.shape[0] != 225:
                    # 如果高度图未准备好或尺寸不对，使用默认值（机器人 Z 坐标）
                    heights = np.full(225, base_pos[2], dtype=np.float32)
            except Exception as e:
                # 异常处理：使用默认值
                if ROS1_AVAILABLE:
                    rospy.logwarn_throttle(1.0, f"Failed to get heightmap: {e}. Using default.")
                heights = np.full(225, base_pos[2], dtype=np.float32)
        else:
            # ROS1 不可用，使用默认值
            heights = np.full(225, base_pos[2], dtype=np.float32)
        
        # 特权信息（3+29=32维）：全为 0
        priv_explicit = np.zeros(3, dtype=np.float32)
        priv_latent = np.zeros(29, dtype=np.float32)
        
        # 动态更新观测维度
        if self.use_target_command:
            self.num_obs = 48 + 225 + 3 + 29  # 305 维（包含目标点相关观测）
        else:
            self.num_obs = 45 + 225 + 3 + 29  # 302 维（不包含目标点相关观测）
        
        # 拼接完整观测
        ob = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent], axis=0)
        
        return torch.tensor(ob, device=self.device).float().unsqueeze(0)

    def publish_action(self, action, hard_reset=False):
        """
        发布策略网络的关节命令给 C++ 程序
        
        功能：
        1. 将策略网络的输出（12个动作）转换为关节目标位置
        2. 通过 LCM 发布给 C++ 程序（g1_control.cpp）
        
        发布的消息：
        - 通道："pd_plustau_targets"
        - 内容：29 个关节的目标位置和力矩
          * 索引 0-11：来自策略网络的 12 个动作（左右腿各 6 个）
          * 索引 12-28：当前为 0（腰部 3 个 + 手臂 14 个未使用）
        
        @param action 策略网络输出的动作（12个，对应左右腿各 6 个关节）
        @param hard_reset 是否硬重置（未使用）
        """
        # 将动作从 GPU 转到 CPU，并转换为 numpy 数组
        action = action.cpu().numpy()
        
        # 创建 LCM 消息对象
        command_for_robot = pd_tau_targets_lcmt()
        
        # 将策略输出转换为关节目标位置（与 MuJoCo 版本一致）
        # 策略输出是相对于默认位置的增量，需要缩放（0.25）后加上默认位置
        action_scale = 0.25
        scaled_pos_target = action * action_scale + self.default_dof_pos[:12]
        
        # 注释掉的代码：计算 PD 控制力矩（当前未使用，直接发送位置命令）
        # torques = (scaled_pos_target - self.dof_pos[:12]) * self.p_gains[:12]  - self.dof_vel[:12] * self.d_gains[:12]   
        # torques = np.clip(torques[:12], -self.torque_limit[:12], self.torque_limit[:12])
        
        # 填充前 12 个关节的目标位置（左右腿各 6 个，来自策略网络）
        self.joint_pos_target[:12] = scaled_pos_target[:12]
        
        # 注释掉的代码：手臂和腰部的控制（当前未使用）
        # arm_actions = self.se.get_arm_action()
        # self.joint_pos_target[15:] = 0.#arm_actions
        # self.joint_pos_target[12] = scaled_pos_target[12] # waist
        # self.joint_pos_target[15:] = scaled_pos_target[13:]
        # self.torques[:12] = torques[:12]
        # self.torques[15:] = torques[13:] 
        
        # 填充 LCM 消息
        command_for_robot.q_des = self.joint_pos_target      # 29 个关节的目标位置（前12个来自策略，其余为0）
        command_for_robot.tau_ff = self.torques               # 29 个关节的前馈力矩（当前全为 0）
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)  # 时间戳（微秒）

        # 通过 LCM 发布关节命令（C++ 程序订阅 "pd_plustau_targets" 通道）
        lc.publish("pd_plustau_targets", command_for_robot.encode())

        # 注释掉的代码：发布手臂动作（当前未使用）
        # arm_action = arm_action_lcmt()
        # arm_action.act = arm_actions
        # lc.publish("new_arm_action", arm_action.encode())

    def reset(self):
        """
        重置环境（开始新的回合）
        
        @return 初始观测向量
        """
        self.actions = torch.zeros(12)  # 重置动作为零
        self.time = time.time()         # 重置时间
        self.timestep = 0               # 重置时间步
        return self.get_obs()


    def step(self, actions, hard_reset=False):
        """
        执行一步控制循环
        
        流程：
        1. 裁剪动作（防止过大）
        2. 发布动作给 C++ 程序
        3. 等待控制周期（50Hz = 20ms）
        4. 获取新的观测
        
        @param actions 策略网络输出的动作（形状：[1, 12]）
        @param hard_reset 是否硬重置（未使用）
        @return 新的观测向量
        """
        # 动作裁剪阈值（与 MuJoCo 版本一致：clip_actions / action_scale = 1.2 / 0.25 = 4.8）
        action_scale = 0.25
        clip_actions = 1.2 / action_scale  # 4.8
        
        # 保存上一时刻的动作（用于历史）
        self.last_actions = self.actions[:]
        
        # 裁剪动作（与训练环境一致）
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        
        # 发布动作给 C++ 程序
        self.publish_action(self.actions, hard_reset=hard_reset)
        
        # 等待控制周期（50Hz = 20ms），确保实时性
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        
        # 每 100 步打印一次控制频率
        if self.timestep % 100 == 0: 
            print(f'frq: {1 / (time.time() - self.time)} Hz')
        
        # 更新时间
        self.time = time.time()
        
        # 获取新的观测
        obs = self.get_obs()

        # 更新时间步
        self.timestep += 1
        return obs
