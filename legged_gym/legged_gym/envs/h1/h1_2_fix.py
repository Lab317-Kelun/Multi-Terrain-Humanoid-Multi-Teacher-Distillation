# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1_2FixCfg( LeggedRobotCfg ):
    """
    H1_2机器人的环境配置类
    继承自基础的腿式机器人配置，定义H1机器人的特定参数
    """
    
    class init_state( LeggedRobotCfg.init_state ):
        """初始状态配置"""
        pos = [0.0, 0.0, 1.05]  # 机器人初始位置 [x, y, z] (米)
        # 各关节的默认角度设置 (弧度)
        default_joint_angles = {
            # 左腿关节角度
            'left_hip_yaw_joint': 0,        # 左髋偏航关节
            'left_hip_roll_joint': 0,       # 左髋滚转关节
            'left_hip_pitch_joint': -0.16,  # 左髋俯仰关节
            'left_knee_joint': 0.36,        # 左膝关节
            'left_ankle_pitch_joint': -0.2, # 左踝俯仰关节
            'left_ankle_roll_joint': 0.0,   # 左踝滚转关节

            # 右腿关节角度
            'right_hip_yaw_joint': 0,        # 右髋偏航关节
            'right_hip_roll_joint': 0,       # 右髋滚转关节
            'right_hip_pitch_joint': -0.16,  # 右髋俯仰关节
            'right_knee_joint': 0.36,        # 右膝关节
            'right_ankle_pitch_joint': -0.2, # 右踝俯仰关节
            'right_ankle_roll_joint': 0.0,   # 右踝滚转关节

            # 躯干关节
            'torso_joint': 0,               # 躯干关节

            # 左臂关节角度
            'left_shoulder_pitch_joint': 0.4,  # 左肩俯仰关节
            'left_shoulder_roll_joint': 0,     # 左肩滚转关节
            'left_shoulder_yaw_joint': 0,      # 左肩偏航关节
            'left_elbow_pitch_joint': 0.3,     # 左肘俯仰关节

            # 右臂关节角度
            'right_shoulder_pitch_joint': 0.4,  # 右肩俯仰关节
            'right_shoulder_roll_joint': 0,     # 右肩滚转关节
            'right_shoulder_yaw_joint': 0,      # 右肩偏航关节
            'right_elbow_pitch_joint': 0.3,     # 右肘俯仰关节
        }

    class env( LeggedRobotCfg.env ):
        """环境配置参数"""
        num_envs = 512         # 并行仿真环境数量（影响训练速度和内存使用）
        n_scan = 132            # 激光雷达扫描点数量
        n_priv = 3 + 3 + 3      # 特权信息维度：位置(3) + 速度(3) + 其他(3)
        n_priv_latent = 4 + 1 + 12 + 12  # 特权潜在状态维度
        n_proprio = 51          # 本体感受信息维度（关节角度、角速度等）
        history_len = 10        # 历史信息长度
        # 总观测维度 = 本体感受 + 扫描 + 历史 + 潜在状态 + 特权信息
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv 
        num_actions = 12        # 动作维度（12个关节的目标角度）
        env_spacing = 3.        # 环境之间的间距（米）

        contact_buf_len = 100   # 接触信息缓冲区长度

        
    class control( LeggedRobotCfg.control ):
        """控制参数配置"""
        control_type = 'P'      # 控制类型：P(比例控制)
        # 关节刚度系数 - 控制关节的位置跟踪精度
        stiffness = {
            'hip_yaw_joint': 200.,      # 髋偏航关节刚度
            'hip_roll_joint': 200.,     # 髋滚转关节刚度
            'hip_pitch_joint': 200.,    # 髋俯仰关节刚度
            'knee_joint': 300.,         # 膝关节刚度（较高，支撑体重）
            'ankle_pitch_joint': 40.,   # 踝俯仰关节刚度（较低，适应地形）
            'ankle_roll_joint': 40.,    # 踝滚转关节刚度
        }
        # 关节阻尼系数 - 控制关节运动的平滑性
        damping = {
            'hip_yaw_joint': 2.5,       # 髋偏航关节阻尼
            'hip_roll_joint': 2.5,      # 髋滚转关节阻尼
            'hip_pitch_joint': 2.5,     # 髋俯仰关节阻尼
            'knee_joint': 4,            # 膝关节阻尼（较高，稳定性）
            'ankle_pitch_joint': 2.0,   # 踝俯仰关节阻尼
            'ankle_roll_joint': 2.0,    # 踝滚转关节阻尼
        } 
        action_scale = 0.25     # 动作缩放因子（限制动作幅度）
        decimation = 4          # 控制频率降采样（每4个仿真步执行一次控制）

    class asset( LeggedRobotCfg.asset ):
        """机器人资产配置"""
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_fix_arm.urdf'  # URDF文件路径
        name = "h1_2_fix"                    # 机器人名称
        foot_name = "ankle_roll"             # 脚部链接名称
        knee_name = "knee"                   # 膝关节名称
        penalize_contacts_on = ["hip", "knee"]        # 惩罚接触的部位（避免不当接触）
        terminate_after_contacts_on = ["pelvis"]      # 接触后终止episode的部位（躯干着地）
        self_collisions = 1                  # 启用自碰撞检测
        flip_visual_attachments = False      # 不翻转视觉附件

    class domain_rand(LeggedRobotCfg     .domain_rand):
        """域随机化配置 - 恢复原始设置"""
        randomize_friction = True            # 随机化摩擦系数
        friction_range = [0.8, 0.8]         # 恢复原始摩擦系数
        randomize_base_mass = False          # 恢复：不随机化质量
        added_mass_range = [0., 3.]          # 恢复：只增加质量
        randomize_base_com = False           # 不随机化质心位置
        added_com_range = [-0.2, 0.2]       # 质心偏移范围（米）
        push_robots = False                  # 恢复：不推机器人
        push_interval_s = 8                  # 推力间隔（秒）
        max_push_vel_xy = 0.5               # 恢复原始推力

        randomize_motor = False              # 不随机化电机特性
        motor_strength_range = [1., 1.]      # 电机强度范围

        # 动作延迟相关参数
        delay_update_global_steps = 24 * 8000  # 延迟更新的全局步数
        action_delay = False                 # 不启用动作延迟
        action_curr_step = [1, 1]           # 当前动作步数范围
        action_curr_step_scratch = [0, 1]   # 从头训练时的动作步数
        action_delay_view = 1               # 动作延迟视图
        action_buf_len = 8                  # 动作缓冲区长度

    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        resampling_time = 1.0         # 命令重采样时间间隔（秒）
        heading_command = True         # 启用朝向命令模式
        ang_vel_clip = 0.05            # 角速度命令死区阈值
        lin_vel_clip = 0.1            # 线速度命令死区阈值
        
        # 策略1：智能速度生成配置
        height_adaptive_speed = True   # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4  # 地形复杂度权重
        speed_gradient_weight = 0.4   # 高度梯度权重  
        speed_roughness_weight = 0.2  # 地形粗糙度权重
        
        # 命令范围配置
        class ranges( LeggedRobotCfg.commands.ranges ):
            """命令范围设置"""
            lin_vel_x = [0.1, 1.2]     # 前进速度范围（m/s）      随机前进速度 目标朝向 目标角速度
            lin_vel_y = [0.0, 0.0]     # 侧向速度范围（设为0，只前进）
            ang_vel_yaw = [0, 0]       # 偏航角速度范围（设为0，直线行走）
            heading = [-1.2, 1.2]      # 朝向角度范围

        class max_ranges( LeggedRobotCfg.commands.max_ranges ):
            """最大命令范围（课程学习后期或固定模式）"""
            lin_vel_x = [0.3, 1.5]     # 前向速度范围 [m/s]
            lin_vel_y = [0.0, 0.0]    # 侧向速度范围 [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # 偏航角速度范围 [rad/s]
            heading = [-1.6, 1.6]     # 朝向角度范围

    class rewards:
        """奖励函数配置"""
        class scales:
            """各项奖励的权重系数 - 恢复原始设置"""
            termination = -0.0          # 终止惩罚（设为0）
            tracking_lin_vel = 1.5      # 原:1.0 → 新:1.5 (策略6：提高速度跟踪权重)
            tracking_ang_vel = 1.0      # 原:0.5 → 新:0.8 (策略6：提高速度跟踪权重)
            lin_vel_z = -2.0           # 垂直速度惩罚（避免跳跃）
            ang_vel_xy = -0.05         # 恢复原始权重
            orientation = -0.           # 恢复：不惩罚姿态
            torques = -0.00001         # 恢复原始权重
            dof_vel = -0.              # 恢复：不惩罚关节速度
            dof_acc = -2.5e-7          # 关节加速度惩罚
            action_rate = -0.01        # 动作变化率惩罚
            collision = -1.0           # 碰撞惩罚
            base_height = -0.          # 恢复：不惩罚基座高度
            feet_air_time = 1.0        # 恢复原始权重
            feet_stumble = -0.0        # 恢复：不惩罚绊倒
            stand_still = -0.          # 恢复：不惩罚静止
            
            reach_goal = 2.0           # 到达目标奖励
            heading_tracking = 1.0      # 朝向跟踪奖励
            next_heading_tracking = 1.0  # 下一朝向跟踪奖励
            # bridge_center = 1.0         # 桥上居中奖励
            
            # sin_line = 1.0

        only_positive_rewards = True    # 只使用正奖励（避免早期终止问题）
        tracking_sigma = 0.25          # 跟踪奖励的标准差参数
        feet_air_time_target = 0.25  # 目标腾空时间 (秒)
        soft_dof_pos_limit = 1.        # 关节位置软限制（URDF限制的百分比）
        soft_dof_vel_limit = 1.        # 关节速度软限制
        soft_torque_limit = 1.         # 力矩软限制
        base_height_target = 1.        # 目标基座高度（米）
        max_contact_force = 100.       # 最大接触力（超过此值将被惩罚）
        is_play = False               # 非游戏模式
    


class H1_2FixCfgPPO( LeggedRobotCfgPPO ):
    """
    H1_2机器人的PPO算法配置类
    定义强化学习训练的相关参数
    """
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        """算法参数配置"""
        entropy_coef = 0.01   # 熵系数（鼓励探索）
        
    class runner( LeggedRobotCfgPPO.runner ):
        """训练运行器配置"""
        run_name = ''                    # 运行名称（空字符串使用默认）
        experiment_name = 'h1_2_fix'     # 实验名称

    class estimator(LeggedRobotCfgPPO.estimator):
        """状态估计器配置（用于处理特权信息）"""
        train_with_estimated_states = True    # 使用估计状态进行训练
        learning_rate = 2.e-4                # 学习率
        hidden_dims = [128, 64]              # 隐藏层维度
        priv_states_dim = H1_2FixCfg.env.n_priv      # 特权状态维度
        num_prop = H1_2FixCfg.env.n_proprio          # 本体感受维度
        num_scan = H1_2FixCfg.env.n_scan             # 扫描数据维度




