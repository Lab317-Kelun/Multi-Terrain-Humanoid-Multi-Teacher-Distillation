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
import numpy

class N1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.70]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 左腿 - 使用官方推荐角度
            "left_hip_pitch_joint": -numpy.deg2rad(14.0),
            "left_hip_roll_joint": 0.0,                   
            "left_hip_yaw_joint": 0.0,                      
            "left_knee_pitch_joint": +numpy.deg2rad(29.5),
            "left_ankle_roll_joint": 0.0,
            "left_ankle_pitch_joint": -numpy.deg2rad(13.7),

            # 右腿 - 完全对称
            "right_hip_pitch_joint": -numpy.deg2rad(14.0),
            "right_hip_roll_joint": 0.0,                    
            "right_hip_yaw_joint": 0.0,                    
            "right_knee_pitch_joint": +numpy.deg2rad(29.5),
            "right_ankle_roll_joint": 0.0,
            "right_ankle_pitch_joint": -numpy.deg2rad(13.7),
            
            # 腰部保持直立
            'torso_joint': 0.0
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 512
        n_scan = 132
        n_priv = 3 + 3 + 3 # = 9 base velocity 3个

        n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
        n_proprio = 51 # 所有本体感知信息，即obs_buf
        history_len = 10

        # num obs = 53+132+10*53+43+9 = 187+47+530+43+9 = 816
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 

        contact_buf_len = 100

        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 90,
                     'hip_roll': 120,
                     'hip_pitch': 180,
                     'knee': 120,
                     'ankle': 45,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 8,
                     'hip_roll': 10,
                     'hip_pitch': 10,
                     'knee': 8,
                     'ankle': 2.5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/N1/N1_rotor.urdf'
        name = "N1"
        foot_name = "foot_roll"
        knee_name = "shank"
        penalize_contacts_on = ["thigh", "shank"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        resampling_time = 1.0         # 命令重采样时间间隔（秒）
        heading_command = True         # 启用朝向命令模式
        ang_vel_clip = 0.1            # 角速度命令死区阈值
        lin_vel_clip = 0.1            # 线速度命令死区阈值
        
        # 策略1：智能速度生成配置
        height_adaptive_speed = True   # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4  # 地形复杂度权重
        speed_gradient_weight = 0.4   # 高度梯度权重  
        speed_roughness_weight = 0.2  # 地形粗糙度权重
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.2, 0.8]  # min max [m/s] - 增加速度范围，强制移动
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    class rewards:
        class scales:
            # ========== 抬脚走路核心奖励 ==========
            tracking_lin_vel = 10.0     # 【前进】极强鼓励跟踪速度命令
            tracking_ang_vel = 3.0      # 【转向】鼓励转向
            feet_air_time = 12.0        # 【抬脚】极强鼓励抬脚走路
            
            # ========== 姿态稳定性 ==========
            orientation = -0.5          # 【防倾倒】减少惩罚，允许更多探索
            lin_vel_z = -1.0            # 【垂直稳定】减少惩罚
            ang_vel_xy = -0.02          # 【姿态稳定】减少惩罚
            
            # ========== 动作平滑性 ==========
            torques = -0.00001          # 【能耗】轻微惩罚大力矩
            action_rate = -0.01         # 【平滑控制】惩罚动作剧烈变化
            
            # ========== 强制移动 ==========
            stand_still = -3.0          # 【静止惩罚】强烈惩罚静止不动
            
            # ========== 步态优化 ==========
            feet_distance = 1.0         # 【步幅】鼓励合理脚间距
            feet_clearance = -1.0       # 【脚部高度】轻微惩罚脚部过低
            feet_perpendicular_alignment = 5.0  # 【防外八】适度鼓励正确脚部方向
            
            # ========== 精细步态控制 ==========
            feet_distance_y_too_close = -2.0  # 【防脚太近】惩罚双脚太靠近
            feet_speed_xy_close_to_ground = -1.0  # 【防脚拖地】惩罚脚部在地面滑动
            limits_dof_pos_without_ankle = -0.5  # 【关节限制】惩罚关节超出限制
            limits_dof_vel_without_ankle = -0.3  # 【速度限制】惩罚关节速度过快
            
            # ========== 导航任务 ==========
            reach_goal = 2.0            # 【接近目标】鼓励接近目标
            heading_tracking = 0.5      # 【朝向目标】鼓励朝向目标
            next_heading_tracking = 0.5 # 【预瞄】鼓励预瞄下一个目标

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        feet_air_time_target = 0.08  # 目标腾空时间 (秒) - 更短更容易达到
        min_dist = 0.06  # 最小距离，用于feet_distance奖励计算
        max_dist = 0.2
        target_feet_height = 0.2  # 目标脚部高度，用于feet_clearance奖励计算
        base_height_target = 0.75
        max_contact_force = 300. # forces above this value are penalized
        
        # ========== 精细步态控制参数 ==========
        feet_distance_y_too_close = 0.15  # 双脚Y方向最小距离阈值
        sigma_feet_distance_y_too_close = 10.0  # 脚距太近的惩罚系数
        sigma_feet_speed_xy_close_to_ground = 5.0  # 脚部拖地的惩罚系数
        sigma_limits_dof_pos = 5.0  # 关节位置限制惩罚系数
        sigma_limits_dof_vel = 3.0  # 关节速度限制惩罚系数
        is_play = False


class N1FixCfgPPO( LeggedRobotCfgPPO ):
    # ========== 启用镜像算法 ==========
    runner_class_name = "OnPolicyRunnerMirror"  # 使用镜像版本的Runner
    class_name = "PPOMirror"                     # 使用镜像版本的PPO算法
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'n1_fix_mirror'  # 更新实验名称
        max_iterations = 100001 # number of policy updates
        save_interval = 200
        algorithm_class_name = 'PPOMirror'  # 指定使用PPOMirror算法

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = N1FixCfg.env.n_priv
        num_prop = N1FixCfg.env.n_proprio
        num_scan = N1FixCfg.env.n_scan
    
    # ========== 新增：镜像配置（关键！）==========
    class mirror:
        """
        镜像数据增强配置 - 让机器人学习对称步态
        
        N1机器人关节顺序（12个DOF）：
        0-5: 左腿 [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_roll, ankle_pitch]
        6-11: 右腿 [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_roll, ankle_pitch]
        
        观测值结构（基于humanoid_robot.py第532-546行）：
        0-2: base_ang_vel (3)
        3-4: imu_obs (2) - roll, pitch
        5: 0*delta_yaw (1) - 被置零
        6: delta_yaw (1)
        7: delta_next_yaw (1)
        8-9: 0*commands[:, 0:2] (2) - 被置零
        10: commands[:, 0:1] (1) - x方向速度命令
        11: env_class != 17 (1)
        12: env_class == 17 (1)
        13-24: dof_pos (12) - 关节位置
        25-36: dof_vel (12) - 关节速度
        37-48: action_history_buf (12) - 上一次动作
        49-50: contact_filt (2) - 脚部接触
        51+: heights, priv, history等
        
        镜像规则：
        - Pitch关节保持不变(系数=1.0)
        - Roll/Yaw关节取反(系数=-1.0)
        - 左右腿对应关节互换
        """
        enable_mirror = True  # ✅ 启用镜像训练
        
        # 观测值系数（哪些需要取反）
        observations_coefficient = numpy.array([
            # base_ang_vel (0-2): roll取反, pitch保持, yaw取反
            -1.0, 1.0, -1.0,
            # imu_obs (3-4): roll取反, pitch保持 (按官方标准)
            -1.0, 1.0,
            # delta_yaw相关 (5-7): 取反
            -1.0, -1.0, -1.0,
            # commands相关 (8-10): x保持，y取反，yaw取反 (按官方标准)  
            1.0, -1.0, -1.0,
            # env_class (11-12): 保持
            1.0, 1.0,
            # dof_pos (13-24): 左腿(13-18) + 右腿(19-24)
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 左腿：pitch保持，roll/yaw取反
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 右腿：pitch保持，roll/yaw取反
            # dof_vel (25-36): 左腿(25-30) + 右腿(31-36)
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 左腿速度
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 右腿速度
            # action_history (37-48): 与关节系数一致
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 左腿历史动作
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 右腿历史动作
            # contact (49-50): 保持
            1.0, 1.0,
        ] + [1.0] * 1000)  # 其余观测保持不变（heights, priv, history等）
        
        # 观测值交换（左右腿对应位置互换）
        observations_exchange = numpy.array([
            # dof_pos交换：左腿(13-18) <-> 右腿(19-24)
            (13, 19), (14, 20), (15, 21), (16, 22), (17, 23), (18, 24),
            # dof_vel交换：左腿(25-30) <-> 右腿(31-36)  
            (25, 31), (26, 32), (27, 33), (28, 34), (29, 35), (30, 36),
            # action_history交换：左腿(37-42) <-> 右腿(43-48)
            (37, 43), (38, 44), (39, 45), (40, 46), (41, 47), (42, 48),
            # contact交换：左脚(49) <-> 右脚(50)
            (49, 50),
        ])
        
        # 动作系数（与dof_pos相同）
        actions_coefficient = numpy.array([
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 左腿：pitch保持，roll/yaw取反
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0,  # 右腿：pitch保持，roll/yaw取反
        ])
        
        # 动作交换（左右腿互换）
        actions_exchange = numpy.array([
            (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),  # 左腿 <-> 右腿
        ])

