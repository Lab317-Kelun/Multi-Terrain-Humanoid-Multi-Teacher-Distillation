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

# align with unitree code
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.80]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           'torso_joint' : 0.
        }

    class env( LeggedRobotCfg.env ):
        num_envs = 512
        n_scan = 132
        n_priv = 3 + 3 + 3 # = 9 base velocity 3个
        # n_priv_latent = 4 + 1 + 12 +12
        n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
        #n_proprio = 51 # 所有本体感知信息，即obs_buf
        n_proprio = 45
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
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        foot_name = "ankle_roll"
        knee_name = "knee"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
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
            lin_vel_x = [0.1, 0.6] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [-1.2, 1.2]

    class rewards:
        class scales:
            # termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            # base_height = -10.0
            orientation = -1.25
            # lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -2.5e-6
            action_rate = -0.01
            # smoothness = -1e-3
            # stand_still = -0.05
            dof_vel = -1e-4
            dof_acc = -2.5e-8
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            dof_power = -2e-5
            feet_ground_parallel = -0.02
            feet_distance = -2.0
            feet_air_time =  2.5
            feet_clearance = -1.0 
            # feet_parallel = -2.0
            feet_forward_alignment = 1.5
            # alive = 0.15
            # hip_pos = -1.0
            # feet_swing_height = -20.0
            # contact = 0.18
            # collision = -0.
            
            reach_goal = 2.0
            heading_tracking = 1.0
            next_heading_tracking = 1.0
            

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early   problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        feet_air_time_target = 0.25  # 目标腾空时间 (秒)
        min_dist = 0.08  # 最小距离，用于feet_distance奖励计算
        max_dist = 0.25  # 最大距离，用于feet_distance奖励计算
        target_feet_height = 0.1  # 目标脚部高度，用于feet_clearance奖励计算
        base_height_target = 0.7
        max_contact_force = 100. # forces above this value are penalized
        is_play = False

class G1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'g1_fix'
        max_iterations = 100001 # number of policy updates
        save_interval = 200

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1FixCfg.env.n_priv
        num_prop = G1FixCfg.env.n_proprio
        num_scan = G1FixCfg.env.n_scan

