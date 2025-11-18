# BEAMDOJO + AMP Humanoid Robot Configuration
# 结合BEAMDOJO双Critic与AMP对抗式动作先验
# AMP用于学习自然的人形运动，BeamDojo用于地形导航

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidBEAMDOJOAMPCfg(LeggedRobotCfg):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.78] # x,y,z [m]
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
            "waist_yaw_joint":0.,
            "waist_roll_joint": 0.,
            "waist_pitch_joint": 0.,
            "left_shoulder_pitch_joint": 0.,
            "left_shoulder_roll_joint": 0.,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 0.,
            "left_wrist_roll_joint": 0.,
            "left_wrist_pitch_joint": 0.,
            "left_wrist_yaw_joint": 0.,
            "left_hand_index_0_joint": 0.,
            "left_hand_index_1_joint": 0.,
            "left_hand_middle_0_joint": 0.,
            "left_hand_middle_1_joint": 0.,
            "left_hand_thumb_0_joint": 0.,
            "left_hand_thumb_1_joint": 0.,
            "left_hand_thumb_2_joint": 0.,
            "right_shoulder_pitch_joint": 0.,
            "right_shoulder_roll_joint": -0.,
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.,
            "right_wrist_roll_joint": 0.,
            "right_wrist_pitch_joint": 0.,
            "right_wrist_yaw_joint": 0.,
            "right_hand_index_0_joint": 0.,
            "right_hand_index_1_joint": 0.,
            "right_hand_middle_0_joint": 0.,
            "right_hand_middle_1_joint": 0.,
            "right_hand_thumb_0_joint": 0.,
            "right_hand_thumb_1_joint": 0.,
            "right_hand_thumb_2_joint": 0.,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_dofs = 27     # 机器人总自由度：全身27个关节
        episode_length_s = 20.0
        
        n_scan = 225
        n_priv = 3
        n_priv_latent = 4 + 1 + 12 + 12  # 潜在状态维度
        n_proprio = 75  # 实际obs_buf维度
        history_len = 10
        
        # AMP相关配置
        enable_amp = True                    # 启用AMP
        num_disc_obs_steps = 10              # discriminator观测的历史步数
        amp_motion_files = ['/home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/data/g1_walk.pkl'] 
        amp_replay_buffer_size = 100000      # AMP replay buffer大小
        
        # 重新计算总观测维度
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_actions = 12  # 12个关节动作
        
        # 启用接触信息
        include_foot_contacts = True
        
    # 课程学习配置
    class curriculum_config:
        success_mode = 'vel_tracking'
        success_rate_mode = 'survival_time'
        success_threshold = 3
        failure_threshold = 2
        survival_time_threshold = 20.0
        survival_success_threshold = 3
        survival_failure_threshold = 2
        velocity_success_threshold = 3
        velocity_failure_threshold = 2
    
    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     "waist": 300,
                     "shoulder": 200,
                     "wrist": 20,
                     "elbow": 100,
                     "hand": 10
                     }
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     "waist": 5,
                     "shoulder": 4,
                     "wrist": 0.5,
                     "elbow": 1,
                     "hand": 2
                     }
        action_scale = 0.25
        decimation = 4    
        hip_reduction = 1.0
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 0.8]
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        
        randomize_actuator_offset = True
        actuator_offset_range = [-0.05, 0.05]
        
        randomize_pd_gains = True
        pd_gain_range = [0.85, 1.15]

        delay_update_global_steps = 24 * 8000
        action_delay = True
        action_curr_step = [1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8
        
        use_random = True
        
        randomize_joint_injection = use_random
        joint_injection_range = [-0.05, 0.05]
        
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]
        
        hand_payload_mass_range = [-0.1, 0.3]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.1, 0.1]
        
        randomize_body_displacement = use_random
        body_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 3.0]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.9, 1.1]
        
        randomize_kd = use_random
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.5
        
        init_upper_ratio = 0.
        delay = use_random
        
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['torso']
        curriculum_joints = []
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']
        left_hip_joints = ['left_hip_roll_joint', "left_hip_pitch_joint", "left_hip_yaw_joint"]
        right_hip_joints = ['right_hip_roll_joint', "right_hip_pitch_joint", "right_hip_yaw_joint"]
        hip_pitch_joints = ['right_hip_pitch_joint', 'left_hip_pitch_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_in_pelvis"
        knee_names = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
        self_collision = 1
        flip_visual_attachments = False
        ankle_sole_distance = 0.02
        
        # AMP关键身体部位（用于discriminator观测）
        key_bodies = ["torso_link", "left_hand_palm_link", "right_hand_palm_link", 
                     "left_ankle_roll_link", "right_ankle_roll_link"]
        
    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        curriculum = True
        resampling_time = 4.0
        heading_command = True
        ang_vel_clip = 0.05
        lin_vel_clip = 0.1
        
        height_adaptive_speed = False
        speed_complexity_weight = 0.4
        speed_gradient_weight = 0.4  
        speed_roughness_weight = 0.2
        
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.8, 1.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-1.0, 1.0]
                        
    class rewards(LeggedRobotCfg.rewards):
        """BEAMDOJO + AMP 奖励配置"""
        class scales:
            # ====== BEAMDOJO任务奖励 ======
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.0
            heading_tracking = 1.0
              
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.5 
            action_rate = -0.01
            
            tracking_base_height = 2.
            deviation_hip_joint = -0.2
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -2.
            feet_air_time = 1.0
            feet_clearance = -3.0
            feet_distance_lateral = 0.5  
            knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0  
            feet_parallel = -0.0
            smoothness = -0.05
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -0.15
            
            foothold = 0.05
            
            # ====== AMP风格奖励 (通过discriminator) ======
            # AMP奖励会在训练循环中由discriminator自动计算
            # 这里不需要手动定义scale
            
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.975
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.74
        max_contact_force = 400.
        least_feet_distance = 0.18
        least_feet_distance_lateral = 0.18
        most_feet_distance_lateral = 0.25
        most_knee_distance_lateral = 0.25
        least_knee_distance_lateral = 0.18
        clearance_height_target = 0.18
        is_play = False
        
        foothold_foot_length = 0.12
        foothold_foot_width = 0.06
        foothold_height_tolerance = -0.1
        
            
    class reward_config():
        dense_rewards = [
            "tracking_x_vel", "tracking_y_vel", "tracking_ang_vel",
            "heading_tracking", 
            "lin_vel_z", "ang_vel_xy", "orientation", "action_rate",
            "tracking_base_height", "deviation_hip_joint", "deviation_ankle_joint", 
            "deviation_knee_joint", "dof_acc", "dof_pos_limits", "feet_air_time",
            "feet_clearance", "feet_distance_lateral", "knee_distance_lateral",
            "feet_ground_parallel", "feet_parallel", "smoothness", "joint_power",
            "feet_stumble", "torques", "dof_vel", "dof_vel_limits", "torque_limits",
            "no_fly", "feet_slip", "feet_contact_forces",
            "contact_momentum", "action_vanish", "stand_still",
        ]
        sparse_rewards = ['foothold']
        
    class normalization:
        """归一化配置"""
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.5
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            
        clip_observations = 100.
        clip_actions = 1.2

    class noise:
        """噪声配置"""
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurement = 0.1
            
    class sim:
        """仿真配置"""
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class HumanoidBEAMDOJOAMPCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy(LeggedRobotCfgPPO.policy):
        """策略网络配置"""
        init_noise_std = 0.8
        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
        activation = 'elu'
        
        # 扫描编码器配置
        scan_encoder_dims = [128, 64, 32]
        priv_encoder_dims = [64, 20]
        tanh_encoder_output = False
        
        # 双Critic配置
        use_double_critic = True
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """BEAMDOJO + AMP PPO算法配置"""
        # 基础PPO参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99   
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        adam_epsilon = 1e-8
        
        # BEAMDOJO双Critic配置
        use_double_critic = True
        dense_value_loss_coef = 1.0
        sparse_value_loss_coef = 1.0
        advantage_merge_weight = 0.5
        dense_reward_weight = 1.0
        sparse_reward_weight = 0.25
        
        # ====== AMP discriminator配置 ======
        enable_amp = True                      # 启用AMP
        disc_hidden_dims = [512, 256]         # discriminator隐藏层
        disc_learning_rate = 5e-5              # discriminator学习率
        disc_loss_weight = 5.0                 # discriminator损失权重
        disc_logit_reg = 0.01                  # logit正则化
        disc_grad_penalty = 5.0                # 梯度惩罚
        disc_weight_decay = 0.0001             # 权重衰减
        disc_reward_scale = 2.0                # discriminator奖励缩放
        disc_eval_batch_size = 4096            # 评估batch大小
        
        # AMP奖励权重
        # ⚠️ 重要：需要平衡任务奖励和AMP奖励
        # - task_reward_weight: 控制速度跟踪、地形导航等任务奖励
        # - disc_reward_weight: 控制运动风格奖励（是否像人走路）
        # 如果task_reward_weight=0，机器人不会响应速度命令！
        # 推荐配置：
        #   - 早期训练：task_reward_weight=0.3, disc_reward_weight=0.7 (更注重风格)
        #   - 后期训练：task_reward_weight=0.5, disc_reward_weight=0.5 (平衡)
        #   - 任务优先：task_reward_weight=0.7, disc_reward_weight=0.3 (更注重任务)
        task_reward_weight = 0.5               # 任务奖励权重 (BeamDojo奖励：速度跟踪、地形导航)
        disc_reward_weight = 0.5               # AMP奖励权重 (风格奖励：自然的人形运动)
        
    class runner(LeggedRobotCfgPPO.runner):
        """训练运行器配置"""
        policy_class_name = 'ActorCriticRMADoubleRewardAMP'  # 使用支持AMP的双Critic策略
        algorithm_class_name = 'PPODoubleRewardAMP'           # 使用支持AMP的双Critic算法
        num_steps_per_env = 24  
        max_iterations = 100000
        
        save_interval = 200
        experiment_name = 'humanoid_beamdojo_amp'
        run_name = ''
        
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class estimator(LeggedRobotCfgPPO.estimator):
        """状态估计器配置"""
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [256, 128, 64]
        priv_states_dim = HumanoidBEAMDOJOAMPCfg.env.n_priv
        num_prop = HumanoidBEAMDOJOAMPCfg.env.n_proprio
        num_scan = HumanoidBEAMDOJOAMPCfg.env.n_scan
        num_hist = HumanoidBEAMDOJOAMPCfg.env.history_len

