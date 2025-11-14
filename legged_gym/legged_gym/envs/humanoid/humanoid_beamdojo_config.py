# BEAMDOJO Humanoid Robot Configuration
# 基于BEAMDOJO论文的人形机器人配置示例
# 展示双Critic网络、两阶段训练和Foothold奖励的完整配置

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HumanoidBEAMDOJOCfg(LeggedRobotCfg):
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
            "right_shoulder_roll_joint": -0.,#-0.3
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.,#0.8
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
        episode_length_s = 8 #与课程学习有关 
        
        n_scan = 225
        n_priv = 3
        n_priv_latent = 4 + 1 + 12 + 12  # 潜在状态维度
        n_proprio = 76  # 实际obs_buf维度：3+1+3+3+27+27+12=76
        history_len = 10
        
        # 重新计算总观测维度
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_actions = 12  # 12个关节动作
        
        # 启用接触信息
        include_foot_contacts = True
        
    # 课程学习配置
    class curriculum_config:
        # === 课程学习成功判定模式 ===
        success_mode = 'survival_time'  # 'goal_reached': 到达目标点, 'survival_time': 存活指定时间, 'vel_tracking': 速度跟踪
        
        # === 成功率计算模式 (用于日志记录) ===
        success_rate_mode = 'survival_time'  # 'survival_time': 基于存活时间, 'goal_based': 基于目标完成度
        
        # 目标到达模式参数
        success_threshold = 3  # 连续成功次数阈值
        failure_threshold = 2  # 连续失败次数阈值
         
        # 存活时间模式参数
        survival_time_threshold = 8  # 存活时间阈值（秒）
        survival_success_threshold = 2  # 连续存活成功次数阈值
        survival_failure_threshold = 3   # 连续存活失败次数阈值
        
        # 速度模式参数
        velocity_success_threshold = 3  # 连续速度成功次数阈值
        velocity_failure_threshold = 2   # 连续速度失败次数阈值
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
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
                     }  # [N*m/rad]
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
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4    
        hip_reduction = 1.0
    
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True            # 随机化摩擦系数
        friction_range = [0.8, 0.8]         # 恢复原始摩擦系数
        randomize_base_mass = True          # 随机化质量
        added_mass_range = [-2.0, 2.0]      # 负载质量 U(-2.0, 2.0) kg
        randomize_base_com = True           # 随机化质心位置
        added_com_range = [-0.05, 0.05]     # 质心偏移 U(-0.05, 0.05) m
        push_robots = True                   # 启用外部推力（抗干扰训练）
        push_interval_s = 8                  # 推力间隔：每8秒推一次
        max_push_vel_xy = 0.5                # 最大推力速度：±0.5 m/s

        randomize_motor = True              # 随机化电机特性
        motor_strength_range = [0.9, 1.1]   # 电机强度噪声 U(0.9, 1.1)
        
        randomize_actuator_offset = True    # 随机化执行器零位偏移
        actuator_offset_range = [-0.05, 0.05]  # 执行器偏移 U(-0.05, 0.05) rad
        
        randomize_pd_gains = True           # 随机化PD增益
        pd_gain_range = [0.85, 1.15]        # Kp/Kd噪声因子 U(0.85, 1.15)

        # 动作延迟相关参数（BeamDojo域随机化）
        delay_update_global_steps = 24 * 8000  # 延迟更新的全局步数
        action_delay = True              # 是否启用动作延迟
        action_curr_step = [1, 1]         # 当前动作步数范围
        action_curr_step_scratch = [0, 1] # 从头训练时的动作步数范围
        action_delay_view = 1             # 动作延迟视图
        action_buf_len = 8                # 动作缓冲区长度
        
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
        
        randomize_start_pos = False    # 是否随机化起始位置
        randomize_start_vel = False    # 是否随机化起始速度
        randomize_start_yaw = False    # 是否随机化起始偏航角
        rand_yaw_range = 1.2          # 偏航角随机范围
        randomize_start_y = False     # 是否随机化Y轴起始位置
        rand_y_range = 0.5            # Y轴随机范围
        randomize_start_pitch = False  # 是否随机化起始俯仰角
        rand_pitch_range = 1.6        # 俯仰角随机范围

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

        
    class commands( LeggedRobotCfg.commands ):
        """运动命令配置"""
        curriculum = True           # 是否启用课程学习
        num_commands = 5 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, height, orientation
        resampling_time = 8.0         # 命令重采样时间间隔（秒）
        heading_command = True         # 启用朝向命令模式
        ang_vel_clip = 0.05            # 角速度命令死区阈值
        lin_vel_clip = 0.1            # 线速度命令死区阈值
        
        # 策略1：智能速度生成配置
        height_adaptive_speed = False   # 启用基于高度的自适应速度
        speed_complexity_weight = 0.4  # 地形复杂度权重
        speed_gradient_weight = 0.4   # 高度梯度权重  
        speed_roughness_weight = 0.2  # 地形粗糙度权重
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.8, 1.2] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
            heading = [-1.0, 1.0]
            height = [-0.5, 0.0]
                        
    class rewards(LeggedRobotCfg.rewards):
        """BEAMDOJO奖励配置"""
        class scales:
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.0 #2.0
            heading_tracking = 1.0 #2.0 3.0
            # next_heading_tracking = 0.5 #1.5 2.0
            # reach_goal = 2.0
            # center = -1.0 
              
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.5 #-1.5 -2.0 -5.0 -10.0
            action_rate = -0.01
            
            # base_height = -10.0
            tracking_base_height = 2.
            deviation_hip_joint = -0.2
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -2.
            feet_air_time = 0.05   #0.5
            feet_clearance = -0.25  #-1.0 
            feet_distance_lateral = 0.5
            knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0
            feet_parallel = -3.0
            smoothness = -0.05
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            # joint_tracking_error = -0.1
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -0.15   
            # termination = -20 #-10 -20 -30
            
            foothold = 0.1 #1.0 0.05 0.15 0.25 0.1 0.12
            
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.975
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.74
        max_contact_force = 400.
        least_feet_distance = 0.2
        least_feet_distance_lateral = 0.2
        most_feet_distance_lateral = 0.35
        most_knee_distance_lateral = 0.35
        least_knee_distance_lateral = 0.2
        clearance_height_target = 0.14 #0.18
        is_play = False                   # 是否为播放模式
        
        foothold_foot_length = 0.12         # 脚长度 [m] 
        foothold_foot_width = 0.06          # 脚宽度 [m]
        foothold_height_tolerance = -0.1    # 高度容忍度 [m]
        
            
    class reward_config():
        dense_rewards = [
            "tracking_x_vel", "tracking_y_vel", "tracking_ang_vel",
            "heading_tracking", 
            # "next_heading_tracking", 
            # "reach_goal","center",
            "lin_vel_z", "ang_vel_xy", "orientation", "action_rate",
            "tracking_base_height", "deviation_hip_joint", "deviation_ankle_joint", 
            "deviation_knee_joint", "dof_acc", "dof_pos_limits", "feet_air_time",
            "feet_clearance", "feet_distance_lateral", "knee_distance_lateral",
            "feet_ground_parallel", "feet_parallel", "smoothness", "joint_power",
            "feet_stumble", "torques", "dof_vel", "dof_vel_limits", "torque_limits",
            "no_fly", "feet_slip", "feet_contact_forces",
            "contact_momentum", "action_vanish", "stand_still",
            # 'termination'
        ]
        sparse_rewards = ['foothold']
        
    class normalization:
        """归一化配置"""
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
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
            max_gpu_contact_pairs = 2**23  # 增加到2**24以支持更多环境
            default_buffer_size_multiplier = 5  # 增加缓冲区倍数
            contact_collection = 2


class HumanoidBEAMDOJOCfgPPO(LeggedRobotCfgPPO):
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
        tanh_encoder_output = False  # 编码器输出是否使用tanh激活
        
        # 支持双Critic的编码器
        use_double_critic = True  # 在这里可以启用双Critic
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """BEAMDOJO PPO算法配置"""
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
        use_double_critic = True      # 设置为True启用双Critic
        dense_value_loss_coef = 1.0    # 密集奖励价值损失系数
        sparse_value_loss_coef = 1.0   # 稀疏奖励价值损失系数
        advantage_merge_weight = 0.5   # 优势函数合并权重
        dense_reward_weight = 1.0
        sparse_reward_weight = 0.25
        
    class runner(LeggedRobotCfgPPO.runner):
        """训练运行器配置"""
        policy_class_name = 'ActorCriticRMADoubleReward'  # 使用 'ActorCriticRMADoubleReward' 启用双Critic
        algorithm_class_name = 'PPODoubleReward'       # 使用 'PPODoubleReward' 启用双Critic算法
        num_steps_per_env = 24  
        max_iterations = 100000
        
        save_interval = 200
        experiment_name = 'humanoid_beamdojo'
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
        priv_states_dim = HumanoidBEAMDOJOCfg.env.n_priv
        num_prop = HumanoidBEAMDOJOCfg.env.n_proprio
        num_scan = HumanoidBEAMDOJOCfg.env.n_scan
        num_hist = HumanoidBEAMDOJOCfg.env.history_len

    # class depth_encoder(LeggedRobotCfgPPO.depth_encoder):
    #     """深度编码器配置"""
    #     pass  # 使用基类配置

    # BEAMDOJO两阶段训练配置
    # class training:
    #     """两阶段训练配置"""
    #     enable_two_stage = False  # 设置为True启用两阶段训练
        
    #     class stage1:
    #         """Stage1软约束训练配置"""
    #         min_steps = 1000000            # 最小训练步数
    #         max_steps = 5000000            # 最大训练步数  
    #         success_threshold = 0.8        # 成功率阈值
    #         terrain_type = "flat_with_target_perception"
    #         use_soft_termination = True    # 软终止：踩空不终止episode
    #         use_target_perception = True   # 使用目标地形感知
            
    #         # Stage1命令范围（全方向）
    #         class command_ranges:
    #             lin_vel_x = [-1.0, 1.0]
    #             lin_vel_y = [-1.0, 1.0]
    #             ang_vel_yaw = [-1.0, 1.0]
        
    #     class stage2:
    #         """Stage2硬约束训练配置"""
    #         terrain_type = "sparse_terrain"
    #         use_soft_termination = False   # 硬终止：踩空立即终止
    #         use_target_perception = False  # 不使用目标地形感知
            
    #         # Stage2命令范围（仅前进）
    #         class command_ranges:
    #             lin_vel_x = [-1.0, 1.0]
    #             lin_vel_y = [0.0, 0.0]     # 固定为0
    #             ang_vel_yaw = [0.0, 0.0]   # 固定为0
