"""
MuJoCo部署脚本：将训练好的策略网络部署到MuJoCo仿真环境中

整体控制流程：
1. 【观测构建】从MuJoCo获取机器人状态（关节角度、速度、基座姿态等）
2. 【观测处理】将状态信息转换为策略网络所需的观测格式（BeamDojo格式）
   - 本体感受信息（75维）：速度指令、角速度、重力方向、关节状态等
   - 雷达扫描数据（225维）：从地形raycast获取的高度信息
   - 特权信息（3+29维）：训练时可用，部署时通常为0
   - 历史信息（750维）：过去10步的本体感受信息
3. 【网络推理】将观测输入策略网络，得到动作输出（27维关节角度）
4. 【动作应用】将动作转换为目标关节角度，通过PD控制器计算关节力矩
5. 【物理仿真】将力矩应用到MuJoCo，执行一步物理仿真
6. 重复步骤1-5，实现闭环控制

关键数据流：
  机器人状态 → 观测向量(1082维) → 策略网络 → 动作(27维) → 目标角度 → PD控制 → 力矩 → MuJoCo
"""

import time
import mujoco.viewer
import mujoco
import numpy as np
import math
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import cv2
import numpy as np
import torch.nn.functional as F


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)

def get_gravity_orientation(quaternion):
    R = quat_to_rot(quaternion)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R.T @ g_world).astype(np.float32)

def get_yaw(quaternion):
    w, x, y, z = quaternion
    s0 = 2.0 * (w * z + x * y)
    s1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s0, s1)

def rotate_world_to_body(vec, quaternion):
    R = quat_to_rot(quaternion)
    return (R.T @ vec).astype(np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    PD控制器：根据目标关节角度和当前关节状态计算关节力矩
    
    Args:
        target_q: 目标关节角度 [n_joints]
        q: 当前关节角度 [n_joints]
        kp: 位置增益（比例系数）[n_joints]
        target_dq: 目标关节角速度（通常为0）[n_joints]
        dq: 当前关节角速度 [n_joints]
        kd: 速度增益（微分系数）[n_joints]
    
    Returns:
        tau: 关节力矩 [n_joints]
    
    公式：tau = kp * (target_q - q) + kd * (target_dq - dq)
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_scan_from_terrain(m, d, base_pos, base_quat, scan_points_body):
    """
    从MuJoCo地形获取225维雷达scan数据
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        base_pos: 机器人基座位置 [x, y, z]
        base_quat: 机器人基座四元数 [w, x, y, z]
        scan_points_body: 本体坐标系采样点 [225, 3]
    
    Returns:
        scan_heights: 225维高度数据 [225] - 相对基座的高度（米）
    """
    # 将本体坐标系点转换到世界坐标系
    R = quat_to_rot(base_quat)
    scan_points_world = (R @ scan_points_body.T).T + base_pos
    
    # 使用MuJoCo的raycast查询地形高度
    scan_heights = np.zeros(225, dtype=np.float32)
    
    # 从上方向下raycast查询地形高度
    ray_start = scan_points_world.copy()
    ray_start[:, 2] += 2.0  # 从上方2米开始
    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 向下
    ray_length = 4.0  # 射线长度4米
    
    for i in range(225):
        # MuJoCo raycast: mj_ray(model, data, pnt, vec, geomgroup=None, flg_static=1, bodyexclude=-1)
        # 返回: (geomid, distance) 或 (-1, inf) 如果未命中
        geomid, distance = mujoco.mj_ray(m, d, ray_start[i], ray_dir, 
                                         geomgroup=None, flg_static=1, bodyexclude=-1)
        
        if geomid >= 0 and distance < ray_length:  # 命中地形
            hit_pos = ray_start[i] + ray_dir * distance
            scan_heights[i] = hit_pos[2] - base_pos[2]  # 相对基座的高度
        else:
            # 如果没有命中，使用默认值（可能是悬空）
            scan_heights[i] = -2.0  # 默认低于基座2米
    
    return scan_heights



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MuJoCo部署脚本:运行训练好的策略网络")
    parser.add_argument("config_file", type=str, help="配置文件名称(位于configs文件夹中)")
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # ========== 仿真配置 ==========
        simulation_duration = config["simulation_duration"]  # 仿真持续时间（秒）
        simulation_dt = config["simulation_dt"]              # 仿真时间步长（秒），通常为0.002
        control_decimation = config["control_decimation"]    # 控制降采样率，策略每N步执行一次

        # ========== PD控制参数 ==========
        kps = np.array(config["kps"], dtype=np.float32)  # 位置增益数组（27维，单位：N·m/rad）
        kds = np.array(config["kds"], dtype=np.float32)  # 速度增益数组（27维，单位：N·m·s/rad）

        # ========== 关节角度配置 ==========
        default_angles = np.array(config["default_angles"], dtype=np.float32)  # 默认关节角度（27维，rad）
        init_qpos = np.array(config["init_qpos"], dtype=np.float32)            # 初始关节位置（包含基座姿态）

        # ========== 观测缩放参数 ==========
        # 这些缩放因子用于归一化观测数据，使其范围适合神经网络输入
        # 训练时使用相同的缩放因子，确保部署时的一致性
        lin_vel_scale = config["lin_vel_scale"]      # 线速度缩放因子（通常为2.0）
        ang_vel_scale = config["ang_vel_scale"]      # 角速度缩放因子（通常为0.5）
        dof_pos_scale = config["dof_pos_scale"]       # 关节角度缩放因子（通常为1.0）
        dof_vel_scale = config["dof_vel_scale"]       # 关节角速度缩放因子（通常为0.05）
        action_scale = config["action_scale"]        # 动作缩放因子（通常为0.25，将[-1,1]映射到±0.25 rad）
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)  # 速度指令缩放因子 [vx, vy, omega_z]

        # ========== 观测和动作维度配置 ==========
        num_actions = config["num_actions"]        # 动作维度（通常为12，下半身关节数）
        num_obs = config["num_obs"]                # 基础观测维度（用于历史记录）
        obs_history_len = config["obs_history_len"] # 历史长度（通常为10步）
        
        # ========== BeamDojo JIT模型维度配置 ==========
        # 这些维度必须与训练时的JIT模型签名完全匹配
        # 如果配置文件中没有指定，使用默认值（向后兼容）
        jit_n_proprio = int(config.get("jit_n_proprio", 75))           # 本体感受维度（75维）
        jit_num_scan = int(config.get("jit_num_scan", 225))            # 雷达扫描维度（225维，15x15网格）
        jit_n_priv_latent = int(config.get("jit_n_priv_latent", 29))   # 特权隐式维度（29维）
        jit_n_priv_explicit = int(config.get("jit_n_priv_explicit", 3))  # 特权显式维度（3维）
        jit_depth_latent_dim = int(config.get("jit_depth_latent_dim", 32))  # 深度图像潜在特征维度（32维）
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        gait_cmd = np.array(config["gait_cmd"], dtype=np.float32)
        num_gaits = gait_cmd.shape[0]

        # depth camera configs
        depth_far_clip = config["depth_far_clip"]
        depth_near_clip = config["depth_near_clip"]
        depth_buffer_len = config["depth_buffer_len"]
        depth_size = config["depth_size"]
        cam_update_interval = config["cam_update_interval"]
        crop_image = config["crop_image"]
        crop_size = config["crop_size"]
        gaussian_filter = config["gaussian_filter"]
        gaussian_filter_kernel = config["gaussian_filter_kernel"]
        gaussian_filter_sigma = config["gaussian_filter_sigma"]
        gaussian_noise = config["gaussian_noise"]
        gaussian_noise_std = config["gaussian_noise_std"]
        depth_dis_noise = config["depth_dis_noise"]
        use_depth = bool(config.get("use_depth", False))
        use_priv = bool(config.get("use_privileged", False))
        # Optional lidar (225-d scan) config - defaults keep backward compatibility
        use_lidar = bool(config.get("use_lidar", True))
        lidar_mode = str(config.get("lidar_mode", "terrain"))  # 'zeros', 'npy', or 'terrain'
        lidar_path = str(config.get("lidar_path", ""))       # path to .npy with shape (225,) or (N,225)
        lidar_scale = float(config.get("lidar_scale", 1.0))
        lidar_clip = float(config.get("lidar_clip", 5.0))
        
        # ========== 初始化雷达扫描点（本体坐标系）==========
        # 在机器人本体坐标系中定义15×15网格（共225个点），用于raycast查询地形高度
        # 扫描范围：x和y方向都是[-0.7m, -0.6m, ..., 0.6m, 0.7m]，共15个点
        # 这些点位于机器人基座平面（z=0），后续会通过raycast查询每个点下方的地形高度
        scan_x = np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        scan_y = np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        scan_points_body = np.zeros((225, 3), dtype=np.float32)  # [225, 3] - 225个点，每个点3个坐标
        idx = 0
        for x in scan_x:
            for y in scan_y:
                scan_points_body[idx] = [x, y, 0.0]  # z=0表示在基座平面上
                idx += 1
        priv_latent_default = float(config.get("priv_latent_default", 0.0))
        priv_explicit_default = float(config.get("priv_explicit_default", 0.0))
        goals_cfg = config.get("goals", None)

    def process_depth_image(depth_image):
        depth_image += depth_dis_noise * 2 * (np.random.rand(1) - 0.5)
        if gaussian_noise:
            depth_image += gaussian_noise_std * np.random.randn(*depth_image.shape)
        depth_image = np.clip(depth_image, depth_near_clip, depth_far_clip)
        depth_image = (depth_image - depth_near_clip) / (depth_far_clip - depth_near_clip) - 0.5
        return depth_image
    
    def crop_resize_depth(depth_image):
        clip_left, clip_top, clip_right, clip_bottom = crop_size
        depth_image = F.interpolate(depth_image[clip_top:-clip_bottom, clip_left:depth_size[1]-clip_right].unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        return depth_image
    
    def adaptive_gaussian_filter(depth_image, kernel_size=gaussian_filter_kernel, sigma=gaussian_filter_sigma):
        imgs = cv2.GaussianBlur(depth_image.numpy(), (kernel_size, kernel_size), sigma)
        return torch.from_numpy(imgs).to(depth_image.device)
    
    def update_depth_cam(depth_image_buffer):
        depth_renderer.update_scene(d, camera=depth_cam_id)
        depth_image = depth_renderer.render()
        depth_image = np.rot90(depth_image, k=1)
        depth_image = process_depth_image(depth_image)
        depth_image = torch.tensor(depth_image)
        if crop_image:
            depth_image = crop_resize_depth(depth_image)
        if gaussian_filter:
            depth_image = adaptive_gaussian_filter(depth_image, kernel_size=gaussian_filter_kernel, sigma=gaussian_filter_sigma)

        cv2.namedWindow('depth image', cv2.WINDOW_NORMAL)
        cv2.imshow("depth image", depth_image_buffer[0, -1].detach().numpy() + 0.5)
        cv2.waitKey(1)
        return depth_image


    # ========== 初始化关键变量 ==========
    # action: 当前动作（归一化的关节角度，范围[-1, 1]），用于下一时刻的观测
    action = np.zeros(num_actions, dtype=np.float32)
    
    # target_dof_pos: 目标关节角度（实际角度值），由策略网络输出转换得到，用于PD控制
    target_dof_pos = default_angles.copy()
    
    # obs: 基础观测向量（用于历史记录），包含速度指令、关节状态等
    obs = np.zeros(num_obs, dtype=np.float32)
    
    # trajectory_history: 观测历史记录，保存过去 obs_history_len 步的观测
    trajectory_history = torch.zeros(size=(1, obs_history_len, num_obs - num_gaits))
    
    # jit_history_buf: 本体感受历史记录，保存过去 obs_history_len 步的本体感受信息（75维）
    # 用于构建BeamDojo观测中的历史部分（750维 = 10步 × 75维）
    jit_history_buf = torch.zeros(size=(1, obs_history_len, jit_n_proprio))
    
    # depth_image_buffer: 深度图像缓冲区，保存最近 depth_buffer_len 帧的深度图像
    depth_image_buffer = torch.zeros(1, depth_buffer_len, 64, 64)
    # Prepare lidar buffer if enabled
    lidar_data = None
    lidar_idx = 0
    if use_lidar:
        if lidar_mode == "npy" and len(lidar_path) > 0:
            try:
                data = np.load(lidar_path)
                if data.ndim == 1 and data.shape[0] == jit_num_scan:
                    lidar_data = data.reshape(1, jit_num_scan)
                    print(f"[sim2sim] Loaded lidar npy with shape {data.shape} -> using 1 frame")
                elif data.ndim == 2 and data.shape[1] == jit_num_scan:
                    lidar_data = data
                    print(f"[sim2sim] Loaded lidar npy with shape {data.shape} -> using sequence")
                else:
                    print(f"[sim2sim] Warning: lidar npy shape {data.shape} not compatible (expected (225,) or (N,225)); fallback to zeros")
            except Exception as e:
                print(f"[sim2sim] Warning: failed to load lidar npy: {e}; fallback to zeros")
    counter = 0
    cam_update_counter = 0
    control_tick = 0
    path_history = []
    map_scale = 50  # pixels per meter for goal visualization
    map_size = (600, 600)
    def world_to_pixel(p):
        cx, cy = map_size[1] // 2, map_size[0] // 2
        return int(cx + p[0] * map_scale), int(cy - p[1] * map_scale)
    if goals_cfg is None:
        goals = [np.array([i * 2.0, 0.0], dtype=np.float32) for i in range(1, 6)]
    else:
        goals = [np.array(g, dtype=np.float32) for g in goals_cfg]
    cur_goal_idx = 0
    goal_thresh = 0.6


    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # set init dof pos
    d.qpos[3:] = init_qpos

    save_imgs = []
    # ========== 定义受控关节名称 ==========
    # 下半身12个关节：左右腿各6个关节（髋部3个+膝盖1个+踝部2个）
    controlled_joint_names = [
        "left_hip_pitch_joint",      # 左髋俯仰
        "left_hip_roll_joint",       # 左髋横滚
        "left_hip_yaw_joint",        # 左髋偏航
        "left_knee_joint",           # 左膝
        "left_ankle_pitch_joint",    # 左踝俯仰
        "left_ankle_roll_joint",     # 左踝横滚
        "right_hip_pitch_joint",     # 右髋俯仰
        "right_hip_roll_joint",      # 右髋横滚
        "right_hip_yaw_joint",       # 右髋偏航
        "right_knee_joint",          # 右膝
        "right_ankle_pitch_joint",   # 右踝俯仰
        "right_ankle_roll_joint",    # 右踝横滚
    ]
    # 根据配置选择前num_actions个关节（通常为12个）
    selected_joint_names = controlled_joint_names[:num_actions]
    
    # ========== 计算关节在MuJoCo状态数组中的索引 ==========
    # 获取关节ID
    joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in selected_joint_names]
    
    # 计算关节在qpos中的索引（相对于d.qpos[7:]）
    # m.jnt_qposadr[i]: 关节i在qpos中的起始地址
    # 减去7是因为前7个是基座状态：位置(3) + 四元数(4) = 7
    # 结果：controlled_qpos_indices是相对于d.qpos[7:]的索引
    controlled_qpos_indices = [m.jnt_qposadr[i] - 7 for i in joint_ids]
    
    # 计算关节在qvel中的索引（相对于d.qvel[6:]）
    # m.jnt_dofadr[i]: 关节i在qvel中的起始地址
    # 减去6是因为前6个是基座速度：线速度(3) + 角速度(3) = 6
    # 结果：controlled_qvel_indices是相对于d.qvel[6:]的索引
    controlled_qvel_indices = [m.jnt_dofadr[i] - 6 for i in joint_ids]

    # load policy
    print(f"[sim2sim] Loading JIT policy from: {policy_path}")
    try:
        policy = torch.jit.load(policy_path)
        print(f"[sim2sim] Policy loaded successfully")
    except Exception as e:
        print(f"[sim2sim ERROR] Failed to load policy: {e}")
        raise
    
    # Print configuration summary
    print(f"\n[sim2sim] ===== Configuration Summary =====")
    print(f"  JIT dimensions: prop={jit_n_proprio}, scan={jit_num_scan}, priv_explicit={jit_n_priv_explicit}, priv_latent={jit_n_priv_latent}, history_len={obs_history_len}")
    print(f"  Expected obs dim: {jit_n_proprio + jit_num_scan + jit_n_priv_explicit + jit_n_priv_latent + obs_history_len * jit_n_proprio}")
    print(f"  Use depth: {use_depth}, Use lidar: {use_lidar}, Use privileged: {use_priv}")
    if use_lidar:
        print(f"  Lidar mode: {lidar_mode}, path: {lidar_path if lidar_path else '(none)'}, scale: {lidar_scale}, clip: {lidar_clip}")
        if lidar_mode == "terrain":
            print(f"  Terrain scan: 15x15 grid (225 points) in body frame [-0.7m to +0.7m]")
    print(f"  Actions: {num_actions}, Control decimation: {control_decimation}")
    print(f"[sim2sim] ===================================\n")

    save_imgs = []
    if use_depth:
        depth_renderer = mujoco.Renderer(m, width=depth_size[1], height=depth_size[0])
        depth_renderer.enable_depth_rendering()
        depth_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # ========== PD控制：将目标关节角度转换为关节力矩 ==========
            # PD控制是位置-速度反馈控制，用于将目标关节角度转换为关节力矩
            # 
            # PD控制公式：tau = kp * (target_q - current_q) + kd * (target_dq - current_dq)
            # 这里 target_dq = 0（期望角速度为0），所以简化为：
            # tau = kp * (target_q - current_q) - kd * current_dq
            # 
            # 参数说明：
            #   - target_dof_pos: 目标关节角度（27维，由策略网络计算得到）
            #   - d.qpos[7:]: 当前所有关节角度（27维，从第7个开始，前7个是基座状态）
            #   - kps: 位置增益数组（27维，比例系数，单位：N·m/rad）
            #   - np.zeros_like(kds): 目标角速度（27维，全0，期望角速度为0）
            #   - d.qvel[6:]: 当前所有关节角速度（27维，从第6个开始，前6个是基座速度）
            #   - kds: 速度增益数组（27维，微分系数，单位：N·m·s/rad）
            # 
            # 输出：
            #   - tau: 计算得到的关节力矩（27维），用于驱动关节运动
            # 
            # 物理意义：
            #   - kp项：位置误差越大，力矩越大（比例项，提供刚度）
            #   - kd项：角速度越大，阻尼力矩越大（微分项，提供阻尼，防止振荡）
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            
            # 将计算得到的力矩应用到MuJoCo的控制输入
            # d.ctrl是MuJoCo的控制输入数组，长度等于执行器数量（通常等于关节数量）
            # 这一步将PD控制器计算的力矩传递给MuJoCo物理引擎
            d.ctrl[:] = tau
            
            # 执行一步物理仿真：根据当前状态和控制输入，计算下一时刻的状态
            # 这一步会：
            #   1. 根据当前状态（d.qpos, d.qvel）和控制输入（d.ctrl）计算加速度
            #   2. 积分得到新的速度和位置
            #   3. 更新 d.qpos（位置）和 d.qvel（速度）
            #   4. 检测碰撞和接触
            # 仿真时间步长：m.opt.timestep（通常为0.002秒，即500Hz）
            mujoco.mj_step(m, d)
            counter += 1
            
            # ========== 控制频率说明 ==========
            # 仿真频率：1 / simulation_dt = 1 / 0.002 = 500 Hz（每步0.002秒）
            # 控制频率：500 / control_decimation = 500 / 10 = 50 Hz（每10步执行一次策略）
            # 控制周期：simulation_dt * control_decimation = 0.002 * 10 = 0.02秒
            # 
            # 这意味着：
            #   - 物理仿真以500Hz运行（每2ms一步）
            #   - 策略网络以50Hz运行（每20ms推理一次）
            #   - PD控制器以500Hz运行（每步都计算力矩）
            #   - 策略输出的目标角度会保持0.02秒，直到下一次更新
            if (counter > 0) and (counter % control_decimation) == 0:
                # ========== 控制循环：每 control_decimation 步执行一次策略推理 ==========
                # 当counter是control_decimation的倍数时，执行策略推理和动作更新
                control_tick += 1

                # ========== 第一步：构建观测（Observation） ==========
                # 观测是策略网络的输入，包含机器人的状态信息
                
                # 1.1 从MuJoCo获取原始状态数据
                qj = d.qpos[7:]          # 所有关节角度（从第7个开始，前7个是基座位置和姿态）
                dqj = d.qvel[6:]         # 所有关节角速度（从第6个开始，前6个是基座线速度和角速度）
                quat = d.qpos[3:7]       # 基座四元数 [w, x, y, z]
                omega = rotate_world_to_body(d.qvel[3:6], quat)  # 基座角速度（转换到本体坐标系）

                # 1.2 对原始数据进行归一化和缩放处理
                qj = (qj - default_angles) * dof_pos_scale      # 关节角度：减去默认角度后缩放
                dqj = dqj * dof_vel_scale                        # 关节角速度：直接缩放
                gravity_orientation = get_gravity_orientation(quat)  # 重力方向（在本体坐标系中）
                omega = omega * ang_vel_scale                     # 基座角速度：缩放

                # 1.3 计算速度指令（cmd）：根据目标点生成期望的线速度
                base_pos = d.qpos[:2]    # 基座位置 [x, y]
                quat = d.qpos[3:7]       # 基座四元数
                yaw = get_yaw(quat)      # 提取偏航角
                if cur_goal_idx < len(goals):
                    goal_vec = goals[cur_goal_idx] - base_pos  # 目标方向向量
                    dist = np.linalg.norm(goal_vec)            # 到目标的距离
                    # 如果到达当前目标，切换到下一个目标
                    if dist < goal_thresh and cur_goal_idx < len(goals) - 1:
                        cur_goal_idx += 1
                        goal_vec = goals[cur_goal_idx] - base_pos
                        dist = np.linalg.norm(goal_vec)
                    # 计算期望速度（世界坐标系）
                    if dist > 1e-6:
                        v_world = 0.8 * goal_vec / dist  # 归一化后乘以0.8m/s
                        # 将世界坐标系速度转换到本体坐标系
                        cy = math.cos(-yaw)
                        sy = math.sin(-yaw)
                        vx = cy * v_world[0] - sy * v_world[1]
                        vy = sy * v_world[0] + cy * v_world[1]
                        cmd = np.array([vx, vy, 0.0], dtype=np.float32)  # [vx, vy, omega_z]
                # 限制速度指令范围在[-1, 1]
                cmd = np.clip(cmd, np.array([-1.0, -1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32))
                
                # 1.4 构建基础观测向量 obs（用于历史记录）
                # obs的结构：[gait_cmd(3) | cmd(3) | omega(3) | gravity(3) | qj(12) | dqj(12) | action(12)]
                # 总维度：3 + 3 + 3 + 3 + 12 + 12 + 12 = 48（如果num_actions=12）
                obs[:3] = gait_cmd                              # [0:3] 步态指令（通常为[1,0,0]）
                obs[3:6] = cmd * cmd_scale                      # [3:6] 速度指令（缩放后，单位：m/s或rad/s）
                obs[6:9] = omega                                # [6:9] 基座角速度（本体坐标系，已缩放）
                obs[9:12] = gravity_orientation                # [9:12] 重力方向（本体坐标系，归一化）
                
                # 从所有关节中选择受控的12个关节
                # 注意：controlled_qpos_indices和controlled_qvel_indices是相对于qj和dqj的索引
                # qj = d.qpos[7:]（所有27个关节角度，已归一化）
                # dqj = d.qvel[6:]（所有27个关节角速度，已归一化）
                qj_sel = qj[controlled_qpos_indices]           # 选中的12个关节角度
                dqj_sel = dqj[controlled_qvel_indices]         # 选中的12个关节角速度
                
                obs[12 : 12 + num_actions] = qj_sel             # [12:24] 关节角度（12个）
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj_sel  # [24:36] 关节角速度（12个）
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action  # [36:48] 上一时刻的动作（12个）

                # 1.5 获取深度图像（如果启用）
                if use_depth:
                    if (cam_update_counter) % cam_update_interval == 0:
                        depth_imgs = update_depth_cam(depth_image_buffer)
                        # 更新深度图像缓冲区（滑动窗口）
                        if (depth_image_buffer==0).all().item():
                            depth_image_buffer = torch.stack([depth_imgs] * depth_buffer_len, dim=0).unsqueeze(0)
                        else:
                            depth_image_buffer = torch.cat([depth_image_buffer[:, 1:, ...], depth_imgs.unsqueeze(0).unsqueeze(1)], dim=1)
                    cam_update_counter += 1

                # ========== 第二步：构建BeamDojo格式的观测 ==========
                # BeamDojo观测格式：[本体感受(75) | 雷达扫描(225) | 特权显式(3) | 特权隐式(29) | 历史(750)]
                # 总维度 = 75 + 225 + 3 + 29 + 750 = 1082
                
                # 2.1 更新历史记录（用于时序信息）
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                trajectory_history = torch.cat([trajectory_history[:, 1:], obs_tensor.unsqueeze(1)[..., num_gaits:]], dim=1)
                
                # 2.2 构建本体感受信息（proprioception）：包含速度指令、角速度、重力方向、关节状态等
                jit_prop = torch.zeros(jit_n_proprio, dtype=torch.float32)  # 默认75维
                cmd_prop = torch.from_numpy(cmd * cmd_scale).float()         # 速度指令 [3]
                omega_prop = torch.from_numpy(omega).float()                  # 基座角速度 [3]
                grav_prop = torch.from_numpy(gravity_orientation).float()      # 重力方向 [3]
                pos27 = torch.zeros(27, dtype=torch.float32)                 # 27个关节角度（填充0）
                vel27 = torch.zeros(27, dtype=torch.float32)                 # 27个关节角速度（填充0）
                pos27[:num_actions] = torch.from_numpy(qj_sel).float()       # 前12个关节角度
                vel27[:num_actions] = torch.from_numpy(dqj_sel).float()     # 前12个关节角速度
                last_act12 = torch.from_numpy(action).float()                # 上一时刻动作 [12]
                # 拼接所有本体感受信息：[cmd(3) | omega(3) | gravity(3) | pos27(27) | vel27(27) | action(12)] = 75维
                jit_prop_comp = torch.cat([cmd_prop, omega_prop, grav_prop, pos27, vel27, last_act12], dim=0)
                fill_len = min(jit_prop_comp.numel(), jit_n_proprio)
                jit_prop[:fill_len] = jit_prop_comp[:fill_len]
                # 更新历史缓冲区（保存过去 obs_history_len 步的本体感受信息）
                jit_history_buf = torch.cat([jit_history_buf[:, 1:], jit_prop.unsqueeze(0).unsqueeze(1)], dim=1)
                
                # 2.3 获取雷达扫描数据（scan）：225维高度信息
                # 优先使用lidar模式，否则使用深度图像池化，最后使用零向量
                hist_flat = jit_history_buf.reshape(1, -1)  # 展平历史：[1, obs_history_len * 75] = [1, 750]
                if use_lidar:
                    if lidar_mode == "terrain":
                        # 从MuJoCo地形获取225维高度数据
                        base_pos_mj = d.qpos[:3]   # 基座位置 [x, y, z]
                        base_quat_mj = d.qpos[3:7] # 基座四元数 [w, x, y, z]
                        # 使用raycast查询地形高度（15x15网格，225个点）
                        scan_np = get_scan_from_terrain(m, d, base_pos_mj, base_quat_mj, scan_points_body)
                        # 缩放和裁剪到合理范围
                        scan_np = np.clip(scan_np * lidar_scale, -lidar_clip, lidar_clip).astype(np.float32)
                        scan_feat = torch.from_numpy(scan_np).unsqueeze(0)  # [1, 225]
                    elif lidar_data is not None:
                        # 从npy文件加载预录制的扫描数据
                        scan_np = lidar_data[lidar_idx % lidar_data.shape[0]]
                        lidar_idx += 1
                        scan_np = np.clip(scan_np * lidar_scale, -lidar_clip, lidar_clip).astype(np.float32)
                        scan_feat = torch.from_numpy(scan_np).unsqueeze(0)
                    else:
                        scan_feat = torch.zeros(1, jit_num_scan)  # 使用零向量
                elif use_depth and torch.count_nonzero(depth_image_buffer).item() > 0:
                    # 从深度图像池化得到225维特征（15x15）
                    last_depth = depth_image_buffer[:, -1, ...]
                    scan_feat = F.adaptive_avg_pool2d(last_depth.unsqueeze(1), (15, 15)).reshape(1, -1)
                else:
                    scan_feat = torch.zeros(1, jit_num_scan)  # 默认零向量
                
                # 2.4 构建特权信息（privileged information）：训练时可用，部署时通常为0
                if use_priv:
                    priv_explicit = torch.full((1, jit_n_priv_explicit), priv_explicit_default, dtype=torch.float32)  # [1, 3]
                    priv_latent = torch.full((1, jit_n_priv_latent), priv_latent_default, dtype=torch.float32)        # [1, 29]
                else:
                    priv_explicit = torch.zeros(1, jit_n_priv_explicit)  # [1, 3]
                    priv_latent = torch.zeros(1, jit_n_priv_latent)      # [1, 29]
                
                # 2.5 拼接完整的BeamDojo观测向量
                # 格式：[本体感受(75) | 雷达扫描(225) | 特权显式(3) | 特权隐式(29) | 历史(750)]
                beamdojo_obs = torch.cat([
                    jit_prop.unsqueeze(0),    # [1, 75]  本体感受信息
                    scan_feat,                 # [1, 225] 雷达扫描数据
                    priv_explicit,             # [1, 3]   特权显式信息
                    priv_latent,               # [1, 29]  特权隐式信息
                    hist_flat                  # [1, 750] 历史本体感受信息
                ], dim=1)  # 最终维度：[1, 1082]
                
                # ========== 调试信息：验证观测维度 ==========
                # 计算期望的观测维度：75 + 225 + 3 + 29 + 750 = 1082
                expected_obs_dim = jit_n_proprio + jit_num_scan + jit_n_priv_explicit + jit_n_priv_latent + obs_history_len * jit_n_proprio
                if beamdojo_obs.shape[1] != expected_obs_dim:
                    print(f"[sim2sim ERROR] Obs dim mismatch: got {beamdojo_obs.shape[1]}, expected {expected_obs_dim}")
                    print(f"  Breakdown: prop={jit_prop.shape[0]}, scan={scan_feat.shape[1]}, priv_explicit={priv_explicit.shape[1]}, priv_latent={priv_latent.shape[1]}, history={hist_flat.shape[1]}")
                elif control_tick == 1:
                    # 第一次控制时打印观测维度信息
                    print(f"[sim2sim] Obs dim OK: {beamdojo_obs.shape[1]} (prop={jit_n_proprio}, scan={jit_num_scan}, priv_explicit={jit_n_priv_explicit}, priv_latent={jit_n_priv_latent}, history={obs_history_len*jit_n_proprio})")
                    print(f"[sim2sim] Using lidar: {use_lidar}, mode: {lidar_mode}, scan shape: {scan_feat.shape}")
                    if lidar_mode == "terrain":
                        # 打印地形扫描数据的统计信息，用于验证raycast是否正常工作
                        scan_np_debug = scan_feat.squeeze().numpy()
                        print(f"[sim2sim] Terrain scan stats: min={scan_np_debug.min():.3f}, max={scan_np_debug.max():.3f}, mean={scan_np_debug.mean():.3f}, std={scan_np_debug.std():.3f}")

                # 2.6 构建深度图像潜在特征（depth_latent）：用于策略网络的第二个输入
                # 如果启用深度相机，从最近2帧深度图像池化得到32维特征
                if use_depth and torch.count_nonzero(depth_image_buffer).item() > 0:
                    depth_slice = depth_image_buffer[:, -2:, ...]  # 取最近2帧 [1, 2, 64, 64]
                    # 池化到4x4，然后展平：[1, 2*4*4] = [1, 32]
                    depth_latent = torch.nn.functional.adaptive_avg_pool2d(depth_slice.reshape(1, 2, 64, 64), (4, 4)).reshape(1, -1)
                    if depth_latent.shape[1] < jit_depth_latent_dim:
                        # 如果维度不足，用0填充
                        depth_latent = torch.nn.functional.pad(depth_latent, (0, jit_depth_latent_dim - depth_latent.shape[1]))
                    else:
                        # 如果维度过多，截断
                        depth_latent = depth_latent[:, :jit_depth_latent_dim]
                else:
                    depth_latent = torch.zeros(1, jit_depth_latent_dim)  # 默认零向量 [1, 32]

                # ========== 第三步：调用策略网络，获取动作输出 ==========
                # 策略网络输入：
                #   - beamdojo_obs: [1, 1082] - 完整观测向量
                #   - depth_latent: [1, 32]   - 深度图像潜在特征
                # 策略网络输出：
                #   - full_action: [27] - 27个关节的目标角度（归一化到[-1, 1]）
                try:
                    # 调用JIT编译的策略网络进行前向推理
                    full_action = policy(beamdojo_obs, depth_latent).detach().numpy().squeeze()
                    if control_tick == 1:
                        print(f"[sim2sim] Policy output shape: {full_action.shape}, expected: (27,)")
                except Exception as e:
                    print(f"[sim2sim ERROR] Policy call failed: {e}")
                    print(f"  beamdojo_obs shape: {beamdojo_obs.shape}, depth_latent shape: {depth_latent.shape}")
                    raise
                
                # 3.1 提取下半身12个关节的动作（前num_actions个）
                # 策略输出是归一化的动作值（-1到1），需要缩放到实际角度范围
                lower_body_action = np.clip(full_action[:num_actions], -1.0, 1.0)  # [12]
                
                # ========== 第四步：将动作应用到机器人 ==========
                # 动作应用流程：策略输出 -> 目标关节角度 -> PD控制 -> 关节力矩 -> MuJoCo执行
                
                # 4.1 平滑过渡：前50步使用线性插值，避免突然的动作变化
                # 这有助于在启动时平滑过渡，避免机器人突然运动
                alpha = min(1.0, control_tick / 50.0)  # 从0逐渐增加到1（前50步）
                target_dof_pos = default_angles.copy()  # 初始化目标角度为默认角度（所有27个关节）
                
                # 4.2 计算期望的关节角度
                # 注意：controlled_qpos_indices是相对于d.qpos[7:]的索引
                # d.qpos结构：[基座位置(3) | 基座四元数(4) | 关节角度(27)]
                # d.qpos[7:] = 所有27个关节角度
                current_lower_q = d.qpos[7:][controlled_qpos_indices]  # 当前受控关节的角度（12个）
                
                # 将归一化动作转换为实际角度
                # 公式：desired_angle = action * action_scale + default_angle
                # action范围：[-1, 1]，action_scale通常为0.25
                # 例如：action=1.0 -> desired_angle = 0.25 + default_angle
                desired_lower_q = lower_body_action * action_scale + default_angles[controlled_qpos_indices]
                
                # 平滑插值：target = (1-alpha) * current + alpha * desired
                # alpha=0时：target = current（保持当前角度）
                # alpha=1时：target = desired（完全使用期望角度）
                target_dof_pos[controlled_qpos_indices] = (1 - alpha) * current_lower_q + alpha * desired_lower_q
                
                # 注意：target_dof_pos包含所有27个关节，但只有前12个（controlled_qpos_indices）会被更新
                # 其他关节保持default_angles的值（通常为0）
                
                # 4.3 保存当前动作（用于下一时刻的观测）
                # 动作会被用于构建下一时刻的观测向量（obs中的action部分）
                action = lower_body_action

                # goal visualization (top-down map)
                path_history.append(base_pos.copy())
                if len(path_history) > 1000:
                    path_history = path_history[-1000:]
                canvas = np.ones((map_size[0], map_size[1], 3), dtype=np.uint8) * 255
                for g in goals:
                    gx, gy = world_to_pixel(g)
                    cv2.circle(canvas, (gx, gy), 6, (0, 0, 255), -1)
                for i in range(1, len(path_history)):
                    x0, y0 = world_to_pixel(path_history[i - 1])
                    x1, y1 = world_to_pixel(path_history[i])
                    cv2.line(canvas, (x0, y0), (x1, y1), (0, 200, 0), 2)
                rx, ry = world_to_pixel(base_pos)
                cv2.circle(canvas, (rx, ry), 6, (0, 0, 0), -1)
                if cur_goal_idx < len(goals):
                    gx, gy = world_to_pixel(goals[cur_goal_idx])
                    cv2.line(canvas, (rx, ry), (gx, gy), (200, 0, 0), 1)
                cv2.putText(canvas, f"goal {cur_goal_idx+1}/{len(goals)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
                cv2.putText(canvas, f"cmd [{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f}]", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
                cv2.imshow('goal map', canvas)
                cv2.waitKey(1)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
