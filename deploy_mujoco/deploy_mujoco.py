import os
import time
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for key in ['policy_path', 'xml_path']:
        if key in config:
            path = config[key]
            if path.startswith('./'):
                config[key] = os.path.join(LEGGED_GYM_ROOT_DIR, path[2:])
            elif '{LEGGED_GYM_ROOT_DIR}' in path:
                config[key] = path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
                
    for key in ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init', 'goal_init']:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    
    # 设置默认值
    config.setdefault('use_lidar', True)
    config.setdefault('lidar_mode', 'terrain')
    config.setdefault('measure_heights', True)
    config.setdefault('obs_history_len', 10)
    config.setdefault('print_period', 100)
    config.setdefault('upper_body_kps', 300.0)  # 上肢关节PD位置增益（用于固定到0）
    config.setdefault('upper_body_kds', 5.0)     # 上肢关节PD速度增益（用于固定到0）
    config.setdefault('goal_init', [2.0, 0.0, 0.78])  # 初始目标点 [x, y, z]
    config.setdefault('goal_reach_threshold', 0.5)  # 到达目标距离阈值（米）
    config.setdefault('goal_dynamic_update', True)  # 是否启用动态目标更新
    config.setdefault('goal_forward_distance_range', [1.5, 3.0])  # 新目标距离范围 [min, max]
    
    return config

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)

def quat_rotate_inverse(q, v):
    R = quat_to_rot(q)
    return (R.T @ v).astype(np.float32)

def get_gravity_orientation(quat):
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_scan_from_terrain(m, d, scan_points_body):
    num_points = scan_points_body.shape[0]
    base_pos = d.qpos[:3]
    base_quat = d.qpos[3:7]
    
    # 转换到世界坐标系
    R = quat_to_rot(base_quat)
    scan_points_world = (R @ scan_points_body.T).T + base_pos
    
    # Raycast查询地形高度
    terrain_heights = np.zeros(num_points, dtype=np.float32)
    ray_start = scan_points_world.copy()
    ray_start[:, 2] += 2.0
    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ray_length = 4.0
    
    # 创建 geomid 输出数组（新版本 MuJoCo API）
    geomid = np.array([-1], dtype=np.int32)
    
    for i in range(num_points):
        distance = mujoco.mj_ray(m, d, ray_start[i], ray_dir, 
                                 geomgroup=None, flg_static=1, bodyexclude=-1, geomid=geomid)
        if geomid[0] >= 0 and distance < ray_length:
            terrain_heights[i] = (ray_start[i] + ray_dir * distance)[2]
        else:
            terrain_heights[i] = base_pos[2] - 2.0
    
    return terrain_heights

def compute_proprioceptive_obs(d, config, action, cmd, cur_goal):
    """
    计算本体感受观测（48维，只包含下肢12个关节）
    结构与legged_gym一致：
    - commands (3): vx, vy, vyaw (scaled)
    - ang_vel (3): 角速度 (scaled)
    - delta_yaw (1): 朝向误差 (scaled)
    - delta_pose_x (1): X方向位置误差 (scaled)
    - delta_pose_y (1): Y方向位置误差 (scaled)
    - gravity (3): 重力方向
    - dof_pos (12): 关节位置偏移 (scaled，只下肢12个关节)
    - dof_vel (12): 关节速度 (scaled，只下肢12个关节)
    - action_history (12): 上一步动作（仅下半身12个关节）
    """
    num_actions = config['num_actions']  # 12个下肢关节
    # 只使用前12个关节（下肢）
    qj = d.qpos[7:7+num_actions]  # 只取前12个关节
    dqj = d.qvel[6:6+num_actions]  # 只取前12个关节
    omega = d.qvel[3:6]
    quat = d.qpos[3:7]
    base_pos = d.qpos[:3]
    
    # 获取yaw角（从四元数提取）
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # 计算目标朝向（目标点相对于机器人的角度）
    target_vec = cur_goal[:2] - base_pos[:2]
    target_yaw = np.arctan2(target_vec[1], target_vec[0])
    
    # 计算朝向误差（归一化到[-π, π]）
    delta_yaw = target_yaw - yaw
    delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
    
    # 计算位置误差
    delta_pose_x = cur_goal[0] - base_pos[0]
    delta_pose_y = cur_goal[1] - base_pos[1]
    
    default_angles = config['default_angles']  # 应该已经是12个
    
    # 构建48维观测
    proprio_obs = np.zeros(48, dtype=np.float32)
    idx = 0
    
    # commands (3): [vx, vy, vyaw]
    proprio_obs[idx:idx+3] = cmd[:3] * config['cmd_scale']
    idx += 3
    
    # ang_vel (3)
    proprio_obs[idx:idx+3] = omega * config['ang_vel_scale']
    idx += 3
    
    # delta_yaw (1)
    proprio_obs[idx] = delta_yaw * config['delta_yaw_scale']
    idx += 1
    
    # delta_pose_x (1)
    proprio_obs[idx] = delta_pose_x * config['delta_pose_scale']
    idx += 1
    
    # delta_pose_y (1)
    proprio_obs[idx] = delta_pose_y * config['delta_pose_scale']
    idx += 1
    
    # gravity (3)
    proprio_obs[idx:idx+3] = get_gravity_orientation(quat)
    idx += 3
    
    # dof_pos (12): 只使用下肢12个关节
    proprio_obs[idx:idx+num_actions] = (qj - default_angles) * config['dof_pos_scale']
    idx += num_actions
    
    # dof_vel (12): 只使用下肢12个关节
    proprio_obs[idx:idx+num_actions] = dqj * config['dof_vel_scale']
    idx += num_actions
    
    # action_history (12): 只保存下半身12个关节的动作
    proprio_obs[idx:idx+num_actions] = action[:num_actions]
    idx += num_actions
    
    assert idx == 48, f"观测维度错误：期望48，实际{idx}"
    
    return proprio_obs

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal):
    """
    计算完整观测
    结构：proprio(48) + heights(225) + priv_explicit(3) + priv_latent(29) + history(480)
    总维度：48 + 225 + 3 + 29 + 480 = 785
    """
    # 本体感受观测（48维，只包含下肢12个关节）
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd, cur_goal)
    
    # 高度观测（225维）：heights = base_height - measured_heights
    num_scan = scan_points_body.shape[0]
    if config.get('measure_heights', True) and config.get('use_lidar', True) and config.get('lidar_mode') == 'terrain':
        measured_heights = get_scan_from_terrain(m, d, scan_points_body)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    # 特权信息（推理时设为0）
    priv_explicit = np.zeros(3, dtype=np.float32)  # base_lin_vel
    priv_latent = np.zeros(29, dtype=np.float32)   # 质量参数等
    
    # 拼接：proprio(48) + heights(225) + priv_explicit(3) + priv_latent(29) + history(480)
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    
    assert full_obs.shape[0] == 48 + 225 + 3 + 29 + 480, f"观测维度错误：期望785，实际{full_obs.shape[0]}"
    
    return full_obs, proprio_obs

def main():
    config_path = os.path.join('deploy_mujoco/configs/g1.yaml')
    config = load_config(config_path)
    
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    n_joints = d.qpos.shape[0] - 7  # 总关节数（可能包含上肢）
    num_actions = config['num_actions']  # 只控制下肢12个关节
    num_upper_joints = n_joints - num_actions if n_joints > num_actions else 0
    print(f"\n{'='*60}")
    print("机器人初始化信息:")
    print(f"  总关节数量: {n_joints}")
    print(f"  动作维度: {num_actions} (策略控制下肢12个关节)")
    if num_upper_joints > 0:
        print(f"  上肢关节: {num_upper_joints} 个 (通过PD控制固定在0位置，不由策略控制)")
    print(f"  仿真时间步长: {config['simulation_dt']} s ({1/config['simulation_dt']:.0f} Hz)")
    print(f"  控制降采样率: {config['control_decimation']}")
    print(f"  控制频率: {1/config['simulation_dt']/config['control_decimation']:.0f} Hz")
    
    # 初始化扫描点
    scan_range = config.get('scan_range', 0.7)
    scan_resolution = config.get('scan_resolution', 15)
    scan_xy = np.linspace(-scan_range, scan_range, scan_resolution, dtype=np.float32)
    scan_points_body = np.zeros((scan_resolution * scan_resolution, 3), dtype=np.float32)
    scan_points_body[:, :2] = np.stack(np.meshgrid(scan_xy, scan_xy), axis=-1).reshape(-1, 2)
    num_scan = scan_points_body.shape[0]
    
    print(f"\n扫描点配置:")
    print(f"  扫描范围: [-{scan_range}, {scan_range}] m")
    print(f"  扫描分辨率: {scan_resolution}*{scan_resolution} = {num_scan} 个点")
    
    print(f"\n目标点配置:")
    print(f"  初始目标点: ({config['goal_init'][0]:.2f}, {config['goal_init'][1]:.2f}, {config['goal_init'][2]:.2f}) m")
    print(f"  到达阈值: {config['goal_reach_threshold']} m")
    print(f"  动态更新: {'启用' if config['goal_dynamic_update'] else '禁用'}")
    if config['goal_dynamic_update']:
        print(f"  新目标距离范围: [{config['goal_forward_distance_range'][0]}, {config['goal_forward_distance_range'][1]}] m")
    
    # 初始化变量
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    obs_history_len = config['obs_history_len']
    obs_history_buf = np.zeros((obs_history_len, 48), dtype=np.float32)  # 48维（只包含下肢12个关节）
    
    # 初始化目标点（从配置文件读取）
    cur_goal = config['goal_init'].copy().astype(np.float32)
    goal_reach_threshold = config['goal_reach_threshold']
    goal_dynamic_update = config['goal_dynamic_update']
    goal_distance_range = config['goal_forward_distance_range']
    
    # 计算总观测维度
    total_obs_dim = 48 + num_scan + 3 + 29 + obs_history_len * 48
    print(f"\n{'='*60}")
    print("观测维度配置:")
    print(f"  本体感受观测 (proprio): 48 维（只包含下肢12个关节）")
    print(f"    - commands: 3, ang_vel: 3, delta_yaw: 1, delta_pose_x: 1, delta_pose_y: 1")
    print(f"    - gravity: 3, dof_pos: 12, dof_vel: 12, action_history: 12")
    print(f"  高度扫描 (heights): {num_scan} 维")
    print(f"  特权显式 (priv_explicit): 3 维")
    print(f"  特权隐式 (priv_latent): 29 维")
    print(f"  历史信息 (history): {obs_history_len} × 48 = {obs_history_len * 48} 维")
    print(f"  总观测维度 (total): {total_obs_dim} 维")
    print(f"{'='*60}\n")
    
    # 加载策略模型
    print(f"加载策略模型: {config['policy_path']}")
    policy = torch.jit.load(config['policy_path'])
    policy.eval()
    print("策略模型加载成功!\n")
    
    counter = 0
    control_counter = 0  # 控制周期计数器
    print_period = config.get('print_period', 100)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            
            # PD控制腿部关节（只控制下肢12个关节，由策略控制）
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+num_actions],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+num_actions],
                config['kds']
            )
            d.ctrl[:num_actions] = leg_tau
            
            # 固定上肢关节到默认位置（0），通过PD控制固定，不由策略控制
            if n_joints > num_actions:
                num_upper_joints = n_joints - num_actions
                # 上肢目标位置为0（固定在原点）
                upper_target_pos = np.zeros(num_upper_joints, dtype=np.float32)
                upper_current_pos = d.qpos[7+num_actions:7+n_joints]
                upper_current_vel = d.qvel[6+num_actions:6+n_joints]
                
                # PD控制固定上肢到0位置
                upper_tau = pd_control(
                    upper_target_pos,
                    upper_current_pos,
                    np.full(num_upper_joints, config['upper_body_kps']),
                    np.zeros(num_upper_joints),
                    upper_current_vel,
                    np.full(num_upper_joints, config['upper_body_kds'])
                )
                d.ctrl[num_actions:num_actions+num_upper_joints] = upper_tau
            
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                # 更新目标点（如果启用动态更新）
                robot_pos = d.qpos[:2]
                distance_to_goal = np.linalg.norm(cur_goal[:2] - robot_pos)
                
                if goal_dynamic_update and distance_to_goal < goal_reach_threshold:
                    # 到达目标，在机器人前方生成新目标
                    w, x, y, z = d.qpos[3:7]
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    forward_dist = np.random.uniform(goal_distance_range[0], goal_distance_range[1])
                    cur_goal[0] = robot_pos[0] + forward_dist * np.cos(yaw)
                    cur_goal[1] = robot_pos[1] + forward_dist * np.sin(yaw)
                    # 保持目标高度不变（或从地形采样）
                    cur_goal[2] = config['goal_init'][2]  # 使用初始高度
                    print(f"[目标更新] 新目标: ({cur_goal[0]:.2f}, {cur_goal[1]:.2f}, {cur_goal[2]:.2f})")
                
                # 计算观测并更新历史
                full_obs, proprio_obs = compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal)
                obs_history_buf = np.roll(obs_history_buf, -1, axis=0)
                obs_history_buf[-1] = proprio_obs
                
                # 策略推理
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(full_obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()
                
                target_dof_pos = action[:num_actions] * config['action_scale'] + config['default_angles']
                
                control_counter += 1
                
                # 每 print_period 个控制周期打印一次观测和动作
                if control_counter % print_period == 0:
                    print(f"\n{'='*60}")
                    print(f"控制周期 #{control_counter}")
                    print(f"  机器人位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
                    print(f"  目标位置: ({cur_goal[0]:.2f}, {cur_goal[1]:.2f})")
                    print(f"  距离目标: {distance_to_goal:.2f} m")
                    print(f"{'='*60}")
                    print(f"\n输入观测 (维度: {full_obs.shape[0]}):")
                    print(f"  本体感受观测 (proprio, 0-47): {full_obs[0:48]}")
                    print(f"  高度扫描 (heights, 48-272): {full_obs[48:48+num_scan]}")
                    print(f"  特权显式 (priv_explicit, 273-275): {full_obs[48+num_scan:48+num_scan+3]}")
                    print(f"  特权隐式 (priv_latent, 276-304): {full_obs[48+num_scan+3:48+num_scan+3+29]}")
                    print(f"  历史信息 (history, 305-784): {full_obs[48+num_scan+3+29:]}")
                    print(f"\n输出动作 (维度: {action.shape[0]}):")
                    print(f"  {action}")
                    print(f"{'='*60}\n")
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
