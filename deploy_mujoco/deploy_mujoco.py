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
                
    for key in ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    
    # 设置默认值
    config.setdefault('use_lidar', True)
    config.setdefault('lidar_mode', 'terrain')
    config.setdefault('measure_heights', True)
    config.setdefault('obs_history_len', 10)
    config.setdefault('print_period', 100) 
    
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
    计算本体感受观测（78维）
    结构与legged_gym一致：
    - commands (3): vx, vy, vyaw (scaled)
    - ang_vel (3): 角速度 (scaled)
    - delta_yaw (1): 朝向误差 (scaled)
    - delta_pose_x (1): X方向位置误差 (scaled)
    - delta_pose_y (1): Y方向位置误差 (scaled)
    - gravity (3): 重力方向
    - dof_pos (27): 关节位置偏移 (scaled)
    - dof_vel (27): 关节速度 (scaled)
    - action_history (12): 上一步动作（仅下半身12个关节）
    """
    n_joints = d.qpos.shape[0] - 7
    qj = d.qpos[7:7+n_joints]
    dqj = d.qvel[6:6+n_joints]
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
    
    default_angles = config['default_angles']
    if len(default_angles) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(default_angles)] = default_angles
    else:
        padded_defaults = default_angles[:n_joints]
    
    # 构建78维观测
    proprio_obs = np.zeros(78, dtype=np.float32)
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
    
    # dof_pos (27)
    proprio_obs[idx:idx+n_joints] = (qj - padded_defaults) * config['dof_pos_scale']
    idx += n_joints
    
    # dof_vel (27)
    proprio_obs[idx:idx+n_joints] = dqj * config['dof_vel_scale']
    idx += n_joints
    
    # action_history (12): 只保存下半身12个关节的动作
    proprio_obs[idx:idx+12] = action[:12]
    idx += 12
    
    assert idx == 78, f"观测维度错误：期望78，实际{idx}"
    
    return proprio_obs

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal):
    """
    计算完整观测
    结构：proprio(78) + heights(225) + priv_explicit(3) + priv_latent(29) + history(780)
    总维度：78 + 225 + 3 + 29 + 780 = 1115
    """
    # 本体感受观测（78维）
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
    
    # 拼接：proprio(78) + heights(225) + priv_explicit(3) + priv_latent(29) + history(780)
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    
    assert full_obs.shape[0] == 78 + 225 + 3 + 29 + 780, f"观测维度错误：{full_obs.shape[0]}"
    
    return full_obs, proprio_obs

def main():
    config_path = os.path.join('deploy_mujoco/configs/g1.yaml')
    config = load_config(config_path)
    
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    n_joints = d.qpos.shape[0] - 7
    num_actions = config['num_actions']
    print(f"\n{'='*60}")
    print("机器人初始化信息:")
    print(f"  关节数量: {n_joints}")
    print(f"  动作维度: {num_actions}")
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
    
    # 初始化变量
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    obs_history_len = config['obs_history_len']
    obs_history_buf = np.zeros((obs_history_len, 78), dtype=np.float32)  # 修改为78维
    
    # 初始化目标点（机器人前方2米处）
    cur_goal = np.array([2.0, 0.0, 0.78], dtype=np.float32)
    
    # 计算总观测维度
    total_obs_dim = 78 + num_scan + 3 + 29 + obs_history_len * 78
    print(f"\n{'='*60}")
    print("观测维度配置:")
    print(f"  本体感受观测 (proprio): 78 维")
    print(f"    - commands: 3, ang_vel: 3, delta_yaw: 1, delta_pose_x: 1, delta_pose_y: 1")
    print(f"    - gravity: 3, dof_pos: 27, dof_vel: 27, action_history: 12")
    print(f"  高度扫描 (heights): {num_scan} 维")
    print(f"  特权显式 (priv_explicit): 3 维")
    print(f"  特权隐式 (priv_latent): 29 维")
    print(f"  历史信息 (history): {obs_history_len} × 78 = {obs_history_len * 78} 维")
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
            
            # PD控制腿部关节
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+num_actions],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+num_actions],
                config['kds']
            )
            d.ctrl[:num_actions] = leg_tau
            
            # 控制其他关节
            if n_joints > num_actions:
                arm_tau = pd_control(
                    np.zeros(n_joints - num_actions),
                    d.qpos[7+num_actions:7+n_joints],
                    np.full(n_joints-num_actions, 100.0),
                    np.zeros(n_joints-num_actions),
                    d.qvel[6+num_actions:6+n_joints],
                    np.full(n_joints-num_actions, 0.5)
                )
                if d.ctrl.shape[0] > num_actions:
                    d.ctrl[num_actions:] = arm_tau
            
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                # 更新目标点（可选：动态更新目标，这里暂时固定）
                # 检查是否到达目标
                robot_pos = d.qpos[:2]
                distance_to_goal = np.linalg.norm(cur_goal[:2] - robot_pos)
                if distance_to_goal < 0.5:  # 到达目标，生成新目标
                    # 在机器人前方生成新目标（示例）
                    yaw = np.arctan2(2.0 * (d.qpos[6] * d.qpos[5] + d.qpos[3] * d.qpos[4]), 
                                     1.0 - 2.0 * (d.qpos[4]**2 + d.qpos[5]**2))
                    forward_dist = np.random.uniform(1.5, 3.0)
                    cur_goal[0] = robot_pos[0] + forward_dist * np.cos(yaw)
                    cur_goal[1] = robot_pos[1] + forward_dist * np.sin(yaw)
                    print(f"[目标更新] 新目标: ({cur_goal[0]:.2f}, {cur_goal[1]:.2f})")
                
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
                    print(f"  本体感受观测 (proprio, 0-77): {full_obs[0:78]}")
                    print(f"  高度扫描 (heights, 78-302): {full_obs[78:78+num_scan]}")
                    print(f"  特权显式 (priv_explicit, 303-305): {full_obs[78+num_scan:78+num_scan+3]}")
                    print(f"  特权隐式 (priv_latent, 306-334): {full_obs[78+num_scan+3:78+num_scan+3+29]}")
                    print(f"  历史信息 (history, 335-1114): {full_obs[78+num_scan+3+29:]}")
                    print(f"\n输出动作 (维度: {action.shape[0]}):")
                    print(f"  {action}")
                    print(f"{'='*60}\n")
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
