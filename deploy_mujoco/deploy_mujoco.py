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

def compute_proprioceptive_obs(d, config, action, cmd):
    n_joints = d.qpos.shape[0] - 7
    qj = d.qpos[7:7+n_joints]
    dqj = d.qvel[6:6+n_joints]
    omega = d.qvel[3:6]
    quat = d.qpos[3:7]
    
    default_angles = config['default_angles']
    if len(default_angles) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(default_angles)] = default_angles
    else:
        padded_defaults = default_angles[:n_joints]
    
    proprio_obs = np.zeros(75, dtype=np.float32)
    proprio_obs[0:3] = cmd[:3] * config['cmd_scale']
    proprio_obs[3:6] = omega * config['ang_vel_scale']
    proprio_obs[6:9] = get_gravity_orientation(quat)
    proprio_obs[9:9+n_joints] = (qj - padded_defaults) * config['dof_pos_scale']
    proprio_obs[9+n_joints:9+2*n_joints] = dqj * config['dof_vel_scale']
    proprio_obs[9+2*n_joints:9+2*n_joints+12] = action[:12]  # 修复：使用正确的结束索引
    
    return proprio_obs

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf):
    # 本体感受观测（75维）
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd)
    
    # 高度观测（225维）：heights = base_height - measured_heights
    num_scan = scan_points_body.shape[0]
    if config.get('measure_heights', True) and config.get('use_lidar', True) and config.get('lidar_mode') == 'terrain':
        measured_heights = get_scan_from_terrain(m, d, scan_points_body)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    # 特权信息（推理时设为0）
    priv_explicit = np.zeros(3, dtype=np.float32)
    priv_latent = np.zeros(29, dtype=np.float32)
    
    # 拼接：proprio(75) + heights(225) + priv_explicit(3) + priv_latent(29) + history(750)
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    
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
    obs_history_buf = np.zeros((obs_history_len, 75), dtype=np.float32)
    
    # 计算总观测维度
    total_obs_dim = 75 + num_scan + 3 + 29 + obs_history_len * 75
    print(f"\n{'='*60}")
    print("观测维度配置:")
    print(f"  本体感受观测 (proprio): 75 维")
    print(f"  高度扫描 (heights): {num_scan} 维")
    print(f"  特权显式 (priv_explicit): 3 维")
    print(f"  特权隐式 (priv_latent): 29 维")
    print(f"  历史信息 (history): {obs_history_len} × 75 = {obs_history_len * 75} 维")
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
                # 计算观测并更新历史
                full_obs, proprio_obs = compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf)
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
                    print(f"{'='*60}")
                    print(f"\n输入观测 (维度: {full_obs.shape[0]}):")
                    print(f"  本体感受观测 (proprio, 0-74): {full_obs[0:75]}")
                    print(f"  高度扫描 (heights, 75-299): {full_obs[75:75+num_scan]}")
                    print(f"  特权显式 (priv_explicit, 300-302): {full_obs[75+num_scan:75+num_scan+3]}")
                    print(f"  特权隐式 (priv_latent, 303-331): {full_obs[75+num_scan+3:75+num_scan+3+29]}")
                    print(f"  历史信息 (history, 332-1081): {full_obs[75+num_scan+3+29:]}")
                    print(f"\n输出动作 (维度: {action.shape[0]}):")
                    print(f"  {action}")
                    print(f"{'='*60}\n")
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
