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
    
    config.setdefault('use_lidar', True)
    config.setdefault('lidar_mode', 'terrain')
    config.setdefault('measure_heights', True)
    config.setdefault('obs_history_len', 10)
    config.setdefault('print_period', 100)
    config.setdefault('upper_body_kps', 300.0)
    config.setdefault('upper_body_kds', 5.0)
    config.setdefault('goal_init', [2.0, 2.0, 0.78])
    config.setdefault('goal_reach_threshold', 0.5)
    config.setdefault('goal_dynamic_update', True)
    config.setdefault('goal_forward_distance_range', [1.5, 3.0])
    
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
    
    R = quat_to_rot(base_quat)
    scan_points_world = (R @ scan_points_body.T).T + base_pos
    
    terrain_heights = np.zeros(num_points, dtype=np.float32)
    ray_start = scan_points_world.copy()
    ray_start[:, 2] += 2.0
    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ray_length = 4.0
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
    num_actions = config['num_actions']
    qj = d.qpos[7:7+num_actions]
    dqj = d.qvel[6:6+num_actions]
    omega = d.qvel[3:6]
    quat = d.qpos[3:7]
    base_pos = d.qpos[:3]
    
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    target_vec = cur_goal[:2] - base_pos[:2]
    target_yaw = np.arctan2(target_vec[1], target_vec[0])
    delta_yaw = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
    cmd[2] = np.clip(0.8 * delta_yaw, -0.5, 0.5)
    
    delta_pose_x = cur_goal[0] - base_pos[0]
    delta_pose_y = cur_goal[1] - base_pos[1]
    default_angles = config['default_angles']
    
    proprio_obs = np.zeros(48, dtype=np.float32)
    idx = 0
    
    proprio_obs[idx:idx+3] = cmd[:3] * config['cmd_scale']
    idx += 3
    proprio_obs[idx:idx+3] = omega * config['ang_vel_scale']
    idx += 3
    proprio_obs[idx] = delta_yaw * config['delta_yaw_scale']
    idx += 1
    proprio_obs[idx] = delta_pose_x * config['delta_pose_scale']
    idx += 1
    proprio_obs[idx] = delta_pose_y * config['delta_pose_scale']
    idx += 1
    proprio_obs[idx:idx+3] = get_gravity_orientation(quat)
    idx += 3
    proprio_obs[idx:idx+num_actions] = (qj - default_angles) * config['dof_pos_scale']
    idx += num_actions
    proprio_obs[idx:idx+num_actions] = dqj * config['dof_vel_scale']
    idx += num_actions
    proprio_obs[idx:idx+num_actions] = action[:num_actions]
    idx += num_actions
    
    return proprio_obs

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal):
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd, cur_goal)
    
    num_scan = scan_points_body.shape[0]
    if config.get('measure_heights', True) and config.get('use_lidar', True) and config.get('lidar_mode') == 'terrain':
        measured_heights = get_scan_from_terrain(m, d, scan_points_body)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    priv_explicit = np.zeros(3, dtype=np.float32)
    priv_latent = np.zeros(29, dtype=np.float32)
    
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    return full_obs, proprio_obs

def update_visualization_markers(m, d, start_pos, cur_goal):
    robot_pos = d.qpos[:3]
    z_height = robot_pos[2]
    
    def update_marker(name, pos):
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            mocap_idx = m.body_mocapid[body_id]
            if mocap_idx >= 0:
                d.mocap_pos[mocap_idx] = pos
    
    update_marker("viz_start", [start_pos[0], start_pos[1], z_height])
    update_marker("viz_goal", [cur_goal[0], cur_goal[1], cur_goal[2]])
    update_marker("viz_robot", [robot_pos[0], robot_pos[1], z_height])
    
    for i in range(16):
        t = i / 15.0
        point_x = start_pos[0] + t * (cur_goal[0] - start_pos[0])
        point_y = start_pos[1] + t * (cur_goal[1] - start_pos[1])
        update_marker(f"viz_path_{i}", [point_x, point_y, z_height])

def main():
    config_path = os.path.join('deploy_mujoco/configs/g1.yaml')
    config = load_config(config_path)
    
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    n_joints = d.qpos.shape[0] - 7
    num_actions = config['num_actions']
    num_upper_joints = n_joints - num_actions if n_joints > num_actions else 0
    
    scan_range = config.get('scan_range', 0.7)
    scan_resolution = config.get('scan_resolution', 15)
    scan_xy = np.linspace(-scan_range, scan_range, scan_resolution, dtype=np.float32)
    scan_points_body = np.zeros((scan_resolution * scan_resolution, 3), dtype=np.float32)
    scan_points_body[:, :2] = np.stack(np.meshgrid(scan_xy, scan_xy), axis=-1).reshape(-1, 2)
    
    print(f"MuJoCo部署 | 控制频率: {1/config['simulation_dt']/config['control_decimation']:.0f}Hz | 目标: ({config['goal_init'][0]:.1f}, {config['goal_init'][1]:.1f})")
    
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    obs_history_buf = np.zeros((config['obs_history_len'], 48), dtype=np.float32)
    
    cur_goal = config['goal_init'].copy().astype(np.float32)
    goal_reach_threshold = config['goal_reach_threshold']
    goal_dynamic_update = config['goal_dynamic_update']
    goal_distance_range = config['goal_forward_distance_range']
    goal_start_pos = d.qpos[:2].copy().astype(np.float32)
    
    print(f"加载策略: {os.path.basename(config['policy_path'])}")
    policy = torch.jit.load(config['policy_path'])
    policy.eval()
    print("运行中...\n")
    
    counter = 0
    control_counter = 0
    print_period = config.get('print_period', 100)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            
            leg_tau = pd_control(target_dof_pos, d.qpos[7:7+num_actions], config['kps'],
                                np.zeros_like(config['kps']), d.qvel[6:6+num_actions], config['kds'])
            d.ctrl[:num_actions] = leg_tau
            
            if n_joints > num_actions:
                upper_tau = pd_control(np.zeros(num_upper_joints), d.qpos[7+num_actions:7+n_joints],
                                      np.full(num_upper_joints, config['upper_body_kps']), np.zeros(num_upper_joints),
                                      d.qvel[6+num_actions:6+n_joints], np.full(num_upper_joints, config['upper_body_kds']))
                d.ctrl[num_actions:num_actions+num_upper_joints] = upper_tau
            
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                robot_pos = d.qpos[:2]
                distance_to_goal = np.linalg.norm(cur_goal[:2] - robot_pos)
                
                if goal_dynamic_update and distance_to_goal < goal_reach_threshold:
                    w, x, y, z = d.qpos[3:7]
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    forward_dist = np.random.uniform(goal_distance_range[0], goal_distance_range[1])
                    angle_range = config.get('goal_angle_range', [-np.pi/2, np.pi/2])
                    angle_offset = np.random.uniform(angle_range[0], angle_range[1])
                    target_yaw = yaw + angle_offset
                    
                    cur_goal[0] = robot_pos[0] + forward_dist * np.cos(target_yaw)
                    cur_goal[1] = robot_pos[1] + forward_dist * np.sin(target_yaw)
                    cur_goal[2] = config['goal_init'][2]
                    goal_start_pos = robot_pos[:2].copy()
                    
                    print(f"[目标更新] ({cur_goal[0]:.2f}, {cur_goal[1]:.2f}) | 距离:{forward_dist:.2f}m, 角度:{np.degrees(angle_offset):+.0f}°")
                
                full_obs, proprio_obs = compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal)
                obs_history_buf = np.roll(obs_history_buf, -1, axis=0)
                obs_history_buf[-1] = proprio_obs
                
                with torch.no_grad():
                    action = policy(torch.from_numpy(full_obs).unsqueeze(0)).detach().numpy().squeeze()
                
                target_dof_pos = action[:num_actions] * config['action_scale'] + config['default_angles']
                control_counter += 1
                
                if control_counter % print_period == 0:
                    print(f"[{control_counter:5d}] 位置:({robot_pos[0]:5.2f},{robot_pos[1]:5.2f}) 目标:({cur_goal[0]:5.2f},{cur_goal[1]:5.2f}) 距离:{distance_to_goal:.2f}m")
            
            update_visualization_markers(m, d, goal_start_pos, cur_goal)
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
