"""
MuJoCo 部署脚本（ONNX 版本）
功能：使用 ONNX 模型在 MuJoCo 中部署策略，验证 ONNX 模型是否正常工作

与 deploy_mujoco.py 的区别：
- 使用 ONNX Runtime 加载模型（而不是 JIT）
- 使用 ONNX Runtime 的推理 API（而不是 PyTorch）
- 其他功能完全相同（观测构造、控制循环等）
"""
import os
import time
import yaml
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from legged_gym import LEGGED_GYM_ROOT_DIR

def load_config(config_path):
    """
    加载配置文件
    
    @param config_path 配置文件路径
    @return 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理路径（支持相对路径和模板路径）
    for key in ['policy_path', 'xml_path']:
        if key in config:
            path = config[key]
            if path.startswith('./'):
                config[key] = os.path.join(LEGGED_GYM_ROOT_DIR, path[2:])
            elif '{LEGGED_GYM_ROOT_DIR}' in path:
                config[key] = path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
                
    # 转换数组类型
    for key in ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init', 'goal_init']:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    
    # 设置默认值
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
    """
    四元数转旋转矩阵
    
    @param q 四元数 [w, x, y, z]
    @return 旋转矩阵 (3x3)
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)

def quat_rotate_inverse(q, v):
    """
    四元数逆旋转（将世界坐标系向量转换到身体坐标系）
    
    @param q 四元数 [w, x, y, z]
    @param v 世界坐标系中的向量
    @return 身体坐标系中的向量
    """
    R = quat_to_rot(q)
    return (R.T @ v).astype(np.float32)

def get_gravity_orientation(quat):
    """
    获取重力方向（在身体坐标系中）
    
    @param quat 四元数 [w, x, y, z]
    @return 重力向量（在身体坐标系中）[gx, gy, gz]
    """
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    PD 控制器
    
    @param target_q 目标关节位置
    @param q 当前关节位置
    @param kp 位置增益
    @param target_dq 目标关节速度
    @param dq 当前关节速度
    @param kd 速度增益
    @return 控制力矩
    """
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_scan_from_terrain(m, d, scan_points_body):
    """
    从 MuJoCo 地形获取高度扫描数据
    
    @param m MuJoCo 模型
    @param d MuJoCo 数据
    @param scan_points_body 扫描点（在身体坐标系中）
    @return 地形高度数组
    """
    num_points = scan_points_body.shape[0]
    base_pos = d.qpos[:3]
    base_quat = d.qpos[3:7]
    
    # 将扫描点从身体坐标系转换到世界坐标系
    R = quat_to_rot(base_quat)
    scan_points_world = (R @ scan_points_body.T).T + base_pos
    
    terrain_heights = np.zeros(num_points, dtype=np.float32)
    ray_start = scan_points_world.copy()
    ray_start[:, 2] += 2.0  # 从上方 2 米开始
    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 向下
    ray_length = 4.0
    geomid = np.array([-1], dtype=np.int32)
    
    # 对每个扫描点进行射线检测
    for i in range(num_points):
        distance = mujoco.mj_ray(m, d, ray_start[i], ray_dir, 
                                 geomgroup=None, flg_static=1, bodyexclude=-1, geomid=geomid)
        if geomid[0] >= 0 and distance < ray_length:
            terrain_heights[i] = (ray_start[i] + ray_dir * distance)[2]
        else:
            terrain_heights[i] = base_pos[2] - 2.0  # 未检测到，使用默认值
    
    return terrain_heights

def compute_proprioceptive_obs(d, config, action, cmd, cur_goal, goal_start_pos, goal_started_moving, original_lin_vel_cmd):
    """
    计算本体感受观测（48 维）
    
    组成：
    - commands(3) + ang_vel(3) + delta_yaw(1) + delta_pose_x(1) + delta_pose_y(1)
    - + gravity(3) + dof_pos(12) + dof_vel(12) + action_history(12) = 48
    
    @param d MuJoCo 数据
    @param config 配置字典
    @param action 当前动作
    @param cmd 速度命令（会被修改）
    @param cur_goal 当前目标点
    @param goal_start_pos 目标起点位置
    @param goal_started_moving 是否已开始移动
    @param original_lin_vel_cmd 原始线速度命令
    @return 本体感受观测（48 维）
    """
    num_actions = config['num_actions']
    qj = d.qpos[7:7+num_actions]      # 关节位置（12 个）
    dqj = d.qvel[6:6+num_actions]     # 关节速度（12 个）
    omega = d.qvel[3:6]               # 身体角速度
    quat = d.qpos[3:7]                # 身体四元数
    base_pos = d.qpos[:3]             # 身体位置
    
    # 计算偏航角和朝向误差
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    target_vec = cur_goal[:2] - base_pos[:2]
    target_yaw = np.arctan2(target_vec[1], target_vec[0])
    delta_yaw = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
    
    # 计算角速度命令
    ang_vel_cmd = 0.8 * delta_yaw
    ang_vel_cmd = np.clip(ang_vel_cmd, -0.5, 0.5)
    cmd[2] = ang_vel_cmd
    
    # 计算 y 方向速度命令（基于偏离起点到目标直线的侧向偏差）
    y_vel_cmd = 0.0
    yaw_tolerance = config.get('yaw_tolerance_for_linear_vel', 0.15)
    heading_aligned = np.abs(delta_yaw) < yaw_tolerance
    
    if goal_started_moving:
        # 计算侧向偏移并生成 y 方向速度命令
        start_to_goal = cur_goal[:2] - goal_start_pos
        line_length = np.linalg.norm(start_to_goal)
        line_length = np.clip(line_length, a_min=1e-5, a_max=None)
        line_dir_normalized = start_to_goal / line_length
        
        start_to_robot = base_pos[:2] - goal_start_pos
        proj_length = np.sum(start_to_robot * line_dir_normalized)
        proj_point = goal_start_pos + proj_length * line_dir_normalized
        lateral_offset_vec = base_pos[:2] - proj_point
        
        perpendicular_dir = np.array([-line_dir_normalized[1], line_dir_normalized[0]])
        lateral_offset_moving = np.sum(lateral_offset_vec * perpendicular_dir)
        
        y_vel_gain = config.get('y_vel_gain', 0.8)
        y_vel_max = config.get('y_vel_max', 0.5)
        y_vel_cmd = y_vel_gain * lateral_offset_moving
        y_vel_cmd = np.clip(y_vel_cmd, -y_vel_max, y_vel_max)
    
    # 设置线速度命令
    if goal_started_moving:
        cmd[0] = original_lin_vel_cmd[0]
        cmd[1] = y_vel_cmd
    else:
        cmd[0] = 0.0
        cmd[1] = 0.0
    
    delta_pose_x = cur_goal[0] - base_pos[0]
    delta_pose_y = cur_goal[1] - base_pos[1]
    default_angles = config['default_angles']
    
    # 构建本体感受观测（48 维）
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

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal, goal_start_pos, goal_started_moving, original_lin_vel_cmd):
    """
    计算完整观测（785 维）
    
    组成：
    - proprio(48) + scan(225) + priv_explicit(3) + priv_latent(29) + history*proprio(10*48) = 785
    
    @param m MuJoCo 模型
    @param d MuJoCo 数据
    @param config 配置字典
    @param action 当前动作
    @param cmd 速度命令
    @param scan_points_body 扫描点
    @param obs_history_buf 历史观测缓冲区
    @param cur_goal 当前目标
    @param goal_start_pos 目标起点
    @param goal_started_moving 是否已开始移动
    @param original_lin_vel_cmd 原始线速度命令
    @return 完整观测和本体感受观测
    """
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd, cur_goal, goal_start_pos, goal_started_moving, original_lin_vel_cmd)
    
    num_scan = scan_points_body.shape[0]
    if config.get('measure_heights', True) and config.get('use_lidar', True) and config.get('lidar_mode') == 'terrain':
        measured_heights = get_scan_from_terrain(m, d, scan_points_body)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    # 特权信息（部署时设为 0）
    priv_explicit = np.zeros(3, dtype=np.float32)   # base_lin_vel
    priv_latent = np.zeros(29, dtype=np.float32)    # 质量、摩擦、电机强度
    
    # 拼接完整观测
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    return full_obs, proprio_obs

def update_visualization_markers(m, d, start_pos, cur_goal):
    """
    更新可视化标记（起点、目标、路径等）
    
    @param m MuJoCo 模型
    @param d MuJoCo 数据
    @param start_pos 起点位置
    @param cur_goal 当前目标位置
    """
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
    
    # 绘制路径（16 个点）
    for i in range(16):
        t = i / 15.0
        point_x = start_pos[0] + t * (cur_goal[0] - start_pos[0])
        point_y = start_pos[1] + t * (cur_goal[1] - start_pos[1])
        update_marker(f"viz_path_{i}", [point_x, point_y, z_height])

def main():
    """
    主函数：MuJoCo 部署主循环（使用 ONNX 模型）
    """
    config_path = os.path.join('deploy_mujoco/configs/g1_onnx.yaml')
    config = load_config(config_path)
    
    # 加载 MuJoCo 模型
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    n_joints = d.qpos.shape[0] - 7
    num_actions = config['num_actions']
    num_upper_joints = n_joints - num_actions if n_joints > num_actions else 0
    
    # 初始化扫描点（15×15 网格）
    scan_range = config.get('scan_range', 0.7)
    scan_resolution = config.get('scan_resolution', 15)
    scan_xy = np.linspace(-scan_range, scan_range, scan_resolution, dtype=np.float32)
    scan_points_body = np.zeros((scan_resolution * scan_resolution, 3), dtype=np.float32)
    scan_points_body[:, :2] = np.stack(np.meshgrid(scan_xy, scan_xy), axis=-1).reshape(-1, 2)
    
    print(f"MuJoCo部署（ONNX版本）| 控制频率: {1/config['simulation_dt']/config['control_decimation']:.0f}Hz | 目标: ({config['goal_init'][0]:.1f}, {config['goal_init'][1]:.1f})")
    
    # 初始化状态变量
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    original_lin_vel_cmd = config['cmd_init'][:2].copy()
    obs_history_buf = np.zeros((config['obs_history_len'], 48), dtype=np.float32)
    
    cur_goal = config['goal_init'].copy().astype(np.float32)
    goal_reach_threshold = config['goal_reach_threshold']
    goal_dynamic_update = config['goal_dynamic_update']
    goal_distance_range = config['goal_forward_distance_range']
    goal_start_pos = d.qpos[:2].copy().astype(np.float32)
    goal_started_moving = False
    need_recalc_timeout = True
    
    onnx_path = config['policy_path']
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    print(f"✓ ONNX 模型加载成功")
    print(f"  输入名称: {input_name}")
    print(f"  输入形状: {session.get_inputs()[0].shape}")
    print(f"  输出形状: {session.get_outputs()[0].shape}")
    print("运行中...\n")
    
    counter = 0
    control_counter = 0
    print_period = config.get('print_period', 100)
    
    # 主循环
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            
            # PD 控制下肢
            leg_tau = pd_control(target_dof_pos, d.qpos[7:7+num_actions], config['kps'],
                                np.zeros_like(config['kps']), d.qvel[6:6+num_actions], config['kds'])
            d.ctrl[:num_actions] = leg_tau
            
            # PD 控制上肢（固定到零位）
            if n_joints > num_actions:
                upper_tau = pd_control(np.zeros(num_upper_joints), d.qpos[7+num_actions:7+n_joints],
                                      np.full(num_upper_joints, config['upper_body_kps']), np.zeros(num_upper_joints),
                                      d.qvel[6+num_actions:6+n_joints], np.full(num_upper_joints, config['upper_body_kds']))
                d.ctrl[num_actions:num_actions+num_upper_joints] = upper_tau
            
            # 仿真步进
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                robot_pos = d.qpos[:2]
                distance_to_goal = np.linalg.norm(cur_goal[:2] - robot_pos)
                
                # 计算朝向误差
                quat = d.qpos[3:7]
                w, x, y, z = quat
                yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                target_vec = cur_goal[:2] - robot_pos
                target_yaw = np.arctan2(target_vec[1], target_vec[0])
                delta_yaw = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
                yaw_tolerance = config.get('yaw_tolerance_for_linear_vel', 0.15)
                heading_aligned = np.abs(delta_yaw) < yaw_tolerance
                
                # 检测首次对齐
                if need_recalc_timeout and heading_aligned:
                    goal_start_pos = robot_pos.copy()
                    goal_started_moving = True
                    need_recalc_timeout = False  
                
                # 动态更新目标
                if goal_dynamic_update and distance_to_goal < goal_reach_threshold:
                    forward_dist = np.random.uniform(goal_distance_range[0], goal_distance_range[1])
                    angle_range = config.get('goal_angle_range', [-np.pi/2, np.pi/2])
                    angle_offset = np.random.uniform(angle_range[0], angle_range[1])
                    target_yaw = yaw + angle_offset
                    
                    cur_goal[0] = robot_pos[0] + forward_dist * np.cos(target_yaw)
                    cur_goal[1] = robot_pos[1] + forward_dist * np.sin(target_yaw)
                    cur_goal[2] = config['goal_init'][2]
                    goal_started_moving = False
                    need_recalc_timeout = True
                    
                    print(f"[目标更新] ({cur_goal[0]:.2f}, {cur_goal[1]:.2f}) | 距离:{forward_dist:.2f}m, 角度:{np.degrees(angle_offset):+.0f}°")
                
                # 计算观测
                full_obs, proprio_obs = compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, cur_goal, goal_start_pos, goal_started_moving, original_lin_vel_cmd)
                obs_history_buf = np.roll(obs_history_buf, -1, axis=0)
                obs_history_buf[-1] = proprio_obs
                
                # ONNX 推理（关键区别：使用 ONNX Runtime 而不是 PyTorch）
                obs_input = full_obs.reshape(1, -1).astype(np.float32)  # 确保形状为 [1, 785]
                ort_inputs = {input_name: obs_input}
                ort_outputs = session.run(None, ort_inputs)
                action = ort_outputs[0].squeeze()  # 输出形状: [1, 12] -> [12]
                
                # 动作裁剪（与训练环境一致）
                clip_actions = config.get('clip_actions', 1.2) / config['action_scale']
                action[:num_actions] = np.clip(action[:num_actions], -clip_actions, clip_actions)
                
                target_dof_pos = action[:num_actions] * config['action_scale'] + config['default_angles']
                control_counter += 1
                
                # 实时打印
                print(f"[{control_counter:5d}] 速度: x={cmd[0]:5.2f} y={cmd[1]:5.2f} | 角速度: {cmd[2]:6.3f} | 状态: {'✓移动' if goal_started_moving else '✗转向'}")
                
                if control_counter % print_period == 0:
                    print(f"[{control_counter:5d}] 位置:({robot_pos[0]:5.2f},{robot_pos[1]:5.2f}) 目标:({cur_goal[0]:5.2f},{cur_goal[1]:5.2f}) 距离:{distance_to_goal:.2f}m")
            
            # 更新可视化
            update_visualization_markers(m, d, goal_start_pos, cur_goal)
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

