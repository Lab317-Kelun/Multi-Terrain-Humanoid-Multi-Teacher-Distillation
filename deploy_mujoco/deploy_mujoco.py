import os
import time
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR

def load_config(config_path):
    """加载并处理YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理路径
    for key in ['policy_path', 'xml_path']:
        if key in config:
            path = config[key]
            if path.startswith('./'):
                config[key] = os.path.join(LEGGED_GYM_ROOT_DIR, path[2:])
            elif '{LEGGED_GYM_ROOT_DIR}' in path:
                config[key] = path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    
    # 转换为numpy数组
    for key in ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    
    # 设置默认值
    config.setdefault('use_lidar', True)
    config.setdefault('lidar_mode', 'terrain')
    config.setdefault('measure_heights', True)
    config.setdefault('obs_history_len', 10)
    
    return config

def quat_to_rot(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)

def quat_rotate_inverse(q, v):
    """将世界坐标系向量转换到机体坐标系"""
    R = quat_to_rot(q)
    return (R.T @ v).astype(np.float32)

def get_gravity_orientation(quat):
    """获取重力向量在机体坐标系中的方向"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制器：根据位置和速度误差计算力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_scan_from_terrain(m, d, scan_points_body, pelvis_bodyid):
    """从MuJoCo地形获取地形高度（世界坐标系z坐标）
    射线会穿过机器人身体，只检测地形
    """
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
    
    # 使用 bodyexclude 排除机器人身体，只检测静态地形
    for i in range(num_points):
        distance = mujoco.mj_ray(m, d, ray_start[i], ray_dir, 
                                 geomgroup=None, flg_static=1,  # 只检测静态几何体（地形）
                                 bodyexclude=pelvis_bodyid if pelvis_bodyid >= 0 else -1, 
                                 geomid=geomid)
        
        if geomid[0] >= 0 and distance < ray_length:
            terrain_heights[i] = (ray_start[i] + ray_dir * distance)[2]
        else:
            # 如果没有找到地形，使用默认值
            terrain_heights[i] = base_pos[2] - 2.0
    
    return terrain_heights

def compute_proprioceptive_obs(d, config, action, cmd):
    """计算当前本体感受观测（75维）"""
    n_joints = d.qpos.shape[0] - 7
    
    # 获取状态
    qj = d.qpos[7:7+n_joints]
    dqj = d.qvel[6:6+n_joints]
    omega = d.qvel[3:6]
    quat = d.qpos[3:7]
    
    # 处理默认角度
    default_angles = config['default_angles']
    if len(default_angles) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(default_angles)] = default_angles
    else:
        padded_defaults = default_angles[:n_joints]
    
    # 修改！只包含下半身关节观测（总共45维）
    # 构成：3(cmd) + 3(ang_vel) + 3(gravity) + 12(dof_pos下半身) + 12(dof_vel下半身) + 12(action) = 45
    proprio_obs = np.zeros(45, dtype=np.float32)
    proprio_obs[0:3] = cmd[:3] * config['cmd_scale']  # 3维 - 命令
    proprio_obs[3:6] = omega * config['ang_vel_scale']  # 3维 - 角速度
    proprio_obs[6:9] = get_gravity_orientation(quat)  # 3维 - 重力方向
    # 只使用前12个关节（下半身）
    proprio_obs[9:21] = (qj[:12] - padded_defaults[:12]) * config['dof_pos_scale']  # 12维 - 下半身关节位置
    proprio_obs[21:33] = dqj[:12] * config['dof_vel_scale']  # 12维 - 下半身关节速度
    proprio_obs[33:45] = action[:12]  # 12维 - 动作历史
    
    return proprio_obs

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, pelvis_bodyid, debug=False):
    """计算完整的RMA格式观测向量
    修改！维度变化：
    - 原来：75 + 225 + 3 + 29 + 75*10 = 1082
    - 现在：45 + 225 + 3 + 29 + 45*10 = 752
    """
    # 本体感受观测（45维 - 只包含下半身）
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd)
    
    # 高度观测（225维）：heights = base_height - measured_heights
    num_scan = scan_points_body.shape[0]
    if config.get('measure_heights', True) and config.get('use_lidar', True) and config.get('lidar_mode') == 'terrain':
        # 测试模式：将 measured_heights 全部设为0，这样 heights = base_height - 0 = base_height
        measured_heights = np.zeros(num_scan, dtype=np.float32)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
        print('heights: ', heights)
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    # 特权信息（推理时设为0）
    priv_explicit = np.zeros(3, dtype=np.float32)
    priv_latent = np.zeros(29, dtype=np.float32)
    
    # 拼接：proprio(75) + heights(225) + priv_explicit(3) + priv_latent(29) + history(750)
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    
    # 调试信息
    if debug:
        print("\n" + "="*60)
        print("观测维度信息:")
        print(f"  本体感受观测 (proprio_obs): {proprio_obs.shape[0]} 维")
        print(f"  高度观测 (heights): {heights.shape[0]} 维")
        print(f"  特权显式 (priv_explicit): {priv_explicit.shape[0]} 维")
        print(f"  特权隐式 (priv_latent): {priv_latent.shape[0]} 维")
        print(f"  历史信息 (history): {obs_history_buf.flatten().shape[0]} 维")
        print(f"  总观测维度 (full_obs): {full_obs.shape[0]} 维")
        print("\n观测统计信息:")
        print(f"  proprio_obs: min={proprio_obs.min():.4f}, max={proprio_obs.max():.4f}, mean={proprio_obs.mean():.4f}, std={proprio_obs.std():.4f}")
        print(f"  heights: min={heights.min():.4f}, max={heights.max():.4f}, mean={heights.mean():.4f}, std={heights.std():.4f}")
        print(f"  priv_explicit: {priv_explicit}")
        print(f"  priv_latent: {priv_latent}")
        print(f"  history: min={obs_history_buf.min():.4f}, max={obs_history_buf.max():.4f}, mean={obs_history_buf.mean():.4f}")
        print(f"  full_obs: min={full_obs.min():.4f}, max={full_obs.max():.4f}, mean={full_obs.mean():.4f}, std={full_obs.std():.4f}")
        print("="*60 + "\n")
    
    return full_obs, proprio_obs

def main():
    """主函数：运行MuJoCo仿真并控制机器人"""
    # 加载配置
    config_path = os.path.join('/home/cft/kelun/Humanoid-Terrain-Bench/deploy_mujoco/configs/g1.yaml')
    config = load_config(config_path)
    
    # 加载模型
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    # 如果是 gap.xml，设置机器人初始位置在第一个平台中心顶部
    if 'gap.xml' in config['xml_path']:
        # 平台中心: x=2.5, y=0, 平台顶部: z=0.3, 机器人原始高度: 0.793
        # 所以 pelvis z = 0.3 + 0.793 = 1.093
        d.qpos[0] = 1.5  # x 位置：平台中心
        d.qpos[1] = 0.0  # y 位置：平台中心
        d.qpos[2] = 1.093  # z 位置：平台顶部 + 机器人高度
        # 姿态保持默认 (qw=1, qx=0, qy=0, qz=0)
        d.qpos[3] = 1.0  # qw
        d.qpos[4] = 0.0  # qx
        d.qpos[5] = 0.0  # qy
        d.qpos[6] = 0.0  # qz
        # 前向一步以应用初始位置
        mujoco.mj_forward(m, d)
    
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
    print(f"  扫描分辨率: {scan_resolution}×{scan_resolution} = {num_scan} 个点")
    
    # 初始化变量
    # 注意：action应该是27维（与训练时一致），即使只使用前12维
    action = np.zeros(27, dtype=np.float32)  # 修复：使用27维而不是num_actions
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    obs_history_len = config['obs_history_len']
    # 修改！history buffer现在是45维（只包含下半身观测）
    obs_history_buf = np.zeros((obs_history_len, 45), dtype=np.float32)
    
    # 预处理上肢PD参数（避免每次循环都读取）
    n_upper = n_joints - num_actions  # 15个上肢关节
    
    # 处理upper_body_kps：可以是单个数字或数组
    kps_config = config.get('upper_body_kps', 300.0)
    if isinstance(kps_config, (int, float)):
        upper_body_kps = np.full(n_upper, kps_config, dtype=np.float32)
    else:
        upper_body_kps = np.array(kps_config, dtype=np.float32)
        if len(upper_body_kps) < n_upper:
            upper_body_kps = np.pad(upper_body_kps, (0, n_upper - len(upper_body_kps)), constant_values=300.0)
        elif len(upper_body_kps) > n_upper:
            upper_body_kps = upper_body_kps[:n_upper]
    
    # 处理upper_body_kds：可以是单个数字或数组
    kds_config = config.get('upper_body_kds', 5.0)
    if isinstance(kds_config, (int, float)):
        upper_body_kds = np.full(n_upper, kds_config, dtype=np.float32)
    else:
        upper_body_kds = np.array(kds_config, dtype=np.float32)
        if len(upper_body_kds) < n_upper:
            upper_body_kds = np.pad(upper_body_kds, (0, n_upper - len(upper_body_kds)), constant_values=5.0)
        elif len(upper_body_kds) > n_upper:
            upper_body_kds = upper_body_kds[:n_upper]
    
    # 处理upper_body_target：可以是单个数字或数组
    target_config = config.get('upper_body_default_angles', 0.0)
    if isinstance(target_config, (int, float)):
        upper_body_target = np.full(n_upper, target_config, dtype=np.float32)
    else:
        upper_body_target = np.array(target_config, dtype=np.float32)
        if len(upper_body_target) < n_upper:
            upper_body_target = np.pad(upper_body_target, (0, n_upper - len(upper_body_target)), constant_values=0.0)
        elif len(upper_body_target) > n_upper:
            upper_body_target = upper_body_target[:n_upper]
    
    # 计算总观测维度（修改！）
    # 45(proprio) + 225(scan) + 3(priv_explicit) + 29(priv_latent) + 45*10(history) = 752
    total_obs_dim = 45 + num_scan + 3 + 29 + obs_history_len * 45
    print(f"\n观测维度配置:")
    print(f"  本体感受观测: 75 维")
    print(f"  高度扫描: {num_scan} 维")
    print(f"  特权显式: 3 维")
    print(f"  特权隐式: 29 维")
    print(f"  历史信息: {obs_history_len} × 75 = {obs_history_len * 75} 维")
    print(f"  总观测维度: {total_obs_dim} 维")
    print(f"{'='*60}\n")
    
    # 找到机器人根 body (pelvis) 的 ID，用于排除机器人身体（只计算一次）
    pelvis_bodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_bodyid < 0:
        print("警告: 未找到 pelvis body，射线检测可能包含机器人身体")
    
    # 加载策略模型
    print(f"加载策略模型: {config['policy_path']}")
    policy = torch.jit.load(config['policy_path'])
    policy.eval()
    print("策略模型加载成功!\n")
    
    counter = 0
    debug_counter = 0  # 调试信息计数器
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
            
            # 控制上肢关节（保持在默认位置，与legged_gym的Position模式一致）
            if n_joints > num_actions:
                # 使用预处理好的上肢PD参数
                arm_tau = pd_control(
                    upper_body_target,
                    d.qpos[7+num_actions:7+n_joints],
                    upper_body_kps,
                    np.zeros(n_upper),
                    d.qvel[6+num_actions:6+n_joints],
                    upper_body_kds
                )
                if d.ctrl.shape[0] > num_actions:
                    d.ctrl[num_actions:] = arm_tau
            
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                # 计算观测并更新历史
                debug_flag = (debug_counter < 3)  # 前3次控制周期打印调试信息
                full_obs, proprio_obs = compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf, pelvis_bodyid, debug=debug_flag)
                obs_history_buf = np.roll(obs_history_buf, -1, axis=0)
                obs_history_buf[-1] = proprio_obs
                
                # 策略推理
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(full_obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()
                
                # !!关键修复!! 对action进行clipping（与Isaac Gym训练一致）
                # clip_actions = normalization.clip_actions / control.action_scale = 1.2 / 0.25 = 4.8
                clip_actions = config.get('clip_actions', 1.2) / config['action_scale']
                action = np.clip(action, -clip_actions, clip_actions)

                if action.shape[0] > num_actions:
                    action[num_actions:] = 0.0  # 后15维（上肢）强制为0
                
                target_dof_pos = action[:num_actions] * config['action_scale'] + config['default_angles']
                
                # 调试信息：动作输出
                if debug_flag:
                    print(f"\n控制周期 #{debug_counter + 1}:")
                    print(f"  动作输出维度: {action.shape[0]}")
                    print(f"  动作统计: min={action.min():.4f}, max={action.max():.4f}, mean={action.mean():.4f}")
                    print(f"  目标关节位置: {target_dof_pos[:6]} ... (前6个关节)")
                    print(f"  速度指令: {cmd}")
                    print(f"  基座位置: [{d.qpos[0]:.3f}, {d.qpos[1]:.3f}, {d.qpos[2]:.3f}]")
                    print(f"  基座高度: {d.qpos[2]:.3f} m\n")
                
                debug_counter += 1
                
                # 每100个控制周期打印一次简要信息
                if debug_counter > 0 and debug_counter % 100 == 0:
                    print(f"[控制周期 {debug_counter}] 基座高度: {d.qpos[2]:.3f} m, 速度指令: {cmd}, 动作均值: {action[:num_actions].mean():.4f}")
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
