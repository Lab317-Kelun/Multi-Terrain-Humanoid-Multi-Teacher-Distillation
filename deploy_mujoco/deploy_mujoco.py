import os
import time
import yaml
from typing import Tuple, Dict, Any
import torch
import numpy as np
import mujoco
import mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR

# 常量定义
RAY_START_OFFSET = 2.0  # 射线起始点高度偏移（米）
RAY_LENGTH = 4.0  # 射线长度（米）
DEFAULT_TERRAIN_HEIGHT_OFFSET = 2.0  # 未找到地形时的默认高度偏移（米）

def load_config(config_path: str) -> Dict[str, Any]:
    """加载并处理YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # 处理路径
    for key in ['policy_path', 'xml_path']:
        if key not in config:
            continue
        path = config[key]
        if path.startswith('./'):
            config[key] = os.path.join(LEGGED_GYM_ROOT_DIR, path[2:])
        elif '{LEGGED_GYM_ROOT_DIR}' in path:
            config[key] = path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        if not os.path.exists(config[key]):
            raise FileNotFoundError(f"{key}文件不存在: {config[key]}")
    
    # 转换为numpy数组
    for key in ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    
    # 设置默认值
    defaults = {
        'use_lidar': True, 'lidar_mode': 'terrain', 'measure_heights': True,
        'obs_history_len': 10, 'scan_range': 0.7, 'scan_resolution': 15,
        'simulation_duration': 60.0, 'clip_actions': 1.2
    }
    for k, v in defaults.items():
        config.setdefault(k, v)
    
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

def quat_apply_yaw(quat, vec):
    """只围绕yaw轴旋转（与训练代码一致：只使用yaw分量）"""
    w, x, y, z = quat
    # 提取yaw分量：将x和y设为0，只保留z和w
    quat_yaw = np.array([w, 0.0, 0.0, z], dtype=np.float32)
    # 归一化
    quat_yaw = quat_yaw / np.linalg.norm(quat_yaw)
    # 应用旋转
    return quat_rotate(quat_yaw, vec)

def quat_rotate(q, v):
    """四元数旋转向量（与quat_apply等价）"""
    R = quat_to_rot(q)
    return (R @ v.T).T if v.ndim > 1 else R @ v

def get_gravity_orientation(quat):
    """获取重力向量在机体坐标系中的方向"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制器：根据位置和速度误差计算力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_scan_from_terrain(m: mujoco.MjModel, d: mujoco.MjData, 
                          scan_points_body: np.ndarray, 
                          pelvis_bodyid: int) -> np.ndarray:
    """从MuJoCo地形获取地形高度（与训练代码一致：只围绕yaw旋转）"""
    base_pos = d.qpos[:3]
    base_quat = d.qpos[3:7]
    
    # 与训练代码一致：先平移到测量点中心，只围绕yaw旋转，再平移回去
    measurement_center = np.mean(scan_points_body, axis=0)
    points_centered = scan_points_body - measurement_center
    points_rotated = quat_apply_yaw(base_quat, points_centered)
    scan_points_world = points_rotated + measurement_center + base_pos
    
    terrain_heights = np.zeros(len(scan_points_body), dtype=np.float32)
    ray_start = scan_points_world.copy()
    ray_start[:, 2] += RAY_START_OFFSET
    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    
    for i in range(len(scan_points_body)):
        distance = mujoco.mj_ray(m, d, ray_start[i], ray_dir, 
                                 geomgroup=None, flg_static=1,
                                 bodyexclude=pelvis_bodyid if pelvis_bodyid >= 0 else -1, 
                                 geomid=geomid)
        terrain_heights[i] = (ray_start[i] + ray_dir * distance)[2] if (geomid[0] >= 0 and distance < RAY_LENGTH) else base_pos[2] - DEFAULT_TERRAIN_HEIGHT_OFFSET
    
    return terrain_heights

def compute_proprioceptive_obs(d: mujoco.MjData, config: Dict[str, Any], 
                                action: np.ndarray, cmd: np.ndarray) -> np.ndarray:
    """计算本体感受观测（45维）"""
    n_joints = d.qpos.shape[0] - 7
    qj = d.qpos[7:7+n_joints]
    dqj = d.qvel[6:6+n_joints]
    
    default_angles = config['default_angles']
    padded_defaults = np.pad(default_angles, (0, max(0, n_joints - len(default_angles))), 
                            constant_values=0.0)[:n_joints]
    
    proprio_obs = np.zeros(45, dtype=np.float32)
    proprio_obs[0:3] = cmd[:3] * config['cmd_scale']
    proprio_obs[3:6] = d.qvel[3:6] * config.get('ang_vel_scale', 1.0)
    proprio_obs[6:9] = get_gravity_orientation(d.qpos[3:7])
    proprio_obs[9:21] = (qj[:12] - padded_defaults[:12]) * config.get('dof_pos_scale', 1.0)
    proprio_obs[21:33] = dqj[:12] * config.get('dof_vel_scale', 1.0)
    proprio_obs[33:45] = action[:12]
    
    return proprio_obs

def compute_full_observation(m: mujoco.MjModel, d: mujoco.MjData, 
                            config: Dict[str, Any], action: np.ndarray, 
                            cmd: np.ndarray, scan_points_body: np.ndarray,
                            obs_history_buf: np.ndarray, pelvis_bodyid: int,
                            debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """计算完整的RMA格式观测向量"""
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd)
    
    num_scan = scan_points_body.shape[0]
    use_terrain = (config.get('measure_heights', True) and 
                   config.get('use_lidar', True) and 
                   config.get('lidar_mode') == 'terrain')
    if use_terrain:
        measured_heights = get_scan_from_terrain(m, d, scan_points_body, pelvis_bodyid)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
        heights = np.clip(heights, -1.0, 1.0)  # 与训练代码一致：clip到[-1, 1]
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    full_obs = np.concatenate([
        proprio_obs, heights, 
        np.zeros(3, dtype=np.float32),  # priv_explicit
        np.zeros(29, dtype=np.float32),  # priv_latent
        obs_history_buf.flatten()
    ]).astype(np.float32)
    
    return full_obs, proprio_obs

def setup_robot_initial_pose(m: mujoco.MjModel, d: mujoco.MjData, xml_path: str) -> None:
    """设置机器人初始姿态"""
    if 'gap.xml' in xml_path:
        d.qpos[:3] = [1.5, 0.0, 1.093]  # x, y, z
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quaternion
        mujoco.mj_forward(m, d)


def _normalize_array_param(value: Any, n: int, default: float) -> np.ndarray:
    """将参数标准化为长度为n的数组"""
    if isinstance(value, (int, float)):
        return np.full(n, value, dtype=np.float32)
    arr = np.array(value, dtype=np.float32)
    return np.pad(arr, (0, max(0, n - len(arr))), constant_values=default)[:n]

def initialize_upper_body_pd_params(config: Dict[str, Any], n_upper: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """初始化上肢PD控制参数"""
    return (
        _normalize_array_param(config.get('upper_body_kps', 300.0), n_upper, 300.0),
        _normalize_array_param(config.get('upper_body_kds', 5.0), n_upper, 5.0),
        _normalize_array_param(config.get('upper_body_default_angles', 0.0), n_upper, 0.0)
    )


def main():
    """主函数：运行MuJoCo仿真并控制机器人"""
    config_path = os.path.join('/home/cft/kelun/Humanoid-Terrain-Bench/deploy_mujoco/configs/g1.yaml')
    config = load_config(config_path)
    
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    setup_robot_initial_pose(m, d, config['xml_path'])
    
    n_joints = d.qpos.shape[0] - 7
    num_actions = config['num_actions']
    n_upper = n_joints - num_actions
    
    # 初始化扫描点
    scan_resolution = config['scan_resolution']
    scan_xy = np.linspace(-config['scan_range'], config['scan_range'], scan_resolution, dtype=np.float32)
    scan_points_body = np.zeros((scan_resolution * scan_resolution, 3), dtype=np.float32)
    scan_points_body[:, :2] = np.stack(np.meshgrid(scan_xy, scan_xy), axis=-1).reshape(-1, 2)
    
    # 初始化变量
    action = np.zeros(27, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    obs_history_buf = np.zeros((config['obs_history_len'], 45), dtype=np.float32)
    
    upper_body_kps, upper_body_kds, upper_body_target = initialize_upper_body_pd_params(config, n_upper)
    pelvis_bodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    
    policy = torch.jit.load(config['policy_path'])
    policy.eval()
    print("策略模型加载成功!\n")
    
    counter = 0
    control_decimation = config.get('control_decimation', 1)
    clip_actions = config['clip_actions'] / config['action_scale']
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < config['simulation_duration']:
            step_start = time.time()
            
            # PD控制
            d.ctrl[:num_actions] = pd_control(
                target_dof_pos, d.qpos[7:7+num_actions], config['kps'],
                np.zeros_like(config['kps']), d.qvel[6:6+num_actions], config['kds']
            )
            if n_joints > num_actions and d.ctrl.shape[0] > num_actions:
                d.ctrl[num_actions:] = pd_control(
                    upper_body_target, d.qpos[7+num_actions:7+n_joints], upper_body_kps,
                    np.zeros(n_upper, dtype=np.float32), d.qvel[6+num_actions:6+n_joints], upper_body_kds
                )
            
            mujoco.mj_step(m, d)
            
            if (counter := counter + 1) % control_decimation == 0:
                full_obs, proprio_obs = compute_full_observation(
                    m, d, config, action, cmd, scan_points_body, obs_history_buf, pelvis_bodyid
                )
                obs_history_buf = np.roll(obs_history_buf, -1, axis=0)
                obs_history_buf[-1] = proprio_obs
                
                with torch.no_grad():
                    action = policy(torch.from_numpy(full_obs).unsqueeze(0)).detach().numpy().squeeze()
                
                action = np.clip(action, -clip_actions, clip_actions)
                if action.shape[0] > num_actions:
                    action[num_actions:] = 0.0
                target_dof_pos = action[:num_actions] * config['action_scale'] + config['default_angles']
            
            viewer.sync()
            sleep_time = m.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    main()
