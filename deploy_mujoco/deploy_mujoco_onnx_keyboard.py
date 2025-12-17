import os
import time
import yaml
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import sys
import termios
import tty
import fcntl
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
    config.setdefault('keyboard_vel_max', 0.5)  # 最大线速度
    config.setdefault('keyboard_ang_vel_max', 0.5)  # 最大角速度
    config.setdefault('keyboard_vel_step', 0.1)  # 速度增量
    
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

def compute_proprioceptive_obs(d, config, action, cmd):
    """计算本体感受观测（45维，不包含delta_yaw, delta_pose_x, delta_pose_y）"""
    num_actions = config['num_actions']
    qj = d.qpos[7:7+num_actions]
    dqj = d.qvel[6:6+num_actions]
    omega = d.qvel[3:6]
    quat = d.qpos[3:7]
    
    default_angles = config['default_angles']
    
    # 观测维度：commands(3) + ang_vel(3) + gravity(3) + dof_pos(12) + dof_vel(12) + action_history(12) = 45
    proprio_obs = np.zeros(45, dtype=np.float32)
    idx = 0
    
    proprio_obs[idx:idx+3] = cmd[:3] * config['cmd_scale']
    idx += 3
    proprio_obs[idx:idx+3] = omega * config['ang_vel_scale']
    idx += 3
    proprio_obs[idx:idx+3] = get_gravity_orientation(quat)
    idx += 3
    proprio_obs[idx:idx+num_actions] = (qj - default_angles) * config['dof_pos_scale']
    idx += num_actions
    proprio_obs[idx:idx+num_actions] = dqj * config['dof_vel_scale']
    idx += num_actions
    proprio_obs[idx:idx+num_actions] = action[:num_actions]
    idx += num_actions
    
    return proprio_obs

def compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf):
    proprio_obs = compute_proprioceptive_obs(d, config, action, cmd)
    
    num_scan = scan_points_body.shape[0]
    if config.get('measure_heights', True) and config.get('use_lidar', True) and config.get('lidar_mode') == 'terrain':
        measured_heights = get_scan_from_terrain(m, d, scan_points_body)
        heights = (d.qpos[2] - measured_heights).astype(np.float32)
        # print('hieghts',heights)
    else:
        heights = np.zeros(num_scan, dtype=np.float32)
    
    priv_explicit = np.zeros(3, dtype=np.float32)
    priv_latent = np.zeros(29, dtype=np.float32)
    
    full_obs = np.concatenate([proprio_obs, heights, priv_explicit, priv_latent, obs_history_buf.flatten()]).astype(np.float32)
    return full_obs, proprio_obs

class KeyboardController:
    """键盘控制器：使用线程监听键盘输入"""
    def __init__(self, config):
        self.config = config
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [vx, vy, vyaw]
        self.running = True
        self.vel_max = config.get('keyboard_vel_max', 0.5)
        self.ang_vel_max = config.get('keyboard_ang_vel_max', 0.5)
        self.vel_step = config.get('keyboard_vel_step', 0.1)
        
    def get_key(self):
        """非阻塞获取键盘输入（Linux）"""
        try:
            # 非阻塞读取一个字符
            return sys.stdin.read(1)
        except (IOError, OSError, BlockingIOError):
            # 非阻塞读取时如果没有数据会抛出异常，这是正常的
            return None
        except Exception:
            return None
    
    def update(self):
        """更新速度命令"""
        key = self.get_key()
        if key is None:
            return
        
        key = key.lower()
        
        # i/k: 前进/后退 (vx)
        if key == 'i':
            self.cmd[0] = np.clip(self.cmd[0] + self.vel_step, 0.0, self.vel_max)
        elif key == 'k':
            self.cmd[0] = np.clip(self.cmd[0] - self.vel_step, -self.vel_max, 0.0)
        
        # j/l: 左/右 (vy)
        elif key == 'j':
            self.cmd[1] = np.clip(self.cmd[1] - self.vel_step, -self.vel_max, 0.0)
        elif key == 'l':
            self.cmd[1] = np.clip(self.cmd[1] + self.vel_step, 0.0, self.vel_max)
        
        # u/o: 左转/右转 (vyaw)
        elif key == 'u':
            self.cmd[2] = np.clip(self.cmd[2] - self.vel_step, -self.ang_vel_max, 0.0)
        elif key == 'o':
            self.cmd[2] = np.clip(self.cmd[2] + self.vel_step, 0.0, self.ang_vel_max)
        
        # 空格键：停止所有速度
        elif key == ' ':
            self.cmd[:] = 0.0
        
        # 打印当前速度
        if key in ['i', 'k', 'j', 'l', 'u', 'o', ' ']:
            print(f"[键盘控制] vx={self.cmd[0]:.2f} vy={self.cmd[1]:.2f} vyaw={self.cmd[2]:.2f}")
    
    def get_cmd(self):
        """获取当前速度命令"""
        return self.cmd.copy()

def main():
    config_path = os.path.join('deploy_mujoco/configs/g1_onnx_keyboard.yaml')
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
    
    print("=" * 60)
    print("MuJoCo部署 (ONNX + 键盘控制)")
    print(f"控制频率: {1/config['simulation_dt']/config['control_decimation']:.0f}Hz")
    print("=" * 60)
    print("键盘控制说明:")
    print("  I/K: 前进/后退 (vx)")
    print("  J/L: 左/右 (vy)")
    print("  U/O: 左转/右转 (vyaw)")
    print("  空格: 停止所有速度")
    print("=" * 60)
    
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    obs_history_buf = np.zeros((config['obs_history_len'], 45), dtype=np.float32)  # 45维观测
    
    # 键盘控制器
    keyboard = KeyboardController(config)
    
    # 加载 ONNX 模型
    print(f"\n加载 ONNX 策略: {os.path.basename(config['policy_path'])}")
    session = ort.InferenceSession(config['policy_path'], providers=['CPUExecutionProvider'])
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"ONNX 模型输入: {input_name}, 输出: {output_name}")
    print("运行中... (请确保终端窗口处于焦点状态以接收键盘输入)\n")
    
    counter = 0
    control_counter = 0
    print_period = config.get('print_period', 100)
    
    # 设置终端为非阻塞模式（Linux）
    old_termios = termios.tcgetattr(sys.stdin)
    old_flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
    
    try:
        # 设置终端为cbreak模式（字符立即可用，不需要回车）
        tty.setcbreak(sys.stdin.fileno())
        # 设置stdin为非阻塞模式
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
    except Exception as e:
        print(f"警告：无法设置终端为非阻塞模式: {e}")
        print("键盘控制可能无法正常工作")
    
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < config['simulation_duration']:
                step_start = time.time()
                
                # 更新键盘控制
                keyboard.update()
                cmd = keyboard.get_cmd()
                
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
                    
                    full_obs, proprio_obs = compute_full_observation(m, d, config, action, cmd, scan_points_body, obs_history_buf)
                    obs_history_buf = np.roll(obs_history_buf, -1, axis=0)
                    obs_history_buf[-1] = proprio_obs
                    
                    # 使用 ONNX Runtime 进行推理
                    obs_input = full_obs.reshape(1, -1).astype(np.float32)
                    action = session.run([output_name], {input_name: obs_input})[0].squeeze()
                    
                    # 动作裁剪：与训练环境一致
                    clip_actions = config.get('clip_actions', 1.2) / config['action_scale']
                    action[:num_actions] = np.clip(action[:num_actions], -clip_actions, clip_actions)
                    
                    target_dof_pos = action[:num_actions] * config['action_scale'] + config['default_angles']
                    control_counter += 1
                    
                    if control_counter % print_period == 0:
                        print(f"[{control_counter:5d}] 位置:({robot_pos[0]:5.2f},{robot_pos[1]:5.2f}) | 速度: vx={cmd[0]:5.2f} vy={cmd[1]:5.2f} vyaw={cmd[2]:6.3f}")
                
                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        # 恢复终端设置（Linux）
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)
            fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags)
        except Exception as e:
            print(f"警告：无法恢复终端设置: {e}")

if __name__ == "__main__":
    main()

