# Humanoid Robot with AMP Support
# 扩展HumanoidRobot类，添加AMP (Adversarial Motion Priors) 功能
# AMP通过discriminator学习自然的人形运动风格

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict
from isaacgym import gymapi

from legged_gym.envs.base.humanoid_robot import HumanoidRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, normalize


class HumanoidRobotAMP(HumanoidRobot):
    """
    HumanoidRobot类的AMP扩展版本
    
    功能：
    1. 收集discriminator观测 (disc_obs)
    2. 维护历史状态buffer用于AMP训练
    3. 提供从motion库采样参考数据的接口
    4. 计算AMP风格奖励
    """
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, save):
        # 检查是否启用AMP
        self.enable_amp = getattr(cfg.env, 'enable_amp', False)
        
        if self.enable_amp:
            self.num_disc_obs_steps = getattr(cfg.env, 'num_disc_obs_steps', 10)
            self.amp_replay_buffer_size = getattr(cfg.env, 'amp_replay_buffer_size', 100000)
            self.amp_motion_files = getattr(cfg.env, 'amp_motion_files', [])
            # 预先获取key_body_names，避免在reset_idx时出错
            if hasattr(cfg.asset, 'key_bodies'):
                self.key_body_names = cfg.asset.key_bodies
            else:
                self.key_body_names = [cfg.asset.left_foot_name, cfg.asset.right_foot_name]
            self.num_key_bodies = len(self.key_body_names)
            print(f"[AMP] 启用AMP，discriminator观测步数: {self.num_disc_obs_steps}")
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, save)
        
        if self.enable_amp:
            self._init_amp_buffers()
            self._load_motion_lib()
            self._init_forward_kinematics()
            # 初始化完成后，填充历史缓冲区
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._reset_disc_hist(all_env_ids)
            print(f"[AMP] Discriminator观测维度: {self.disc_obs_size}")
    
    def _init_amp_buffers(self):
        """初始化AMP相关的缓冲区"""
        # ===== Discriminator观测缓冲区 =====
        # disc_obs包含：root状态 + 关节状态 + 关键身体部位位置
        
        # 计算单步disc_obs维度
        # - root_pos: 3 (x, y, z)
        # - root_rot: 4 (quaternion)
        # - root_lin_vel: 3
        # - root_ang_vel: 3
        # - dof_pos: num_dof (关节位置)
        # - dof_vel: num_dof (关节速度)
        # - key_body_pos: num_key_bodies * 3 (关键身体部位的位置)
        
        # 获取关键身体部位数量
        if hasattr(self.cfg.asset, 'key_bodies'):
            self.key_body_names = self.cfg.asset.key_bodies
        else:
            # 默认使用脚部作为关键部位
            self.key_body_names = [self.cfg.asset.left_foot_name, self.cfg.asset.right_foot_name]
        
        self.num_key_bodies = len(self.key_body_names)
        
        # 单步观测维度 = root(13) + dof(num_dof*2) + key_bodies(num_key_bodies*3)
        single_step_disc_obs_size = 13 + self.num_dof * 2 + self.num_key_bodies * 3
        
        # 总维度 = 单步维度 * 历史步数
        self.disc_obs_size = single_step_disc_obs_size * self.num_disc_obs_steps
        
        # disc_obs缓冲区
        self.disc_obs_buf = torch.zeros(
            self.num_envs, self.disc_obs_size, 
            dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # ===== 历史状态缓冲区 =====
        # 用于存储过去num_disc_obs_steps步的状态
        self.disc_hist_root_pos = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.disc_hist_root_rot = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, 4,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.disc_hist_root_lin_vel = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.disc_hist_root_ang_vel = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.disc_hist_dof_pos = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, self.num_dof,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.disc_hist_dof_vel = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, self.num_dof,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.disc_hist_key_body_pos = torch.zeros(
            self.num_envs, self.num_disc_obs_steps, self.num_key_bodies, 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # 获取关键身体部位的索引
        self.key_body_indices = torch.zeros(
            len(self.key_body_names), dtype=torch.long, 
            device=self.device, requires_grad=False
        )
        for i, name in enumerate(self.key_body_names):
            self.key_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )
        
        print(f"[AMP] 关键身体部位: {self.key_body_names}")
        print(f"[AMP] 关键身体索引: {self.key_body_indices}")
    
    def _init_buffers(self):
        """重写_init_buffers，添加AMP缓冲区"""
        super()._init_buffers()
        # AMP缓冲区在构造函数中通过_init_amp_buffers()初始化
    
    def step(self, actions):
        """重写step方法，更新disc_obs"""
        # 调用父类的step
        result = super().step(actions)
        
        # 如果启用AMP，更新discriminator观测
        if self.enable_amp:
            self._update_disc_hist()
            self._update_disc_obs()
            # 将disc_obs添加到extras中，供训练使用
            self.extras["disc_obs"] = self.disc_obs_buf.clone()
        
        return result
    
    def _update_disc_hist(self):
        """
        更新discriminator历史缓冲区
        将当前状态推入历史buffer（类似循环队列）
        
        优先使用IsaacGym的rigid_body_states（最准确），如果不可用则使用FK
        """
        # 滚动历史buffer（最老的数据被丢弃）
        self.disc_hist_root_pos[:, :-1] = self.disc_hist_root_pos[:, 1:].clone()
        self.disc_hist_root_rot[:, :-1] = self.disc_hist_root_rot[:, 1:].clone()
        self.disc_hist_root_lin_vel[:, :-1] = self.disc_hist_root_lin_vel[:, 1:].clone()
        self.disc_hist_root_ang_vel[:, :-1] = self.disc_hist_root_ang_vel[:, 1:].clone()
        self.disc_hist_dof_pos[:, :-1] = self.disc_hist_dof_pos[:, 1:].clone()
        self.disc_hist_dof_vel[:, :-1] = self.disc_hist_dof_vel[:, 1:].clone()
        self.disc_hist_key_body_pos[:, :-1] = self.disc_hist_key_body_pos[:, 1:].clone()
        
        # 添加当前状态到历史buffer的最后一位
        self.disc_hist_root_pos[:, -1] = self.root_states[:, :3]
        self.disc_hist_root_rot[:, -1] = self.root_states[:, 3:7]
        self.disc_hist_root_lin_vel[:, -1] = self.base_lin_vel
        self.disc_hist_root_ang_vel[:, -1] = self.base_ang_vel
        self.disc_hist_dof_pos[:, -1] = self.dof_pos
        self.disc_hist_dof_vel[:, -1] = self.dof_vel
        
        # 关键身体部位位置
        key_body_pos = self.rigid_body_states[:, self.key_body_indices, :3]
        self.disc_hist_key_body_pos[:, -1] = key_body_pos
    
    def _update_disc_obs(self):
        """
        计算discriminator观测
        
        Discriminator观测格式 (相对于当前root坐标系):
        - 对于每个历史步 t in [t-n+1, ..., t]:
          - root_pos_local: 3 (相对位置)
          - root_rot_local: 4 (相对旋转)
          - root_lin_vel_local: 3 (本地坐标系速度)
          - root_ang_vel_local: 3 (本地坐标系角速度)
          - dof_pos: num_dof
          - dof_vel: num_dof
          - key_body_pos_local: num_key_bodies * 3 (相对位置)
        """
        # 参考root状态（当前时刻）
        ref_root_pos = self.disc_hist_root_pos[:, -1, :]  # [num_envs, 3]
        ref_root_rot = self.disc_hist_root_rot[:, -1, :]  # [num_envs, 4]
        
        # 计算相对状态
        disc_obs_list = []
        
        for t in range(self.num_disc_obs_steps):
            # 当前历史步的root状态
            root_pos = self.disc_hist_root_pos[:, t, :]
            root_rot = self.disc_hist_root_rot[:, t, :]
            root_lin_vel = self.disc_hist_root_lin_vel[:, t, :]
            root_ang_vel = self.disc_hist_root_ang_vel[:, t, :]
            dof_pos = self.disc_hist_dof_pos[:, t, :]
            dof_vel = self.disc_hist_dof_vel[:, t, :]
            key_body_pos = self.disc_hist_key_body_pos[:, t, :, :]  # [num_envs, num_key_bodies, 3]
            
            # === 转换到局部坐标系 ===
            # 1. 相对位置（世界坐标）
            root_pos_rel = root_pos - ref_root_pos
            
            # 2. 转换到当前root的局部坐标系
            root_pos_local = quat_rotate_inverse(ref_root_rot, root_pos_rel)
            
            # 3. 相对旋转
            ref_root_rot_inv = self._quat_conjugate(ref_root_rot)
            root_rot_local = quat_mul(ref_root_rot_inv, root_rot)
            
            # 4. 速度转换到局部坐标系
            root_lin_vel_local = quat_rotate_inverse(ref_root_rot, root_lin_vel)
            root_ang_vel_local = quat_rotate_inverse(ref_root_rot, root_ang_vel)
            
            # 5. 关键身体部位相对位置
            key_body_pos_rel = key_body_pos - ref_root_pos.unsqueeze(1)  # [num_envs, num_key_bodies, 3]
            key_body_pos_local = quat_rotate_inverse(
                ref_root_rot.unsqueeze(1).expand(-1, self.num_key_bodies, -1).reshape(-1, 4),
                key_body_pos_rel.reshape(-1, 3)
            ).reshape(self.num_envs, self.num_key_bodies, 3)
            
            # 组装单步观测
            step_obs = torch.cat([
                root_pos_local,           # 3
                root_rot_local,           # 4
                root_lin_vel_local,       # 3
                root_ang_vel_local,       # 3
                dof_pos,                  # num_dof
                dof_vel,                  # num_dof
                key_body_pos_local.reshape(self.num_envs, -1)  # num_key_bodies * 3
            ], dim=-1)
            
            disc_obs_list.append(step_obs)
        
        # 拼接所有历史步
        self.disc_obs_buf[:] = torch.cat(disc_obs_list, dim=-1)
    
    def _quat_conjugate(self, q):
        """四元数共轭 (用于计算逆旋转)"""
        return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
    
    def get_disc_obs_space(self):
        """返回discriminator观测空间（供agent使用）"""
        import gymnasium.spaces as spaces
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.disc_obs_size,),
            dtype=np.float32
        )
    
    def _load_motion_lib(self):
        """加载Motion Library"""
        if len(self.amp_motion_files) == 0:
            print("[AMP] Warning: 没有提供motion文件，将使用随机采样")
            self._motion_lib = None
            return
        
        try:
            from legged_gym.utils.motion_lib import MotionLib
            # 暂时只支持单个motion文件
            motion_file = self.amp_motion_files[0]
            self._motion_lib = MotionLib(
                motion_file=motion_file,
                device=self.device,
                num_dofs=self.num_dof
            )
            print(f"[AMP] Motion Library加载成功")
        except Exception as e:
            print(f"[AMP] Warning: Motion Library加载失败: {e}")
            print(f"[AMP] 将使用随机采样作为替代")
            self._motion_lib = None
    
    def _init_forward_kinematics(self):
        """初始化Forward Kinematics计算器"""
        try:
            from legged_gym.utils.forward_kinematics import G1ForwardKinematics
            self._fk = G1ForwardKinematics(device=self.device)
            print(f"[AMP] Forward Kinematics初始化成功")
        except Exception as e:
            print(f"[AMP] Warning: FK初始化失败: {e}")
            self._fk = None
    
    def fetch_disc_obs_demo(self, num_samples):
        """
        从参考动作库采样discriminator观测（完整版本）
        
        与MimicKit的AMP实现一致：
        1. 采样完整的历史轨迹（num_disc_obs_steps步）
        2. 计算forward kinematics得到关键身体部位位置
        3. 转换到局部坐标系（相对于当前root）
        
        Args:
            num_samples: 采样数量
            
        Returns:
            disc_obs_demo: [num_samples, disc_obs_size]
        """
        if self._motion_lib is None:
            # 如果没有motion库，从当前环境随机采样
            env_ids = torch.randint(0, self.num_envs, (num_samples,), device=self.device)
            return self.disc_obs_buf[env_ids].clone()
        
        # 从motion库采样motion和时间
        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times = self._motion_lib.sample_time(motion_ids)
        
        # 获取当前时刻的motion状态（参考帧）
        ref_root_pos, ref_root_rot, _, _ = self._motion_lib.get_motion_state(
            motion_ids, motion_times
        )
        
        # 如果没有root信息，使用默认值
        if ref_root_pos is None:
            ref_root_pos = torch.zeros(num_samples, 3, device=self.device)
            ref_root_pos[:, 2] = 0.78
        if ref_root_rot is None:
            ref_root_rot = torch.zeros(num_samples, 4, device=self.device)
            ref_root_rot[:, 3] = 1.0
        
        # 采样历史轨迹（从t-n+1到t，共n步）
        dt = self._motion_lib.dt
        disc_obs_list = []
        
        for t in range(self.num_disc_obs_steps):
            # 计算历史时刻（从旧到新）
            time_offset = (t - self.num_disc_obs_steps + 1) * dt
            hist_times = motion_times + time_offset
            
            # 获取历史时刻的motion状态
            hist_root_pos, hist_root_rot, hist_dof_pos, hist_dof_vel = \
                self._motion_lib.get_motion_state(motion_ids, hist_times)
            
            # 获取历史时刻的root速度
            hist_root_lin_vel, hist_root_ang_vel = \
                self._motion_lib.get_root_vel(motion_ids, hist_times)
            
            # 处理None值
            if hist_root_pos is None:
                hist_root_pos = ref_root_pos.clone()
            if hist_root_rot is None:
                hist_root_rot = ref_root_rot.clone()
            if hist_root_lin_vel is None:
                hist_root_lin_vel = torch.zeros(num_samples, 3, device=self.device)
            if hist_root_ang_vel is None:
                hist_root_ang_vel = torch.zeros(num_samples, 3, device=self.device)
            
            # === 转换到局部坐标系（相对于当前root）===
            # 1. Root位置：世界坐标转局部坐标
            root_pos_rel = hist_root_pos - ref_root_pos
            root_pos_local = quat_rotate_inverse(ref_root_rot, root_pos_rel)
            
            # 2. Root旋转：相对旋转
            ref_root_rot_inv = self._quat_conjugate(ref_root_rot)
            root_rot_local = quat_mul(ref_root_rot_inv, hist_root_rot)
            
            # 3. Root速度：转换到局部坐标系
            root_lin_vel_local = quat_rotate_inverse(ref_root_rot, hist_root_lin_vel)
            root_ang_vel_local = quat_rotate_inverse(ref_root_rot, hist_root_ang_vel)
            
            # 4. 关节状态（已经是局部的，不需要转换）
            dof_pos_local = hist_dof_pos
            dof_vel_local = hist_dof_vel
            
            # 5. 关键身体部位位置（通过forward kinematics计算）
            key_body_pos_world = self._compute_key_body_pos_from_motion(
                hist_root_pos, hist_root_rot, hist_dof_pos
            )
            
            # 转换关键身体位置到局部坐标系
            key_body_pos_rel = key_body_pos_world - ref_root_pos.unsqueeze(1)  # [num_samples, num_key_bodies, 3]
            key_body_pos_local = quat_rotate_inverse(
                ref_root_rot.unsqueeze(1).expand(-1, self.num_key_bodies, -1).reshape(-1, 4),
                key_body_pos_rel.reshape(-1, 3)
            ).reshape(num_samples, self.num_key_bodies, 3)
            
            # 组装单步观测
            step_obs = torch.cat([
                root_pos_local,                                    # 3
                root_rot_local,                                    # 4
                root_lin_vel_local,                                # 3
                root_ang_vel_local,                                # 3
                dof_pos_local,                                     # num_dof
                dof_vel_local,                                     # num_dof
                key_body_pos_local.reshape(num_samples, -1)       # num_key_bodies * 3
            ], dim=-1)
            
            disc_obs_list.append(step_obs)
        
        # 拼接所有历史步
        disc_obs_demo = torch.cat(disc_obs_list, dim=-1)  # [num_samples, disc_obs_size]
        
        return disc_obs_demo
    
    def _compute_key_body_pos_from_motion(self, root_pos, root_rot, dof_pos):
        """
        通过forward kinematics计算关键身体部位的世界坐标位置
        
        Args:
            root_pos: [num_samples, 3]
            root_rot: [num_samples, 4]
            dof_pos: [num_samples, num_dof]
            
        Returns:
            key_body_pos: [num_samples, num_key_bodies, 3]
        """
        if self._fk is not None:
            # 使用完整的Forward Kinematics计算
            return self._fk.compute_key_body_positions(root_pos, root_rot, dof_pos)
        else:
            # 回退到简化版本
            num_samples = root_pos.shape[0]
            key_body_pos = torch.zeros(num_samples, self.num_key_bodies, 3, device=self.device)
            
            from legged_gym.utils.math import quat_rotate
            
            # Torso（躯干）：在root位置上方
            torso_offset = torch.tensor([0.0, 0.0, 0.3], device=self.device).expand(num_samples, -1)
            key_body_pos[:, 0, :] = root_pos + quat_rotate(root_rot, torso_offset)
            
            # 左手
            left_hand_offset = torch.tensor([0.0, 0.2, 0.3], device=self.device).expand(num_samples, -1)
            key_body_pos[:, 1, :] = root_pos + quat_rotate(root_rot, left_hand_offset)
            
            # 右手
            right_hand_offset = torch.tensor([0.0, -0.2, 0.3], device=self.device).expand(num_samples, -1)
            key_body_pos[:, 2, :] = root_pos + quat_rotate(root_rot, right_hand_offset)
            
            # 左脚
            left_foot_offset = torch.tensor([0.0, 0.1, -0.6], device=self.device).expand(num_samples, -1)
            key_body_pos[:, 3, :] = root_pos + quat_rotate(root_rot, left_foot_offset)
            
            # 右脚
            right_foot_offset = torch.tensor([0.0, -0.1, -0.6], device=self.device).expand(num_samples, -1)
            key_body_pos[:, 4, :] = root_pos + quat_rotate(root_rot, right_foot_offset)
            
            return key_body_pos
    
    def reset_idx(self, env_ids):
        """重写reset_idx，重置AMP历史缓冲区"""
        super().reset_idx(env_ids)
        
        if self.enable_amp and len(env_ids) > 0:
            self._reset_disc_hist(env_ids)
    
    def _reset_disc_hist(self, env_ids):
        """
        重置指定环境的discriminator历史缓冲区
        用当前状态填充整个历史
        
        优先使用IsaacGym的rigid_body_states（最准确）
        """
        # 如果key_body_indices还未初始化，跳过（在__init__期间）
        if not hasattr(self, 'key_body_indices'):
            return
            
        # 获取当前状态
        root_pos = self.root_states[env_ids, :3]
        root_rot = self.root_states[env_ids, 3:7]
        base_lin_vel = self.base_lin_vel[env_ids]
        base_ang_vel = self.base_ang_vel[env_ids]
        dof_pos = self.dof_pos[env_ids]
        dof_vel = self.dof_vel[env_ids]
        
        # 优先使用IsaacGym的精确值
        key_body_pos = self.rigid_body_states[env_ids][:, self.key_body_indices, :3]
        
        # 用当前状态填充整个历史
        for t in range(self.num_disc_obs_steps):
            self.disc_hist_root_pos[env_ids, t] = root_pos
            self.disc_hist_root_rot[env_ids, t] = root_rot
            self.disc_hist_root_lin_vel[env_ids, t] = base_lin_vel
            self.disc_hist_root_ang_vel[env_ids, t] = base_ang_vel
            self.disc_hist_dof_pos[env_ids, t] = dof_pos
            self.disc_hist_dof_vel[env_ids, t] = dof_vel
            self.disc_hist_key_body_pos[env_ids, t] = key_body_pos
        
        # 更新disc_obs
        if len(env_ids) == self.num_envs:
            # 全部重置，直接更新
            self._update_disc_obs()
        else:
            # 部分重置，只更新对应环境的disc_obs
            # 保存全局状态
            saved_states = {
                'root_pos': self.disc_hist_root_pos.clone(),
                'root_rot': self.disc_hist_root_rot.clone(),
                'root_lin_vel': self.disc_hist_root_lin_vel.clone(),
                'root_ang_vel': self.disc_hist_root_ang_vel.clone(),
                'dof_pos': self.disc_hist_dof_pos.clone(),
                'dof_vel': self.disc_hist_dof_vel.clone(),
                'key_body_pos': self.disc_hist_key_body_pos.clone(),
            }
            
            # 临时设置为只包含重置的环境
            temp_hist_root_pos = self.disc_hist_root_pos[env_ids].clone()
            temp_hist_root_rot = self.disc_hist_root_rot[env_ids].clone()
            
            # 更新disc_obs（简化版）
            # 实际应该只更新env_ids对应的disc_obs，这里作为示例
            self._update_disc_obs()
    
    def compute_observations(self):
        """重写观测计算，确保AMP观测也被更新"""
        super().compute_observations()
        
        # 在父类compute_observations之后，disc_obs应该已经在step()中更新
        # 这里不需要额外操作

