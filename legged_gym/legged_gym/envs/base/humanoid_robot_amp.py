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
from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, normalize, quat_rotate


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
            # AMP观测配置（参考MimicKit）
            self.amp_global_obs = getattr(cfg.env, 'amp_global_obs', True)  # 是否使用全局坐标系
            self.amp_root_height_obs = getattr(cfg.env, 'amp_root_height_obs', True)  # 是否包含root高度
            # 预先获取key_body_names，避免在reset_idx时出错
            if hasattr(cfg.asset, 'key_bodies'):
                self.key_body_names = cfg.asset.key_bodies
            else:
                self.key_body_names = [cfg.asset.left_foot_name, cfg.asset.right_foot_name]
            self.num_key_bodies = len(self.key_body_names)
            print(f"[AMP] 启用AMP，discriminator观测步数: {self.num_disc_obs_steps}")
            print(f"[AMP] global_obs={self.amp_global_obs}, root_height_obs={self.amp_root_height_obs}")
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, save)
        
        # 保存cfg引用（用于后续方法）
        self._cfg = cfg
        
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
        
        # 单步观测维度计算（参考MimicKit格式）：
        # - root_pos: 2或3维（根据root_height_obs）
        # - root_rot: 6维（tan_norm格式）
        # - joint_rot: num_dof * 6维（每个关节6维tan_norm，完全对齐MimicKit）
        # - key_pos: num_key_bodies * 3维（如果有关键部位）
        # - root_vel: 3维
        # - root_ang_vel: 3维
        # - dof_vel: num_dof维
        root_pos_dim = 3 if self.amp_root_height_obs else 2
        root_rot_dim = 6  # tan_norm格式
        joint_rot_dim = self.num_dof * 6  # 每个关节6维tan_norm（完全对齐MimicKit）
        key_pos_dim = self.num_key_bodies * 3 if self.num_key_bodies > 0 else 0
        root_vel_dim = 3
        root_ang_vel_dim = 3
        dof_vel_dim = self.num_dof
        
        pos_obs_dim = root_pos_dim + root_rot_dim + joint_rot_dim + key_pos_dim
        vel_obs_dim = root_vel_dim + root_ang_vel_dim + dof_vel_dim
        single_step_disc_obs_size = pos_obs_dim + vel_obs_dim
        
        # 总维度 = 单步维度 * 历史步数
        self.disc_obs_size = single_step_disc_obs_size * self.num_disc_obs_steps
        
        # 打印详细的维度信息（用于调试）
        print(f"[AMP] 单步观测维度详情:")
        print(f"  - root_pos: {root_pos_dim}维")
        print(f"  - root_rot: {root_rot_dim}维 (tan_norm)")
        print(f"  - joint_rot: {joint_rot_dim}维 (num_dof={self.num_dof} * 6维tan_norm)")
        print(f"  - key_pos: {key_pos_dim}维")
        print(f"  - root_vel: {root_vel_dim}维")
        print(f"  - root_ang_vel: {root_ang_vel_dim}维")
        print(f"  - dof_vel: {dof_vel_dim}维")
        print(f"  - 单步总维度: {single_step_disc_obs_size}维")
        print(f"  - 总维度 ({self.num_disc_obs_steps}步): {self.disc_obs_size}维")
        
        # 启用调试标志（可通过配置控制）
        cfg = self._cfg if hasattr(self, '_cfg') else None
        if cfg and hasattr(cfg, 'env') and hasattr(cfg.env, 'debug_amp_obs'):
            self._debug_amp_obs = cfg.env.debug_amp_obs
        else:
            self._debug_amp_obs = False
        
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
        
        # ===== 调试信息：历史buffer更新 =====
        if hasattr(self, '_debug_amp_obs') and self._debug_amp_obs and self.common_step_counter % 100 == 0:
            print(f"[AMP DEBUG] 历史buffer更新 (step={self.common_step_counter}):")
            print(f"  - root_pos范围: [{self.disc_hist_root_pos[:, -1].min().item():.4f}, {self.disc_hist_root_pos[:, -1].max().item():.4f}]")
            print(f"  - root_rot范围: [{self.disc_hist_root_rot[:, -1].min().item():.4f}, {self.disc_hist_root_rot[:, -1].max().item():.4f}]")
            print(f"  - root_lin_vel范围: [{self.disc_hist_root_lin_vel[:, -1].min().item():.4f}, {self.disc_hist_root_lin_vel[:, -1].max().item():.4f}]")
            print(f"  - dof_pos范围: [{self.disc_hist_dof_pos[:, -1].min().item():.4f}, {self.disc_hist_dof_pos[:, -1].max().item():.4f}]")
            print(f"  - key_body_pos范围: [{self.disc_hist_key_body_pos[:, -1].min().item():.4f}, {self.disc_hist_key_body_pos[:, -1].max().item():.4f}]")
    
    def _update_disc_obs(self):
        """
        计算discriminator观测（完全对齐MimicKit实现）
        
        参考: MimicKit/mimickit/envs/deepmimic_env.py compute_tar_obs
              MimicKit/mimickit/envs/amp_env.py compute_disc_vel_obs
        
        AMP Discriminator观测格式：
        1. 使用heading坐标系（只考虑yaw旋转）或全局坐标系（根据global_obs配置）
        2. 旋转使用tan_norm格式（6维）而非四元数（4维）
        3. key_pos相对于root_pos（而非ref_root_pos）
        4. root_pos根据root_height_obs决定是否包含z坐标
        """
        # ===== 参考状态（Reference Frame）：当前时刻（最后一步）的root状态 =====
        ref_root_pos = self.disc_hist_root_pos[:, -1, :]  # [num_envs, 3]
        ref_root_rot = self.disc_hist_root_rot[:, -1, :]  # [num_envs, 4]
        
        # 计算相对状态
        disc_obs_list = []
        
        for t in range(self.num_disc_obs_steps):
            # ===== 历史步t的状态（世界坐标系）=====
            root_pos = self.disc_hist_root_pos[:, t, :]      # [num_envs, 3]
            root_rot = self.disc_hist_root_rot[:, t, :]      # [num_envs, 4]
            root_lin_vel = self.disc_hist_root_lin_vel[:, t, :]  # [num_envs, 3]
            root_ang_vel = self.disc_hist_root_ang_vel[:, t, :]  # [num_envs, 3]
            dof_pos = self.disc_hist_dof_pos[:, t, :]        # [num_envs, num_dof]
            dof_vel = self.disc_hist_dof_vel[:, t, :]        # [num_envs, num_dof]
            key_body_pos = self.disc_hist_key_body_pos[:, t, :, :]  # [num_envs, num_key_bodies, 3]
            
            # ===== 位置观测（参考MimicKit compute_tar_obs）=====
            # 1. Root位置：相对于ref_root_pos
            ref_root_pos_expand = ref_root_pos.unsqueeze(1)  # [num_envs, 1, 3]
            root_pos_obs = root_pos.unsqueeze(1) - ref_root_pos_expand  # [num_envs, 1, 3]
            
            # 2. Key位置：相对于root_pos（注意：不是ref_root_pos！）
            if self.num_key_bodies > 0:
                key_pos_obs = key_body_pos - root_pos.unsqueeze(1)  # [num_envs, num_key_bodies, 3]
            else:
                key_pos_obs = torch.zeros(self.num_envs, 0, 3, device=self.device)
            
            # 3. 坐标系转换（如果global_obs=False，使用heading坐标系）
            if not self.amp_global_obs:
                # 使用heading坐标系（只考虑yaw旋转）
                heading_inv_rot = self._calc_heading_quat_inv(ref_root_rot)  # [num_envs, 4]
                heading_inv_rot_expand = heading_inv_rot.unsqueeze(1)  # [num_envs, 1, 4]
                
                # 转换root_pos_obs
                root_pos_obs_flat = root_pos_obs.reshape(-1, 3)  # [num_envs, 3]
                heading_inv_rot_flat = heading_inv_rot_expand.reshape(-1, 4)  # [num_envs, 4]
                root_pos_obs_flat = quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
                root_pos_obs = root_pos_obs_flat.reshape(self.num_envs, 1, 3)
                
                # 转换root_rot
                root_rot = quat_mul(heading_inv_rot_expand, root_rot.unsqueeze(1))  # [num_envs, 1, 4]
                root_rot = root_rot.squeeze(1)  # [num_envs, 4]
                
                # 转换key_pos_obs
                if self.num_key_bodies > 0:
                    # 将heading_inv_rot扩展到每个key body: [num_envs, 4] -> [num_envs, num_key_bodies, 4]
                    heading_inv_rot_expand_key = heading_inv_rot.unsqueeze(1).expand(-1, self.num_key_bodies, -1)  # [num_envs, num_key_bodies, 4]
                    key_pos_obs_flat = key_pos_obs.reshape(-1, 3)  # [num_envs * num_key_bodies, 3]
                    heading_inv_rot_flat_key = heading_inv_rot_expand_key.reshape(-1, 4)  # [num_envs * num_key_bodies, 4]
                    key_pos_obs_flat = quat_rotate(heading_inv_rot_flat_key, key_pos_obs_flat)
                    key_pos_obs = key_pos_obs_flat.reshape(self.num_envs, self.num_key_bodies, 3)
            
            # 4. Root位置维度处理（根据root_height_obs）
            root_pos_obs = root_pos_obs.squeeze(1)  # [num_envs, 3]
            if self.amp_root_height_obs:
                # 包含z坐标，但z坐标使用绝对高度（参考MimicKit）
                root_pos_obs[:, 2] = root_pos[:, 2]
            else:
                # 只包含x, y坐标
                root_pos_obs = root_pos_obs[:, :2]  # [num_envs, 2]
            
            # 5. 旋转转换为tan_norm格式（6维）
            root_rot_tan_norm = self._quat_to_tan_norm(root_rot)  # [num_envs, 6]
            
            # 6. 关节旋转转换为tan_norm格式（参考MimicKit）
            # MimicKit: joint_rot (四元数) -> tan_norm (6维) -> [num_joints * 6]
            # 步骤1: 将dof_pos（角度）转换为joint_rot（四元数）
            joint_rot = self._dof_pos_to_joint_rot(dof_pos)  # [num_envs, num_dof, 4]
            
            # 步骤2: 将joint_rot（四元数）转换为tan_norm（6维）
            # 参考MimicKit: joint_rot_flat -> quat_to_tan_norm -> reshape
            joint_rot_flat = joint_rot.reshape(-1, 4)  # [num_envs * num_dof, 4]
            joint_rot_tan_norm_flat = self._quat_to_tan_norm(joint_rot_flat)  # [num_envs * num_dof, 6]
            joint_rot_obs = joint_rot_tan_norm_flat.reshape(self.num_envs, -1)  # [num_envs, num_dof * 6]
            
            # ===== 速度观测（参考MimicKit compute_disc_vel_obs）=====
            if not self.amp_global_obs:
                # 使用heading坐标系
                heading_inv_rot = self._calc_heading_quat_inv(ref_root_rot)  # [num_envs, 4]
                root_vel_obs = quat_rotate(heading_inv_rot, root_lin_vel)  # [num_envs, 3]
                root_ang_vel_obs = quat_rotate(heading_inv_rot, root_ang_vel)  # [num_envs, 3]
            else:
                # 使用全局坐标系
                root_vel_obs = root_lin_vel
                root_ang_vel_obs = root_ang_vel
            
            # ===== 组装位置观测 =====
            pos_obs_list = [root_pos_obs, root_rot_tan_norm, joint_rot_obs]
            if self.num_key_bodies > 0:
                key_pos_obs_flat = key_pos_obs.reshape(self.num_envs, -1)  # [num_envs, num_key_bodies * 3]
                pos_obs_list.append(key_pos_obs_flat)
            pos_obs = torch.cat(pos_obs_list, dim=-1)  # [num_envs, pos_obs_dim]
            
            # ===== 组装速度观测 =====
            vel_obs = torch.cat([root_vel_obs, root_ang_vel_obs, dof_vel], dim=-1)  # [num_envs, vel_obs_dim]
            
            # ===== 组装单步观测 =====
            step_obs = torch.cat([pos_obs, vel_obs], dim=-1)  # [num_envs, step_obs_dim]
            
            disc_obs_list.append(step_obs)
        
        # 拼接所有历史步
        self.disc_obs_buf[:] = torch.cat(disc_obs_list, dim=-1)  # [num_envs, disc_obs_size]
        
        # ===== 调试信息：观测维度验证（每100步打印一次，避免日志过多）=====
        if hasattr(self, '_debug_amp_obs') and self._debug_amp_obs:
            # 只在特定步数打印详细调试信息
            should_print_debug = (not hasattr(self, '_last_debug_step') or 
                                 self.common_step_counter - self._last_debug_step >= 100)
            
            actual_size = self.disc_obs_buf.shape[-1]
            expected_size = self.disc_obs_size
            if actual_size != expected_size:
                print(f"[AMP DEBUG] ⚠️ 观测维度不匹配！实际: {actual_size}, 期望: {expected_size}")
            elif should_print_debug:
                print(f"[AMP DEBUG] ✅ 观测维度正确: {actual_size} (step={self.common_step_counter})")
            
            if should_print_debug:
                # 检查数值范围
                obs_min = self.disc_obs_buf.min().item()
                obs_max = self.disc_obs_buf.max().item()
                obs_mean = self.disc_obs_buf.mean().item()
                obs_std = self.disc_obs_buf.std().item()
                print(f"[AMP DEBUG] 观测数值范围: min={obs_min:.4f}, max={obs_max:.4f}, mean={obs_mean:.4f}, std={obs_std:.4f}")
                
                # 检查是否有NaN或Inf
                has_nan = torch.isnan(self.disc_obs_buf).any().item()
                has_inf = torch.isinf(self.disc_obs_buf).any().item()
                if has_nan or has_inf:
                    print(f"[AMP DEBUG] ⚠️ 观测包含异常值！NaN: {has_nan}, Inf: {has_inf}")
                else:
                    print(f"[AMP DEBUG] ✅ 观测数值正常（无NaN/Inf）")
                
                # 检查各组件维度
                print(f"[AMP DEBUG] 观测组件维度:")
                print(f"  - root_pos_obs: {root_pos_obs.shape if 'root_pos_obs' in locals() else 'N/A'}")
                print(f"  - root_rot_tan_norm: {root_rot_tan_norm.shape if 'root_rot_tan_norm' in locals() else 'N/A'}")
                print(f"  - joint_rot_obs: {joint_rot_obs.shape if 'joint_rot_obs' in locals() else 'N/A'}")
                print(f"  - key_pos_obs: {key_pos_obs.shape if 'key_pos_obs' in locals() and self.num_key_bodies > 0 else 'N/A'}")
                print(f"  - root_vel_obs: {root_vel_obs.shape if 'root_vel_obs' in locals() else 'N/A'}")
                print(f"  - root_ang_vel_obs: {root_ang_vel_obs.shape if 'root_ang_vel_obs' in locals() else 'N/A'}")
                print(f"  - dof_vel: {dof_vel.shape if 'dof_vel' in locals() else 'N/A'}")
                
                # 更新最后打印步数
                self._last_debug_step = self.common_step_counter
    
    def _update_disc_obs_for_envs(self, env_ids):
        """
        只更新指定环境的discriminator观测（使用与_update_disc_obs相同的逻辑）
        
        Args:
            env_ids: 要更新的环境ID列表
        """
        if len(env_ids) == 0:
            return
        
        # 参考root状态（当前时刻）
        ref_root_pos = self.disc_hist_root_pos[env_ids, -1, :]  # [len(env_ids), 3]
        ref_root_rot = self.disc_hist_root_rot[env_ids, -1, :]  # [len(env_ids), 4]
        
        # 计算相对状态
        disc_obs_list = []
        
        for t in range(self.num_disc_obs_steps):
            # 当前历史步的root状态（只取env_ids对应的环境）
            root_pos = self.disc_hist_root_pos[env_ids, t, :]
            root_rot = self.disc_hist_root_rot[env_ids, t, :]
            root_lin_vel = self.disc_hist_root_lin_vel[env_ids, t, :]
            root_ang_vel = self.disc_hist_root_ang_vel[env_ids, t, :]
            dof_pos = self.disc_hist_dof_pos[env_ids, t, :]
            dof_vel = self.disc_hist_dof_vel[env_ids, t, :]
            key_body_pos = self.disc_hist_key_body_pos[env_ids, t, :, :]  # [len(env_ids), num_key_bodies, 3]
            
            # ===== 位置观测（参考MimicKit compute_tar_obs）=====
            # 1. Root位置：相对于ref_root_pos
            ref_root_pos_expand = ref_root_pos.unsqueeze(1)  # [len(env_ids), 1, 3]
            root_pos_obs = root_pos.unsqueeze(1) - ref_root_pos_expand  # [len(env_ids), 1, 3]
            
            # 2. Key位置：相对于root_pos（注意：不是ref_root_pos！）
            if self.num_key_bodies > 0:
                key_pos_obs = key_body_pos - root_pos.unsqueeze(1)  # [len(env_ids), num_key_bodies, 3]
            else:
                key_pos_obs = torch.zeros(len(env_ids), 0, 3, device=self.device)
            
            # 3. 坐标系转换（如果global_obs=False，使用heading坐标系）
            if not self.amp_global_obs:
                # 使用heading坐标系（只考虑yaw旋转）
                heading_inv_rot = self._calc_heading_quat_inv(ref_root_rot)  # [len(env_ids), 4]
                heading_inv_rot_expand = heading_inv_rot.unsqueeze(1)  # [len(env_ids), 1, 4]
                
                # 转换root_pos_obs
                root_pos_obs_flat = root_pos_obs.reshape(-1, 3)  # [len(env_ids), 3]
                heading_inv_rot_flat = heading_inv_rot_expand.reshape(-1, 4)  # [len(env_ids), 4]
                root_pos_obs_flat = quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
                root_pos_obs = root_pos_obs_flat.reshape(len(env_ids), 1, 3)
                
                # 转换root_rot
                root_rot = quat_mul(heading_inv_rot_expand, root_rot.unsqueeze(1))  # [len(env_ids), 1, 4]
                root_rot = root_rot.squeeze(1)  # [len(env_ids), 4]
                
                # 转换key_pos_obs
                if self.num_key_bodies > 0:
                    # 将heading_inv_rot扩展到每个key body: [len(env_ids), 4] -> [len(env_ids), num_key_bodies, 4]
                    heading_inv_rot_expand_key = heading_inv_rot.unsqueeze(1).expand(-1, self.num_key_bodies, -1)  # [len(env_ids), num_key_bodies, 4]
                    key_pos_obs_flat = key_pos_obs.reshape(-1, 3)  # [len(env_ids) * num_key_bodies, 3]
                    heading_inv_rot_flat_key = heading_inv_rot_expand_key.reshape(-1, 4)  # [len(env_ids) * num_key_bodies, 4]
                    key_pos_obs_flat = quat_rotate(heading_inv_rot_flat_key, key_pos_obs_flat)
                    key_pos_obs = key_pos_obs_flat.reshape(len(env_ids), self.num_key_bodies, 3)
            
            # 4. Root位置维度处理（根据root_height_obs）
            root_pos_obs = root_pos_obs.squeeze(1)  # [len(env_ids), 3]
            if self.amp_root_height_obs:
                # 包含z坐标，但z坐标使用绝对高度（参考MimicKit）
                root_pos_obs[:, 2] = root_pos[:, 2]
            else:
                # 只包含x, y坐标
                root_pos_obs = root_pos_obs[:, :2]  # [len(env_ids), 2]
            
            # 5. 旋转转换为tan_norm格式（6维）
            root_rot_tan_norm = self._quat_to_tan_norm(root_rot)  # [len(env_ids), 6]
            
            # 6. 关节旋转转换为tan_norm格式（参考MimicKit）
            # 步骤1: 将dof_pos（角度）转换为joint_rot（四元数）
            joint_rot = self._dof_pos_to_joint_rot(dof_pos)  # [len(env_ids), num_dof, 4]
            
            # 步骤2: 将joint_rot（四元数）转换为tan_norm（6维）
            joint_rot_flat = joint_rot.reshape(-1, 4)  # [len(env_ids) * num_dof, 4]
            joint_rot_tan_norm_flat = self._quat_to_tan_norm(joint_rot_flat)  # [len(env_ids) * num_dof, 6]
            joint_rot_obs = joint_rot_tan_norm_flat.reshape(len(env_ids), -1)  # [len(env_ids), num_dof * 6]
            
            # ===== 速度观测（参考MimicKit compute_disc_vel_obs）=====
            if not self.amp_global_obs:
                # 使用heading坐标系
                heading_inv_rot = self._calc_heading_quat_inv(ref_root_rot)  # [len(env_ids), 4]
                root_vel_obs = quat_rotate(heading_inv_rot, root_lin_vel)  # [len(env_ids), 3]
                root_ang_vel_obs = quat_rotate(heading_inv_rot, root_ang_vel)  # [len(env_ids), 3]
            else:
                # 使用全局坐标系
                root_vel_obs = root_lin_vel
                root_ang_vel_obs = root_ang_vel
            
            # ===== 组装位置观测 =====
            pos_obs_list = [root_pos_obs, root_rot_tan_norm, joint_rot_obs]
            if self.num_key_bodies > 0:
                key_pos_obs_flat = key_pos_obs.reshape(len(env_ids), -1)  # [len(env_ids), num_key_bodies * 3]
                pos_obs_list.append(key_pos_obs_flat)
            pos_obs = torch.cat(pos_obs_list, dim=-1)  # [len(env_ids), pos_obs_dim]
            
            # ===== 组装速度观测 =====
            vel_obs = torch.cat([root_vel_obs, root_ang_vel_obs, dof_vel], dim=-1)  # [len(env_ids), vel_obs_dim]
            
            # ===== 组装单步观测 =====
            step_obs = torch.cat([pos_obs, vel_obs], dim=-1)  # [len(env_ids), step_obs_dim]
            
            disc_obs_list.append(step_obs)
        
        # 拼接所有历史步
        disc_obs_for_envs = torch.cat(disc_obs_list, dim=-1)  # [len(env_ids), disc_obs_size]
        
        # 只更新env_ids对应的disc_obs
        self.disc_obs_buf[env_ids] = disc_obs_for_envs
    
    def _quat_conjugate(self, q):
        """四元数共轭 (用于计算逆旋转)"""
        return torch.cat([-q[:, :3], q[:, 3:4]], dim=-1)
    
    def _calc_heading(self, q):
        """
        计算四元数的heading（yaw角度，参考MimicKit）
        
        Args:
            q: [..., 4] 四元数 (x, y, z, w)
            
        Returns:
            heading: [...,] yaw角度（弧度）
        """
        ref_dir = torch.zeros_like(q[..., 0:3])
        ref_dir[..., 0] = 1  # x方向
        rot_dir = quat_rotate(q, ref_dir)
        heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
        return heading
    
    def _calc_heading_quat_inv(self, q):
        """
        计算heading的逆四元数（只考虑yaw旋转，参考MimicKit）
        
        参考: MimicKit/mimickit/util/torch_util.py 第333-340行
        calc_heading_quat_inv(q):
            heading = calc_heading(q)
            axis = torch.zeros_like(q[..., 0:3])
            axis[..., 2] = 1
            heading_q = axis_angle_to_quat(axis, -heading)
        
        Args:
            q: [..., 4] 四元数
            
        Returns:
            heading_inv_quat: [..., 4] 逆heading四元数
        """
        heading = self._calc_heading(q)
        axis = torch.zeros_like(q[..., 0:3])
        axis[..., 2] = 1  # z轴
        
        # 使用axis_angle_to_quat（与MimicKit完全一致）
        heading_inv_quat = self._axis_angle_to_quat(axis, -heading)
        return heading_inv_quat
    
    def _axis_angle_to_quat(self, axis, angle):
        """
        将axis-angle转换为四元数（完全对齐MimicKit实现）
        
        参考: MimicKit/mimickit/util/torch_util.py 第183-188行
        axis_angle_to_quat(axis, angle):
            theta = (angle / 2).unsqueeze(-1)
            xyz = normalize(axis) * theta.sin()
            w = theta.cos()
            return quat_unit(torch.cat([xyz, w], dim=-1))
        
        Args:
            axis: [..., 3] 旋转轴（支持任意维度）
            angle: [...,] 旋转角度（弧度，支持任意维度，可以是标量）
            
        Returns:
            quat: [..., 4] 四元数 (x, y, z, w)
        """
        # 计算theta: [..., 1]（与MimicKit完全一致）
        theta = (angle / 2).unsqueeze(-1)
        
        # 归一化轴
        axis_norm = normalize(axis)  # [..., 3]
        
        # 计算四元数（与MimicKit完全一致）
        # 注意：MimicKit使用theta.sin()和theta.cos()（tensor方法），我们使用torch.sin/cos（功能等价）
        xyz = axis_norm * torch.sin(theta)  # [..., 3]
        w = torch.cos(theta)  # [..., 1]
        
        quat = torch.cat([xyz, w], dim=-1)  # [..., 4]
        
        # 归一化四元数（quat_unit = normalize，与MimicKit一致）
        quat = normalize(quat)
        return quat
    
    def _dof_pos_to_joint_rot(self, dof_pos):
        """
        将dof_pos（角度）转换为joint_rot（四元数），参考MimicKit实现
        
        G1机器人所有关节都是单轴关节（hinge/revolute），每个DOF对应一个关节。
        需要根据关节轴方向将角度转换为四元数。
        
        Args:
            dof_pos: [..., num_dof] 关节角度（弧度）
            
        Returns:
            joint_rot: [..., num_dof, 4] 关节旋转四元数
        """
        # G1机器人DOF顺序（27个DOF，每个都是单轴关节）:
        # 0-5: 左腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        # 6-11: 右腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        # 12: waist_yaw
        # 13-19: 左臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
        # 20-26: 右臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
        
        # 定义每个关节的旋转轴（基于URDF标准，单轴关节通常绕y轴或z轴旋转）
        # 对于人形机器人，大多数关节绕y轴（pitch）或z轴（yaw/roll）旋转
        # 这里使用常见的关节轴配置，如果需要可以从URDF解析
        
        num_dof = dof_pos.shape[-1]
        device = dof_pos.device
        dtype = dof_pos.dtype
        
        # 扩展dof_pos维度以便处理
        original_shape = dof_pos.shape
        dof_pos_flat = dof_pos.reshape(-1, num_dof)  # [batch, num_dof]
        batch_size = dof_pos_flat.shape[0]
        
        # ===== 关节轴方向（已从URDF验证）=====
        # 所有关节轴方向已从 g1.urdf 文件中验证，确保与URDF定义完全一致
        # URDF格式: <axis xyz="x y z"/> 表示旋转轴方向
        # - pitch关节：绕y轴 (0, 1, 0)
        # - roll关节：绕x轴 (1, 0, 0)  
        # - yaw关节：绕z轴 (0, 0, 1)
        joint_axes = torch.zeros(num_dof, 3, device=device, dtype=dtype)
        
        # 左腿 (0-5): 已验证URDF
        # left_hip_pitch_joint: axis xyz="0 1 0"
        joint_axes[0] = torch.tensor([0, 1, 0], device=device, dtype=dtype)  # hip_pitch
        # left_hip_roll_joint: axis xyz="1 0 0"
        joint_axes[1] = torch.tensor([1, 0, 0], device=device, dtype=dtype)  # hip_roll
        # left_hip_yaw_joint: axis xyz="0 0 1"
        joint_axes[2] = torch.tensor([0, 0, 1], device=device, dtype=dtype)  # hip_yaw
        # left_knee_joint: axis xyz="0 1 0"
        joint_axes[3] = torch.tensor([0, 1, 0], device=device, dtype=dtype)  # knee
        # left_ankle_pitch_joint: axis xyz="0 1 0"
        joint_axes[4] = torch.tensor([0, 1, 0], device=device, dtype=dtype)  # ankle_pitch
        # left_ankle_roll_joint: axis xyz="1 0 0"
        joint_axes[5] = torch.tensor([1, 0, 0], device=device, dtype=dtype)  # ankle_roll
        
        # 右腿 (6-11): 已验证URDF（与左腿对称）
        joint_axes[6:12] = joint_axes[0:6].clone()
        
        # waist_yaw (12): 已验证URDF - waist_yaw_joint: axis xyz="0 0 1"
        joint_axes[12] = torch.tensor([0, 0, 1], device=device, dtype=dtype)
        
        # 左臂 (13-19): 已验证URDF
        # left_shoulder_pitch_joint: axis xyz="0 1 0"
        joint_axes[13] = torch.tensor([0, 1, 0], device=device, dtype=dtype)  # shoulder_pitch
        # left_shoulder_roll_joint: axis xyz="1 0 0"
        joint_axes[14] = torch.tensor([1, 0, 0], device=device, dtype=dtype)  # shoulder_roll
        # left_shoulder_yaw_joint: axis xyz="0 0 1"
        joint_axes[15] = torch.tensor([0, 0, 1], device=device, dtype=dtype)  # shoulder_yaw
        # left_elbow_joint: axis xyz="0 1 0"
        joint_axes[16] = torch.tensor([0, 1, 0], device=device, dtype=dtype)  # elbow
        # left_wrist_roll_joint: axis xyz="1 0 0"
        joint_axes[17] = torch.tensor([1, 0, 0], device=device, dtype=dtype)  # wrist_roll
        # left_wrist_pitch_joint: axis xyz="0 1 0"
        joint_axes[18] = torch.tensor([0, 1, 0], device=device, dtype=dtype)  # wrist_pitch
        # left_wrist_yaw_joint: axis xyz="0 0 1"
        joint_axes[19] = torch.tensor([0, 0, 1], device=device, dtype=dtype)  # wrist_yaw
        
        # 右臂 (20-26): 已验证URDF（与左臂对称）
        joint_axes[20:27] = joint_axes[13:20].clone()
        
        # 将每个DOF的角度转换为四元数（向量化实现）
        # 扩展joint_axes: [num_dof, 3] -> [batch, num_dof, 3]
        joint_axes_expanded = joint_axes.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_dof, 3]
        
        # 扩展dof_pos: [batch, num_dof] -> [batch, num_dof, 1]
        angles_expanded = dof_pos_flat.unsqueeze(-1)  # [batch, num_dof, 1]
        
        # 计算theta: [batch, num_dof, 1]
        theta = angles_expanded / 2
        
        # 归一化轴: [batch, num_dof, 3]
        axis_norm = normalize(joint_axes_expanded.reshape(-1, 3)).reshape(batch_size, num_dof, 3)
        
        # 计算四元数
        xyz = axis_norm * torch.sin(theta)  # [batch, num_dof, 3]
        w = torch.cos(theta)  # [batch, num_dof, 1]
        
        # 拼接: [batch, num_dof, 4]
        joint_rot_flat = torch.cat([xyz, w], dim=-1)
        
        # 归一化四元数
        joint_rot_flat = normalize(joint_rot_flat.reshape(-1, 4)).reshape(batch_size, num_dof, 4)
        
        # 恢复原始形状
        joint_rot = joint_rot_flat.reshape(*original_shape, 4)  # [..., num_dof, 4]
        return joint_rot
    
    def _quat_to_tan_norm(self, q):
        """
        将四元数转换为tan_norm格式（6维，参考MimicKit）
        
        Args:
            q: [..., 4] 四元数 (x, y, z, w)
            
        Returns:
            tan_norm: [..., 6] tan_norm表示
        """
        ref_tan = torch.zeros_like(q[..., 0:3])
        ref_tan[..., 0] = 1  # x方向
        tan = quat_rotate(q, ref_tan)
        
        ref_norm = torch.zeros_like(q[..., 0:3])
        ref_norm[..., -1] = 1  # z方向
        norm = quat_rotate(q, ref_norm)
        
        tan_norm = torch.cat([tan, norm], dim=-1)
        return tan_norm
    
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
        从参考动作库采样discriminator观测（完全对齐MimicKit实现）
        
        与_update_disc_obs使用相同的观测格式：
        1. 使用heading坐标系（global_obs=False）或全局坐标系（global_obs=True）
        2. 旋转使用tan_norm格式（6维）
        3. key_pos相对于root_pos
        4. root_pos根据root_height_obs决定是否包含z坐标
        
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
        
        # ===== 调试信息：采样信息 =====
        if hasattr(self, '_debug_amp_obs') and self._debug_amp_obs and self.common_step_counter % 200 == 0:
            print(f"[AMP DEBUG] Expert采样 (step={self.common_step_counter}):")
            print(f"  - 采样数量: {num_samples}")
            print(f"  - Motion时间范围: [{motion_times.min().item():.4f}, {motion_times.max().item():.4f}]秒")
        
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
            
            # 获取历史时刻的root速度（使用预先计算的速度，参考MimicKit）
            # 计算帧索引用于插值
            frame_indices = hist_times / self._motion_lib.dt
            frame_idx0 = torch.floor(frame_indices).long()
            frame_idx1 = frame_idx0 + 1
            blend = frame_indices - frame_idx0.float()
            
            # 处理循环
            if self._motion_lib.loop_mode:
                frame_idx0 = frame_idx0 % self._motion_lib.num_frames
                frame_idx1 = frame_idx1 % self._motion_lib.num_frames
            else:
                frame_idx0 = torch.clamp(frame_idx0, 0, self._motion_lib.num_frames - 1)
                frame_idx1 = torch.clamp(frame_idx1, 0, self._motion_lib.num_frames - 1)
            
            blend = blend.unsqueeze(-1)
            hist_root_lin_vel = (self._motion_lib._frame_root_vel[frame_idx0] * (1 - blend) + 
                                self._motion_lib._frame_root_vel[frame_idx1] * blend)
            hist_root_ang_vel = (self._motion_lib._frame_root_ang_vel[frame_idx0] * (1 - blend) + 
                                self._motion_lib._frame_root_ang_vel[frame_idx1] * blend)
            
            # 处理None值
            if hist_root_pos is None:
                hist_root_pos = ref_root_pos.clone()
            if hist_root_rot is None:
                hist_root_rot = ref_root_rot.clone()
            if hist_root_lin_vel is None:
                hist_root_lin_vel = torch.zeros(num_samples, 3, device=self.device)
            if hist_root_ang_vel is None:
                hist_root_ang_vel = torch.zeros(num_samples, 3, device=self.device)
            
            # ===== 位置观测（参考MimicKit compute_tar_obs）=====
            # 1. Root位置：相对于ref_root_pos
            ref_root_pos_expand = ref_root_pos.unsqueeze(1)  # [num_samples, 1, 3]
            root_pos_obs = hist_root_pos.unsqueeze(1) - ref_root_pos_expand  # [num_samples, 1, 3]
            
            # 2. Key位置：相对于root_pos（注意：不是ref_root_pos！）
            # 先计算关键身体部位的世界坐标位置
            key_body_pos_world = self._compute_key_body_pos_from_motion(
                hist_root_pos, hist_root_rot, hist_dof_pos
            )  # [num_samples, num_key_bodies, 3]
            
            if self.num_key_bodies > 0:
                key_pos_obs = key_body_pos_world - hist_root_pos.unsqueeze(1)  # [num_samples, num_key_bodies, 3]
            else:
                key_pos_obs = torch.zeros(num_samples, 0, 3, device=self.device)
            
            # 3. 坐标系转换（如果global_obs=False，使用heading坐标系）
            if not self.amp_global_obs:
                # 使用heading坐标系（只考虑yaw旋转）
                heading_inv_rot = self._calc_heading_quat_inv(ref_root_rot)  # [num_samples, 4]
                heading_inv_rot_expand = heading_inv_rot.unsqueeze(1)  # [num_samples, 1, 4]
                
                # 转换root_pos_obs
                root_pos_obs_flat = root_pos_obs.reshape(-1, 3)  # [num_samples, 3]
                heading_inv_rot_flat = heading_inv_rot_expand.reshape(-1, 4)  # [num_samples, 4]
                root_pos_obs_flat = quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
                root_pos_obs = root_pos_obs_flat.reshape(num_samples, 1, 3)
                
                # 转换root_rot
                hist_root_rot_expand = hist_root_rot.unsqueeze(1)  # [num_samples, 1, 4]
                root_rot = quat_mul(heading_inv_rot_expand, hist_root_rot_expand)  # [num_samples, 1, 4]
                root_rot = root_rot.squeeze(1)  # [num_samples, 4]
                
                # 转换key_pos_obs
                if self.num_key_bodies > 0:
                    # 将heading_inv_rot扩展到每个key body: [num_samples, 4] -> [num_samples, num_key_bodies, 4]
                    heading_inv_rot_expand_key = heading_inv_rot.unsqueeze(1).expand(-1, self.num_key_bodies, -1)  # [num_samples, num_key_bodies, 4]
                    key_pos_obs_flat = key_pos_obs.reshape(-1, 3)  # [num_samples * num_key_bodies, 3]
                    heading_inv_rot_flat_key = heading_inv_rot_expand_key.reshape(-1, 4)  # [num_samples * num_key_bodies, 4]
                    key_pos_obs_flat = quat_rotate(heading_inv_rot_flat_key, key_pos_obs_flat)
                    key_pos_obs = key_pos_obs_flat.reshape(num_samples, self.num_key_bodies, 3)
            else:
                # 使用全局坐标系，不需要转换
                root_rot = hist_root_rot
            
            # 4. Root位置维度处理（根据root_height_obs）
            root_pos_obs = root_pos_obs.squeeze(1)  # [num_samples, 3]
            if self.amp_root_height_obs:
                # 包含z坐标，但z坐标使用绝对高度（参考MimicKit）
                root_pos_obs[:, 2] = hist_root_pos[:, 2]
            else:
                # 只包含x, y坐标
                root_pos_obs = root_pos_obs[:, :2]  # [num_samples, 2]
            
            # 5. 旋转转换为tan_norm格式（6维）
            root_rot_tan_norm = self._quat_to_tan_norm(root_rot)  # [num_samples, 6]
            
            # 6. 关节旋转转换为tan_norm格式（参考MimicKit）
            # 步骤1: 将dof_pos（角度）转换为joint_rot（四元数）
            joint_rot = self._dof_pos_to_joint_rot(hist_dof_pos)  # [num_samples, num_dof, 4]
            
            # 步骤2: 将joint_rot（四元数）转换为tan_norm（6维）
            joint_rot_flat = joint_rot.reshape(-1, 4)  # [num_samples * num_dof, 4]
            joint_rot_tan_norm_flat = self._quat_to_tan_norm(joint_rot_flat)  # [num_samples * num_dof, 6]
            joint_rot_obs = joint_rot_tan_norm_flat.reshape(num_samples, -1)  # [num_samples, num_dof * 6]
            
            # ===== 速度观测（参考MimicKit compute_disc_vel_obs）=====
            if not self.amp_global_obs:
                # 使用heading坐标系
                heading_inv_rot = self._calc_heading_quat_inv(ref_root_rot)  # [num_samples, 4]
                root_vel_obs = quat_rotate(heading_inv_rot, hist_root_lin_vel)  # [num_samples, 3]
                root_ang_vel_obs = quat_rotate(heading_inv_rot, hist_root_ang_vel)  # [num_samples, 3]
            else:
                # 使用全局坐标系
                root_vel_obs = hist_root_lin_vel
                root_ang_vel_obs = hist_root_ang_vel
            
            # ===== 组装位置观测 =====
            pos_obs_list = [root_pos_obs, root_rot_tan_norm, joint_rot_obs]
            if self.num_key_bodies > 0:
                key_pos_obs_flat = key_pos_obs.reshape(num_samples, -1)  # [num_samples, num_key_bodies * 3]
                pos_obs_list.append(key_pos_obs_flat)
            pos_obs = torch.cat(pos_obs_list, dim=-1)  # [num_samples, pos_obs_dim]
            
            # ===== 组装速度观测 =====
            vel_obs = torch.cat([root_vel_obs, root_ang_vel_obs, hist_dof_vel], dim=-1)  # [num_samples, vel_obs_dim]
            
            # ===== 组装单步观测 =====
            step_obs = torch.cat([pos_obs, vel_obs], dim=-1)  # [num_samples, step_obs_dim]
            
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
            # 如果FK不可用，返回相对于root的默认位置（使用关键部位的默认偏移）
            # 这是一个fallback，应该尽量避免
            num_samples = root_pos.shape[0]
            # 返回相对于root的默认位置（例如，脚部在root下方）
            key_body_pos = torch.zeros(num_samples, self.num_key_bodies, 3, device=self.device)
            # 可以根据key_body_names设置默认偏移
            # 这里暂时返回零向量（相对于root）
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
        # rigid_body_states形状: [num_envs, num_bodies, 13]
        # 先选择env_ids，再选择key_body_indices
        key_body_pos = self.rigid_body_states[env_ids, :, :3][:, self.key_body_indices, :]  # [len(env_ids), num_key_bodies, 3]
        
        # 用当前状态填充整个历史
        for t in range(self.num_disc_obs_steps):
            self.disc_hist_root_pos[env_ids, t] = root_pos
            self.disc_hist_root_rot[env_ids, t] = root_rot
            self.disc_hist_root_lin_vel[env_ids, t] = base_lin_vel
            self.disc_hist_root_ang_vel[env_ids, t] = base_ang_vel
            self.disc_hist_dof_pos[env_ids, t] = dof_pos
            self.disc_hist_dof_vel[env_ids, t] = dof_vel
            self.disc_hist_key_body_pos[env_ids, t] = key_body_pos
        

        # 部分重置，只更新对应环境的disc_obs
        self._update_disc_obs_for_envs(env_ids)
    
    

