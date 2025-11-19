# Motion Library for AMP
# 用于加载和采样参考动作数据

import torch
import pickle
import numpy as np
from typing import List, Tuple
from isaacgym.torch_utils import quat_mul, quat_conjugate

# 指数映射转四元数（参考MimicKit）
def exp_map_to_quat(exp_map, device=None):
    """
    将指数映射转换为四元数
    
    Args:
        exp_map: [..., 3] 指数映射 (axis * angle)，可以是numpy array或torch tensor
        device: 目标设备（如果exp_map是numpy array）
        
    Returns:
        quat: [..., 4] 四元数 (x, y, z, w)
    """
    # 转换为tensor
    if isinstance(exp_map, np.ndarray):
        exp_map = torch.from_numpy(exp_map).float()
        if device is not None:
            exp_map = exp_map.to(device)
    elif isinstance(exp_map, torch.Tensor):
        if device is not None and exp_map.device != device:
            exp_map = exp_map.to(device)
    
    # 计算角度
    angle = torch.norm(exp_map, dim=-1, keepdim=True)  # [..., 1]
    
    # 处理小角度情况
    eps = 1e-7
    angle_safe = torch.clamp(angle, min=eps)
    
    # 计算旋转轴
    axis = exp_map / angle_safe  # [..., 3]
    
    # 当角度很小时，使用默认轴
    small_angle_mask = angle.squeeze(-1) < eps
    if small_angle_mask.any():
        default_axis = torch.zeros_like(axis)
        default_axis[..., 2] = 1.0  # z轴
        axis = torch.where(small_angle_mask.unsqueeze(-1), default_axis, axis)
    
    # 转换为四元数: q = [sin(θ/2) * axis, cos(θ/2)]
    half_angle = angle / 2.0
    xyz = axis * torch.sin(half_angle)  # [..., 3]
    w = torch.cos(half_angle)  # [..., 1]
    
    quat = torch.cat([xyz, w], dim=-1)  # [..., 4]
    return quat


class MotionLib:
    """
    Motion Library for AMP training
    
    支持的数据格式:
    {
        'loop_mode': int,  # 1=循环, 0=不循环
        'fps': int,        # 帧率
        'frames': [        # 帧列表
            [q0, q1, ..., qN],  # 每帧包含关节状态
            ...
        ]
    }
    """
    
    def __init__(self, motion_file: str, device='cuda', num_dofs=27):
        """
        Args:
            motion_file: motion数据文件路径(.pkl)
            device: 设备
            num_dofs: 机器人自由度数量
        """
        self.device = device
        self.num_dofs = num_dofs
        
        # 加载motion数据
        print(f"[MotionLib] 加载motion数据: {motion_file}")
        with open(motion_file, 'rb') as f:
            data = pickle.load(f)
        
        self.loop_mode = data['loop_mode']
        self.fps = data['fps']
        self.dt = 1.0 / self.fps
        
        # 转换frames为tensor
        frames = data['frames']
        self.num_frames = len(frames)
        
        # 或简化为只包含关节数据
        self.frame_data = torch.tensor(frames, dtype=torch.float32, device=device)
        
        print(f"[MotionLib] 加载完成:")
        print(f"  - 帧数: {self.num_frames}")
        print(f"  - FPS: {self.fps}")
        print(f"  - 数据形状: {self.frame_data.shape}")
        print(f"  - 数据范围: min={self.frame_data.min().item():.4f}, max={self.frame_data.max().item():.4f}")
        print(f"  - 循环模式: {self.loop_mode}")
        print(f"  - 每帧维度: {self.frame_data.shape[1]}")
        
        # 只支持MimicKit格式: [root_pos(3), root_rot_exp_map(3), dof_pos(29)] = 35维
        frame_dim = self.frame_data.shape[1]
        assert frame_dim == 35, f"[MotionLib] 只支持MimicKit格式（35维），当前为{frame_dim}维"
        
        self.has_root = True
        self.has_velocity = False
        self.root_rot_is_exp_map = True
        self.root_pos_start, self.root_pos_end = 0, 3
        self.root_rot_start, self.root_rot_end = 3, 6  # 3维指数映射
        self.dof_pos_start, self.dof_pos_end = 6, 35
        # DOF映射：MimicKit 29个DOF -> 我们的27个DOF（跳过waist_roll和waist_pitch）
        self.dof_mapping = list(range(0, 13)) + list(range(15, 29))  # 跳过索引13,14
        print(f"[MotionLib] MimicKit格式: 29个DOF -> {num_dofs}个DOF（跳过waist_roll/pitch）")
        
        # ===== 预先计算所有帧的速度（参考MimicKit实现）=====
        # MimicKit在加载时使用有限差分预先计算速度，避免每次采样时重复计算
        print(f"[MotionLib] 开始预计算速度...")
        self._precompute_velocities()
        print(f"[MotionLib] 速度预计算完成")
    
    def sample_motions(self, num_samples: int) -> torch.Tensor:
        """
        随机采样motion IDs
        
        Args:
            num_samples: 采样数量
            
        Returns:
            motion_ids: [num_samples] 全为0（因为只有一个motion文件）
        """
        # 由于只有一个motion文件，所有motion_id都是0
        return torch.zeros(num_samples, dtype=torch.long, device=self.device)
    
    def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """
        为每个motion随机采样时间
        
        Args:
            motion_ids: [num_samples] motion IDs
            
        Returns:
            motion_times: [num_samples] 时间（秒）
        """
        num_samples = motion_ids.shape[0]
        
        if self.loop_mode:
            # 循环模式：可以采样任意时间
            max_time = (self.num_frames - 1) * self.dt
            motion_times = torch.rand(num_samples, device=self.device) * max_time
        else:
            # 非循环模式：只能采样到最后一帧之前
            max_time = (self.num_frames - 2) * self.dt
            motion_times = torch.rand(num_samples, device=self.device) * max_time
        
        return motion_times
    
    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> Tuple:
        """
        获取指定时间的motion状态（线性插值）
        
        Args:
            motion_ids: [num_samples] motion IDs
            motion_times: [num_samples] 时间（秒）
            
        Returns:
            root_pos: [num_samples, 3] 或 None
            root_rot: [num_samples, 4] 或 None
            dof_pos: [num_samples, num_dofs]
            dof_vel: [num_samples, num_dofs] 或 None
        """
        num_samples = motion_ids.shape[0]
        
        # 将时间转换为帧索引（支持小数）
        frame_indices = motion_times / self.dt
        
        # 分离整数部分和小数部分用于插值
        frame_idx0 = torch.floor(frame_indices).long()
        frame_idx1 = frame_idx0 + 1
        blend = frame_indices - frame_idx0.float()
        
        # 处理循环
        if self.loop_mode:
            frame_idx0 = frame_idx0 % self.num_frames
            frame_idx1 = frame_idx1 % self.num_frames
        else:
            frame_idx0 = torch.clamp(frame_idx0, 0, self.num_frames - 1)
            frame_idx1 = torch.clamp(frame_idx1, 0, self.num_frames - 1)
        
        # 获取两帧数据
        frame0 = self.frame_data[frame_idx0]  # [num_samples, frame_dim]
        frame1 = self.frame_data[frame_idx1]  # [num_samples, frame_dim]
        
        # 线性插值
        blend = blend.unsqueeze(-1)  # [num_samples, 1]
        frame_interp = frame0 * (1 - blend) + frame1 * blend
        
        # 辅助函数：应用DOF映射
        def apply_dof_mapping(data):
            return data[:, self.dof_mapping] if self.dof_mapping is not None else data
        
        # 提取root数据
        root_pos = None
        root_rot = None
        if self.has_root:
            root_pos = frame_interp[:, self.root_pos_start:self.root_pos_end]
            root_rot_raw = frame_interp[:, self.root_rot_start:self.root_rot_end]
            root_rot = exp_map_to_quat(root_rot_raw, device=self.device) if self.root_rot_is_exp_map else \
                      root_rot_raw / (torch.norm(root_rot_raw, dim=-1, keepdim=True) + 1e-8)
        
        # 提取DOF数据并应用映射
        dof_pos = apply_dof_mapping(frame_interp[:, self.dof_pos_start:self.dof_pos_end])[:, :self.num_dofs]
        
        # 使用预先计算的速度（参考MimicKit，在加载时已计算好所有帧的速度）
        # 对速度进行线性插值
        dof_vel0 = self._frame_dof_vel[frame_idx0]  # [num_samples, 27]
        dof_vel1 = self._frame_dof_vel[frame_idx1]  # [num_samples, 27]
        dof_vel = dof_vel0 * (1 - blend) + dof_vel1 * blend
        
        return root_pos, root_rot, dof_pos, dof_vel
    
    def _precompute_velocities(self):
        """
        预先计算所有帧的速度（参考MimicKit实现）
        
        MimicKit在加载motion数据时使用有限差分预先计算速度：
        1. root_vel: 使用位置差分
        2. root_ang_vel: 使用四元数差分
        3. dof_vel: 使用关节角度差分
        """
        # 提取所有帧的root和dof数据
        all_root_pos = self.frame_data[:, self.root_pos_start:self.root_pos_end]  # [num_frames, 3]
        all_root_rot_exp_map = self.frame_data[:, self.root_rot_start:self.root_rot_end]  # [num_frames, 3]
        all_dof_pos_full = self.frame_data[:, self.dof_pos_start:self.dof_pos_end]  # [num_frames, 29]
        
        # 应用DOF映射
        all_dof_pos = all_dof_pos_full[:, self.dof_mapping]  # [num_frames, 27]
        
        # 转换root_rot为四元数
        all_root_rot = exp_map_to_quat(all_root_rot_exp_map, device=self.device)  # [num_frames, 4]
        
        # ===== 计算root_vel（位置差分）=====
        # 参考MimicKit: root_vel[..., :-1, :] = fps * (root_pos[..., 1:, :] - root_pos[..., :-1, :])
        self._frame_root_vel = torch.zeros_like(all_root_pos)
        self._frame_root_vel[:-1] = self.fps * (all_root_pos[1:] - all_root_pos[:-1])
        self._frame_root_vel[-1] = self._frame_root_vel[-2]  # 最后一帧使用倒数第二帧的速度
        print(f"[MotionLib] Root速度预计算: shape={self._frame_root_vel.shape}, "
              f"范围=[{self._frame_root_vel.min().item():.4f}, {self._frame_root_vel.max().item():.4f}]")
        
        # ===== 计算root_ang_vel（四元数差分）=====
        # 参考MimicKit: 使用quat_diff和quat_to_exp_map
        from isaacgym.torch_utils import quat_mul, quat_conjugate
        self._frame_root_ang_vel = torch.zeros_like(all_root_pos)  # [num_frames, 3]
        
        # 计算相邻帧的四元数差分
        root_rot0 = all_root_rot[:-1]  # [num_frames-1, 4]
        root_rot1 = all_root_rot[1:]   # [num_frames-1, 4]
        root_rot0_inv = quat_conjugate(root_rot0)
        dq = quat_mul(root_rot1, root_rot0_inv)  # [num_frames-1, 4]
        
        # 将四元数转换为角速度（exp_map格式）
        # dq = [cos(theta/2), sin(theta/2) * axis]
        dq_xyz = dq[:, :3]  # [num_frames-1, 3]
        dq_w = dq[:, 3:4]   # [num_frames-1, 1]
        dq_xyz_norm = torch.norm(dq_xyz, dim=-1, keepdim=True)
        theta = 2.0 * torch.atan2(dq_xyz_norm, dq_w)
        axis = dq_xyz / (dq_xyz_norm + 1e-8)
        root_ang_vel_exp_map = axis * theta  # [num_frames-1, 3]
        
        self._frame_root_ang_vel[:-1] = self.fps * root_ang_vel_exp_map
        self._frame_root_ang_vel[-1] = self._frame_root_ang_vel[-2]  # 最后一帧使用倒数第二帧的速度
        print(f"[MotionLib] Root角速度预计算: shape={self._frame_root_ang_vel.shape}, "
              f"范围=[{self._frame_root_ang_vel.min().item():.4f}, {self._frame_root_ang_vel.max().item():.4f}]")
        
        # ===== 计算dof_vel（关节角度差分）=====
        # 参考MimicKit: dof_vel = fps * (dof_pos[1:] - dof_pos[:-1])
        self._frame_dof_vel = torch.zeros_like(all_dof_pos)
        self._frame_dof_vel[:-1] = self.fps * (all_dof_pos[1:] - all_dof_pos[:-1])
        self._frame_dof_vel[-1] = self._frame_dof_vel[-2]  # 最后一帧使用倒数第二帧的速度
        print(f"[MotionLib] DOF速度预计算: shape={self._frame_dof_vel.shape}, "
              f"范围=[{self._frame_dof_vel.min().item():.4f}, {self._frame_dof_vel.max().item():.4f}]")
        
        print(f"[MotionLib] ✅ 已预先计算所有帧的速度（{self.num_frames}帧）")
    
    def get_root_vel(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> Tuple:
        """
        获取root速度（使用预计算的速度，参考MimicKit）
        
        Returns:
            root_lin_vel: [num_samples, 3] 或 None
            root_ang_vel: [num_samples, 3] 或 None
        """
        if not self.has_root:
            return None, None
        
        num_samples = motion_ids.shape[0]
        
        # 将时间转换为帧索引（支持小数）
        frame_indices = motion_times / self.dt
        
        # 分离整数部分和小数部分用于插值
        frame_idx0 = torch.floor(frame_indices).long()
        frame_idx1 = frame_idx0 + 1
        blend = frame_indices - frame_idx0.float()
        
        # 处理循环
        if self.loop_mode:
            frame_idx0 = frame_idx0 % self.num_frames
            frame_idx1 = frame_idx1 % self.num_frames
        else:
            frame_idx0 = torch.clamp(frame_idx0, 0, self.num_frames - 1)
            frame_idx1 = torch.clamp(frame_idx1, 0, self.num_frames - 1)
        
        # 对预计算的速度进行线性插值
        blend = blend.unsqueeze(-1)  # [num_samples, 1]
        
        root_vel0 = self._frame_root_vel[frame_idx0]  # [num_samples, 3]
        root_vel1 = self._frame_root_vel[frame_idx1]  # [num_samples, 3]
        root_lin_vel = root_vel0 * (1 - blend) + root_vel1 * blend
        
        root_ang_vel0 = self._frame_root_ang_vel[frame_idx0]  # [num_samples, 3]
        root_ang_vel1 = self._frame_root_ang_vel[frame_idx1]  # [num_samples, 3]
        root_ang_vel = root_ang_vel0 * (1 - blend) + root_ang_vel1 * blend
        
        return root_lin_vel, root_ang_vel

