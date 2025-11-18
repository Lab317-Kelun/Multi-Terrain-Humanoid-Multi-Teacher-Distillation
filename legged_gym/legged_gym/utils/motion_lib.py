# Motion Library for AMP
# 用于加载和采样参考动作数据

import torch
import pickle
import numpy as np
from typing import List, Tuple
from isaacgym.torch_utils import quat_mul, quat_conjugate


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
        
        # 假设每帧格式: [root_pos(3), root_rot(4), dof_pos(27), ...]
        # 或简化为只包含关节数据
        self.frame_data = torch.tensor(frames, dtype=torch.float32, device=device)
        
        print(f"[MotionLib] 加载完成:")
        print(f"  - 帧数: {self.num_frames}")
        print(f"  - FPS: {self.fps}")
        print(f"  - 循环模式: {self.loop_mode}")
        print(f"  - 每帧维度: {self.frame_data.shape[1]}")
        
        # 解析数据维度
        frame_dim = self.frame_data.shape[1]
        
        # 尝试推断数据格式
        # 格式1: root_pos(3) + root_rot(4) + dof_pos(27) = 34
        # 格式2: root_pos(3) + root_rot(4) + dof_pos(27) + dof_vel(27) = 61
        # 格式3: 只有dof_pos(27)
        
        if frame_dim == 34 or frame_dim == 35:
            # root_pos(3) + root_rot(4) + dof_pos(27)
            self.has_root = True
            self.has_velocity = False
            self.root_pos_start = 0
            self.root_pos_end = 3
            self.root_rot_start = 3
            self.root_rot_end = 7
            self.dof_pos_start = 7
            self.dof_pos_end = 7 + num_dofs
        elif frame_dim >= 60:
            # 包含速度
            self.has_root = True
            self.has_velocity = True
            self.root_pos_start = 0
            self.root_pos_end = 3
            self.root_rot_start = 3
            self.root_rot_end = 7
            self.dof_pos_start = 7
            self.dof_pos_end = 7 + num_dofs
            self.dof_vel_start = 7 + num_dofs
            self.dof_vel_end = 7 + num_dofs + num_dofs
        elif frame_dim == num_dofs:
            # 只有关节数据
            self.has_root = False
            self.has_velocity = False
            self.dof_pos_start = 0
            self.dof_pos_end = num_dofs
        else:
            print(f"[MotionLib] Warning: 未知的帧维度 {frame_dim}，假设为 root(7) + dof({num_dofs})")
            self.has_root = True
            self.has_velocity = False
            self.root_pos_start = 0
            self.root_pos_end = 3
            self.root_rot_start = 3
            self.root_rot_end = 7
            self.dof_pos_start = 7
            self.dof_pos_end = min(7 + num_dofs, frame_dim)
        
        print(f"[MotionLib] 数据格式:")
        print(f"  - 包含root: {self.has_root}")
        print(f"  - 包含velocity: {self.has_velocity}")
    
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
        
        # 提取各部分数据
        root_pos = None
        root_rot = None
        dof_vel = None
        
        if self.has_root:
            root_pos = frame_interp[:, self.root_pos_start:self.root_pos_end]
            root_rot = frame_interp[:, self.root_rot_start:self.root_rot_end]
            # 归一化四元数
            root_rot = root_rot / (torch.norm(root_rot, dim=-1, keepdim=True) + 1e-8)
        
        dof_pos = frame_interp[:, self.dof_pos_start:self.dof_pos_end]
        
        if self.has_velocity:
            dof_vel = frame_interp[:, self.dof_vel_start:self.dof_vel_end]
        else:
            # 通过有限差分估计速度
            dof_vel = (frame1[:, self.dof_pos_start:self.dof_pos_end] - 
                      frame0[:, self.dof_pos_start:self.dof_pos_end]) / self.dt
        
        return root_pos, root_rot, dof_pos, dof_vel
    
    def get_root_vel(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> Tuple:
        """
        获取root速度（通过有限差分）
        
        完整实现：使用四元数差分计算角速度
        
        Returns:
            root_lin_vel: [num_samples, 3] 或 None
            root_ang_vel: [num_samples, 3] 或 None
        """
        if not self.has_root:
            return None, None
        
        # 小的时间步长用于差分
        dt_small = 0.01
        
        # 当前时刻
        root_pos0, root_rot0, _, _ = self.get_motion_state(motion_ids, motion_times)
        
        # 稍后时刻
        root_pos1, root_rot1, _, _ = self.get_motion_state(motion_ids, motion_times + dt_small)
        
        # ===== 线速度 =====
        root_lin_vel = (root_pos1 - root_pos0) / dt_small
        
        # ===== 角速度（使用四元数差分）=====
        # 计算相对旋转: dq = q1 * q0^-1
        q0_inv = quat_conjugate(root_rot0)
        dq = quat_mul(root_rot1, q0_inv)
        
        # 将四元数转换为角速度
        # dq = [cos(theta/2), sin(theta/2) * axis]
        # 当theta很小时: dq ≈ [1, theta/2 * axis]
        # 所以 omega = 2 * dq.xyz / dt
        
        # 提取四元数的虚部（xyz部分）
        dq_xyz = dq[:, :3]  # [num_samples, 3]
        dq_w = dq[:, 3:4]   # [num_samples, 1]
        
        # 计算角度
        # theta = 2 * atan2(||xyz||, w)
        dq_xyz_norm = torch.norm(dq_xyz, dim=-1, keepdim=True)
        theta = 2.0 * torch.atan2(dq_xyz_norm, dq_w)
        
        # 计算旋转轴（归一化的xyz）
        axis = dq_xyz / (dq_xyz_norm + 1e-8)
        
        # 角速度 = axis * theta / dt
        root_ang_vel = axis * theta / dt_small
        
        return root_lin_vel, root_ang_vel

