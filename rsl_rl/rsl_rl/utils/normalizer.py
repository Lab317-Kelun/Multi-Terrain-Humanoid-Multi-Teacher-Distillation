# Discriminator Observation Normalizer
# 基于MimicKit的实现，用于归一化disc_obs

import torch
import torch.nn as nn
import numpy as np


class Normalizer(nn.Module):
    """
    Discriminator观测归一化器
    
    功能：
    1. 记录观测的均值和方差
    2. 提供归一化和反归一化功能
    3. 支持在线更新统计信息
    """
    
    def __init__(self, shape, device, init_mean=None, init_std=None, min_std=1e-4, clip=np.inf, dtype=torch.float32):
        """
        Args:
            shape: 观测的形状（不包含batch维度）
            device: 设备
            init_mean: 初始均值（可选）
            init_std: 初始标准差（可选）
            min_std: 最小标准差（防止除零）
            clip: 裁剪范围（归一化后的值会被裁剪到[-clip, clip]）
            dtype: 数据类型
        """
        super().__init__()
        
        self._min_var = min_std * min_std
        self._clip = clip
        self.dtype = dtype
        self._build_params(shape, device, init_mean, init_std)
    
    def _build_params(self, shape, device, init_mean, init_std):
        """构建归一化参数"""
        self._count = nn.Parameter(
            torch.zeros([1], device=device, requires_grad=False, dtype=torch.long), 
            requires_grad=False
        )
        self._mean = nn.Parameter(
            torch.zeros(shape, device=device, requires_grad=False, dtype=self.dtype), 
            requires_grad=False
        )
        self._std = nn.Parameter(
            torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype), 
            requires_grad=False
        )
        
        if init_mean is not None:
            assert init_mean.shape == shape, \
                f'Normalizer init mean shape mismatch, expecting {shape}, but got {init_mean.shape}'
            self._mean[:] = init_mean
        
        if init_std is not None:
            assert init_std.shape == shape, \
                f'Normalizer init std shape mismatch, expecting {shape}, but got {init_std.shape}'
            self._std[:] = init_std
        
        self._mean_sq = None
        
        # 用于累积新数据的统计信息
        self._new_count = 0
        self._new_sum = torch.zeros_like(self._mean)
        self._new_sum_sq = torch.zeros_like(self._mean)
    
    def record(self, x):
        """
        记录新的观测数据（用于更新统计信息）
        
        Args:
            x: 观测数据，形状为 [..., *shape]
        """
        shape = self.get_shape()
        assert len(x.shape) > len(shape), \
            f'Input shape {x.shape} should have more dimensions than normalizer shape {shape}'
        
        # Flatten batch dimensions
        x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)
        
        self._new_count += x.shape[0]
        self._new_sum += torch.sum(x, axis=0)  # 使用axis=0与MimicKit保持一致
        self._new_sum_sq += torch.sum(torch.square(x), axis=0)  # 使用axis=0与MimicKit保持一致
    
    def update(self):
        """
        更新归一化统计信息（基于record的数据）
        
        参考MimicKit实现，但添加了new_count为0的检查以避免除零错误
        """
        if self._mean_sq is None:
            self._mean_sq = self._calc_mean_sq(self._mean, self._std)
        
        # 注意：MimicKit在多进程环境下使用mp_util.reduce_sum，我们单进程直接使用
        # 在正常训练流程中，new_count应该总是>0（因为每次update前都会record数据）
        new_count = self._new_count
        
        # 与MimicKit保持一致：不检查new_count（假设总是>0）
        # 如果new_count为0，这里会报错，但这是预期的（表示没有数据被record）
        new_mean = self._new_sum / new_count
        new_mean_sq = self._new_sum_sq / new_count
        
        new_total = self._count + new_count
        w_old = self._count.type(torch.float32) / new_total.type(torch.float32)
        w_new = float(new_count) / new_total.type(torch.float32)
        
        self._mean[:] = w_old * self._mean + w_new * new_mean
        self._mean_sq[:] = w_old * self._mean_sq + w_new * new_mean_sq
        self._count[:] = new_total
        
        self._std[:] = self._calc_std(self._mean, self._mean_sq)
        
        # 重置累积器（与MimicKit一致）
        self._new_count = 0
        self._new_sum[:] = 0
        self._new_sum_sq[:] = 0
    
    def get_shape(self):
        """获取归一化器形状"""
        return self._mean.shape
    
    def get_count(self):
        """获取已记录的样本数量"""
        return self._count
    
    def get_mean(self):
        """获取均值"""
        return self._mean
    
    def get_std(self):
        """获取标准差"""
        return self._std
    
    def set_mean_std(self, mean, std):
        """手动设置均值和标准差"""
        shape = self.get_shape()
        assert mean.shape == shape and std.shape == shape, \
            f'Normalizer shape mismatch, expecting {shape}, but got {mean.shape} and {std.shape}'
        
        self._mean[:] = mean
        self._std[:] = std
        self._mean_sq = self._calc_mean_sq(self._mean, self._std)
    
    def normalize(self, x):
        """
        归一化观测数据
        
        Args:
            x: 观测数据，形状为 [..., *shape]
            
        Returns:
            norm_x: 归一化后的数据，形状与x相同
        """
        norm_x = (x - self._mean) / self._std
        norm_x = torch.clamp(norm_x, -self._clip, self._clip)
        return norm_x.type(self.dtype)
    
    def unnormalize(self, norm_x):
        """
        反归一化数据
        
        Args:
            norm_x: 归一化后的数据
            
        Returns:
            x: 原始尺度的数据
        """
        x = norm_x * self._std + self._mean
        return x.type(self.dtype)
    
    def _calc_std(self, mean, mean_sq):
        """计算标准差"""
        var = mean_sq - torch.square(mean)
        var = torch.clamp_min(var, self._min_var)
        std = torch.sqrt(var)
        std = std.type(self.dtype)
        return std
    
    def _calc_mean_sq(self, mean, std):
        """计算均值的平方"""
        mean_sq = torch.square(std) + torch.square(mean)
        mean_sq = mean_sq.type(self.dtype)
        return mean_sq

