import time
import os
import json
from collections import deque
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.env import *
from rsl_rl.modules import *
from rsl_rl.algorithms import *
from rsl_rl.algorithms.mirror import Mirror
from rsl_rl.storage import *

from .on_policy_runner import OnPolicyRunner


class OnPolicyRunnerMirror(OnPolicyRunner):
    """镜像版本的OnPolicyRunner，继承原始功能并添加镜像数据增强"""

    def __init__(
            self,
            env: VecEnv,
            train_cfg,
            log_dir=None,
            init_wandb=True,
            device="cpu", **kwargs
    ):
        """初始化镜像Runner，添加镜像功能到原始PPO训练"""
        
        # 直接调用父类初始化，但我们需要额外的镜像参数
        super().__init__(env, train_cfg, log_dir, init_wandb, device, **kwargs)

    def _init_mirror_components(self, train_cfg):
        """初始化镜像相关组件"""
        # 创建镜像对象
        self.mirror = Mirror(self.env)
        
        # 如果算法类支持镜像，传入镜像对象
        if hasattr(self.alg, 'mirror'):
            self.alg.mirror = self.mirror

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        """继承原始learn_RL方法，只需添加镜像初始化"""
        # 初始化镜像组件
        self._init_mirror_components(self.cfg)
        
        # 调用父类的learn_RL方法
        return super().learn_RL(num_learning_iterations, init_at_random_ep_len)
        
    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        """继承视觉学习方法"""
        # 继承父类的learn_vision方法
        return super().learn_vision(num_learning_iterations, init_at_random_ep_len)

    def save(self, path, infos=None):
        """保存模型，包括镜像相关的状态"""
        # 调用父类的保存方法
        super().save(path, infos)
        
        # 如果有镜像对象，可以在这里保存额外的镜像状态
        # (目前镜像是轻量级对象，不需要特殊保存)

    def log(self, locs, width=80, pad=35):
        """日志记录，添加镜像损失"""
        # 调用父类日志记录
        super().log(locs, width, pad)
        
        # 添加镜像损失记录（如果存在）
        if hasattr(self.alg, 'mean_mirror_loss') and self.writer is not None:
            self.writer.add_scalar('Loss/mirror', self.alg.mean_mirror_loss, locs['it'])