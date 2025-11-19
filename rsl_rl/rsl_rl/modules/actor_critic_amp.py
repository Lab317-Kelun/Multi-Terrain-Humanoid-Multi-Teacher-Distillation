# Actor-Critic with AMP Discriminator
# 结合双Critic (BeamDojo) 和 AMP Discriminator

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward


class ActorCriticRMADoubleRewardAMP(ActorCriticRMADoubleReward):
    """
    扩展双Critic Actor-Critic，添加AMP Discriminator
    
    结构：
    - Actor: 策略网络
    - Dense Critic: 密集奖励价值函数
    - Sparse Critic: 稀疏奖励价值函数  
    - Discriminator: AMP风格判别器
    """
    
    def __init__(self, 
                 num_prop,
                 num_scan,
                 num_critic_obs,
                 num_priv_latent,
                 num_priv_explicit,
                 num_hist,
                 num_actions,
                 disc_obs_size=None,
                 scan_encoder_dims=[256, 256, 256],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 disc_hidden_dims=[1024, 512],
                 activation='elu',
                 init_noise_std=1.0,
                 use_double_critic=False,
                 **kwargs):
        """
        Args:
            disc_obs_size: discriminator观测维度
            disc_hidden_dims: discriminator隐藏层维度
            其他参数与ActorCriticRMADoubleReward相同
        """
        # 初始化父类（包含actor和双critic）
        super().__init__(
            num_prop=num_prop,
            num_scan=num_scan,
            num_critic_obs=num_critic_obs,
            num_priv_latent=num_priv_latent,
            num_priv_explicit=num_priv_explicit,
            num_hist=num_hist,
            num_actions=num_actions,
            scan_encoder_dims=scan_encoder_dims,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            use_double_critic=use_double_critic,
            **kwargs
        )
        
        # ===== 构建Discriminator =====
        self.disc_obs_size = disc_obs_size
        if disc_obs_size is not None:
            self.discriminator = self._build_discriminator(
                disc_obs_size, disc_hidden_dims, activation
            )
            print(f"[AMP Model] Discriminator输入维度: {disc_obs_size}")
            print(f"[AMP Model] Discriminator隐藏层: {disc_hidden_dims}")
        else:
            self.discriminator = None
            print("[AMP Model] Warning: disc_obs_size未设置，discriminator未构建")
    
    def _build_discriminator(self, input_size, hidden_dims, activation):
        """
        构建discriminator网络（参考MimicKit实现）
        
        关键差异：
        1. 输出层使用uniform初始化（而非orthogonal）
        2. 隐藏层可以使用orthogonal初始化
        """
        activation_fn = self._get_activation_fn(activation)
        
        layers = []
        last_size = input_size
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_size, hidden_dim))
            layers.append(activation_fn())
            last_size = hidden_dim
        
        # 输出层 (logit，单个值)
        output_layer = nn.Linear(last_size, 1)
        layers.append(output_layer)
        
        # 初始化权重（参考MimicKit）
        disc_net = nn.Sequential(*layers)
        init_output_scale = 1.0  # 与MimicKit保持一致
        
        for m in disc_net.modules():
            if isinstance(m, nn.Linear):
                # 隐藏层：使用orthogonal初始化
                if m is not output_layer:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    # 输出层：使用uniform初始化（MimicKit标准）
                    nn.init.uniform_(m.weight, -init_output_scale, init_output_scale)
                    nn.init.zeros_(m.bias)
        
        return disc_net
    
    def _get_activation_fn(self, activation):
        """获取激活函数"""
        if activation == 'elu':
            return nn.ELU
        elif activation == 'relu':
            return nn.ReLU
        elif activation == 'tanh':
            return nn.Tanh
        else:
            return nn.ELU
    
    def eval_disc(self, disc_obs):
        """
        评估discriminator
        
        Args:
            disc_obs: [batch_size, disc_obs_size]
            
        Returns:
            logits: [batch_size, 1] - discriminator输出logit
        """
        if self.discriminator is None:
            raise RuntimeError("Discriminator未构建，请检查disc_obs_size配置")
        
        return self.discriminator(disc_obs)
    
    def get_disc_logit_weights(self):
        """获取discriminator输出层权重（用于正则化）"""
        if self.discriminator is None:
            return torch.tensor([])
        
        # 获取最后一层（输出层）的权重
        last_layer = list(self.discriminator.modules())[-1]
        if isinstance(last_layer, nn.Linear):
            return torch.flatten(last_layer.weight)
        return torch.tensor([])
    
    def get_disc_weights(self):
        """获取discriminator所有权重（用于权重衰减）"""
        if self.discriminator is None:
            return []
        
        weights = []
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        
        return weights

