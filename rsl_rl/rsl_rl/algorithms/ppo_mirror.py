import torch.nn as nn

from rsl_rl.modules import *
from rsl_rl.storage import *
from rsl_rl.algorithms import *

from .ppo import PPO
from .mirror import Mirror


class PPOMirror(PPO):
    """
    PPO算法类，添加镜像数据增强功能
    """

    def __init__(
            self,
            actor_critic,
            estimator,
            estimator_paras,
            depth_encoder,
            depth_encoder_paras,
            depth_actor,
            num_learning_epochs=1,
            num_mini_batches=1,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=0.01,
            device='cpu',
            dagger_update_freq=20,
            priv_reg_coef_schedual=[0, 0, 0],
            # Mirror参数
            mirror=None,
            mirror_coef=1.0,
            **kwargs
    ):
        # 调用父类初始化
        super().__init__(
            actor_critic=actor_critic,
            estimator=estimator,
            estimator_paras=estimator_paras,
            depth_encoder=depth_encoder,
            depth_encoder_paras=depth_encoder_paras,
            depth_actor=depth_actor,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            dagger_update_freq=dagger_update_freq,
            priv_reg_coef_schedual=priv_reg_coef_schedual,
            **kwargs
        )

        # 初始化镜像组件
        self.mirror = mirror
        self.mirror_coef = mirror_coef
        self.mean_mirror_loss = 0.0
        print(f"Mirror coef: {self.mirror_coef}")

    def update_mirror(self):
        """更新镜像损失"""
        if not self.mirror or self.mirror_coef <= 0:
            return 0.0
            
        mean_mirror_loss = 0.0
        num_updates = 0
        
        # 使用存储的数据计算镜像损失
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            
            # 计算原始动作
            with torch.no_grad():
                origin_actions = self.actor_critic.act_inference(obs_batch, hist_encoding=True)
            
            # 获取镜像观测和动作
            mirror_obs_batch = self.mirror.get_mirror_obs_batch(obs_batch)
            with torch.no_grad():
                mirror_actions = self.actor_critic.act_inference(mirror_obs_batch, hist_encoding=True)
            target_actions = self.mirror.get_mirror_action_batch(mirror_actions)
            
            # 计算镜像损失
            mirror_loss = nn.MSELoss()(origin_actions, target_actions) * self.mirror_coef
            mean_mirror_loss += mirror_loss.item()
            num_updates += 1
            
        if num_updates > 0:
            mean_mirror_loss /= num_updates
            self.mean_mirror_loss = mean_mirror_loss
            
        return mean_mirror_loss