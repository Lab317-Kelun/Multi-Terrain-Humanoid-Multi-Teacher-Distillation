# PPO with Double Reward and AMP
# 结合双Critic (BeamDojo) 和 AMP Discriminator的PPO算法

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
from rsl_rl.storage import RolloutStorage


class PPODoubleRewardAMP(PPODoubleReward):
    """
    扩展双Critic PPO算法，添加AMP训练逻辑
    
    训练流程：
    1. 收集rollouts（包含task reward）
    2. 使用discriminator计算AMP reward
    3. 合并task reward和AMP reward
    4. 更新actor和双critic
    5. 更新discriminator
    """
    
    def __init__(self,
                 actor_critic,
                 estimator=None,
                 estimator_cfg=None,
                 depth_encoder=None,
                 depth_encoder_cfg=None,
                 depth_actor=None,
                 vec_env=None,  # 添加环境引用用于采样expert demonstrations
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
                 # 双Critic参数
                 dense_value_loss_coef=1.0,
                 sparse_value_loss_coef=1.0,
                 advantage_merge_weight=0.5,
                 dense_reward_weight=1.0,
                 sparse_reward_weight=0.25,
                 # AMP参数
                 disc_learning_rate=5e-5,
                 disc_loss_weight=5.0,
                 disc_logit_reg=0.01,
                 disc_grad_penalty=5.0,
                 disc_weight_decay=0.0001,
                 disc_reward_scale=2.0,
                 disc_eval_batch_size=4096,
                 task_reward_weight=0.5,
                 disc_reward_weight=0.5,
                 **kwargs):
        """
        Args:
            AMP相关参数：
            - disc_learning_rate: discriminator学习率
            - disc_loss_weight: discriminator损失权重
            - disc_logit_reg: logit正则化系数
            - disc_grad_penalty: 梯度惩罚系数
            - disc_weight_decay: 权重衰减系数
            - disc_reward_scale: discriminator奖励缩放
            - disc_eval_batch_size: 评估时的batch大小
            - task_reward_weight: 任务奖励权重
            - disc_reward_weight: AMP奖励权重
        """
        # 初始化父类（双Critic PPO）
        super().__init__(
            actor_critic=actor_critic,
            estimator=estimator,
            estimator_paras=estimator_cfg,
            depth_encoder=depth_encoder,
            depth_encoder_paras=depth_encoder_cfg,
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
            dense_value_loss_coef=dense_value_loss_coef,
            sparse_value_loss_coef=sparse_value_loss_coef,
            advantage_merge_weight=advantage_merge_weight,
            dense_reward_weight=dense_reward_weight,
            sparse_reward_weight=sparse_reward_weight,
            **kwargs
        )
        
        # ===== 保存环境引用 =====
        self.vec_env = vec_env
        
        # ===== AMP参数 =====
        self.disc_learning_rate = disc_learning_rate
        self.disc_loss_weight = disc_loss_weight
        self.disc_logit_reg = disc_logit_reg
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_weight_decay = disc_weight_decay
        self.disc_reward_scale = disc_reward_scale
        self.disc_eval_batch_size = disc_eval_batch_size
        self.task_reward_weight = task_reward_weight
        self.disc_reward_weight = disc_reward_weight
        
        # ===== Discriminator优化器 =====
        if hasattr(actor_critic, 'discriminator') and actor_critic.discriminator is not None:
            self.disc_optimizer = optim.Adam(
                actor_critic.discriminator.parameters(),
                lr=disc_learning_rate
            )
            print(f"[AMP Algorithm] 已创建discriminator优化器，学习率: {disc_learning_rate}")
        else:
            self.disc_optimizer = None
            print("[AMP Algorithm] Warning: discriminator未找到，AMP训练将被禁用")
        
        # ===== AMP训练统计 =====
        self.disc_loss_history = []
        self.disc_agent_acc_history = []
        self.disc_demo_acc_history = []
        
        # ===== 最新AMP指标（用于WandB记录）=====
        self.amp_metrics = {
            'disc_loss': 0.0,
            'disc_grad_penalty': 0.0,
            'disc_agent_acc': 0.0,
            'disc_demo_acc': 0.0,
            'disc_agent_logit': 0.0,
            'disc_demo_logit': 0.0,
            'task_reward_mean': 0.0,
            'amp_reward_mean': 0.0,
        }
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, disc_obs_shape=None):
        """初始化Rollout Storage，包含AMP的disc_obs"""
        from rsl_rl.storage import RolloutStorage
        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            actor_obs_shape, 
            critic_obs_shape, 
            action_shape, 
            self.device,
            disc_obs_shape=disc_obs_shape
        )
        self.enable_amp_storage = (disc_obs_shape is not None)
        print(f"[PPO AMP] Storage初始化完成，disc_obs_shape={disc_obs_shape}, AMP={self.enable_amp_storage}")
    
    def process_env_step(self, rewards, dones, infos):
        """处理环境步骤，包含AMP disc_obs的存储"""
        # 调用父类方法处理奖励
        rewards_total = super().process_env_step(rewards, dones, infos)
        
        # 存储disc_obs (如果有的话)
        if self.enable_amp_storage and 'disc_obs' in infos:
            self.transition.disc_obs = infos['disc_obs'].clone()
        
        return rewards_total
    
    def update(self):
        """
        更新策略和discriminator
        
        1. 计算AMP reward并合并到rollouts
        2. 更新actor和双critic (继承自父类)
        3. 更新discriminator
        """
        # ===== Step 1: 计算AMP reward =====
        if self.disc_optimizer is not None and self.storage.disc_obs is not None:
            disc_obs = self.storage.disc_obs
            disc_rewards = self._compute_disc_rewards(disc_obs)
            
            # 合并task reward和AMP reward
            # storage.rewards中已经包含了task rewards (dense + sparse)
            task_rewards = self.storage.rewards.clone()
            combined_rewards = (
                self.task_reward_weight * task_rewards + 
                self.disc_reward_weight * disc_rewards.unsqueeze(-1)
            )
            self.storage.rewards = combined_rewards
            
            # 更新指标
            self.amp_metrics['task_reward_mean'] = task_rewards.mean().item()
            self.amp_metrics['amp_reward_mean'] = disc_rewards.mean().item()
            
            print(f"[AMP] Task reward mean: {task_rewards.mean().item():.4f}, "
                  f"AMP reward mean: {disc_rewards.mean().item():.4f}")
        
        # ===== Step 2: 更新actor和双critic (父类方法) =====
        # 父类返回: mean_value_loss, mean_surrogate_loss, mean_estimator_loss, 
        #          mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, 
        #          priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse
        update_results = super().update()
        (mean_value_loss, mean_surrogate_loss, mean_estimator_loss, 
         mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, 
         priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse) = update_results
        
        # ===== Step 3: 更新discriminator =====
        disc_info = {}
        if self.disc_optimizer is not None and self.storage.disc_obs is not None:
            disc_info = self._update_discriminator()
            # 覆盖discriminator相关的指标（使用AMP的discriminator）
            mean_discriminator_loss = disc_info.get('disc_loss', mean_discriminator_loss)
            mean_discriminator_acc = (disc_info.get('disc_agent_acc', 0) + disc_info.get('disc_demo_acc', 0)) / 2.0
            
            # 更新AMP指标
            self.amp_metrics.update(disc_info)
        
        # 返回与父类相同格式的值，供runner使用
        return (mean_value_loss, mean_surrogate_loss, mean_estimator_loss, 
                mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, 
                priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse)
    
    def _compute_disc_rewards(self, disc_obs):
        """
        使用discriminator计算AMP reward
        
        AMP reward公式：r = -log(1 - D(s))
        其中D(s)是discriminator输出的概率（agent是真实数据的概率）
        
        Args:
            disc_obs: [num_steps, num_envs, disc_obs_size]
            
        Returns:
            disc_rewards: [num_steps, num_envs]
        """
        with torch.no_grad():
            # Flatten time and env dimensions
            num_steps, num_envs = disc_obs.shape[:2]
            disc_obs_flat = disc_obs.reshape(-1, disc_obs.shape[-1])
            
            # 分批评估（避免OOM）
            disc_logits = []
            for i in range(0, disc_obs_flat.shape[0], self.disc_eval_batch_size):
                batch = disc_obs_flat[i:i+self.disc_eval_batch_size]
                logits = self.actor_critic.eval_disc(batch)
                disc_logits.append(logits)
            
            disc_logits = torch.cat(disc_logits, dim=0)
            disc_logits = disc_logits.reshape(num_steps, num_envs)
            
            # ===== AMP reward计算 (MimicKit标准实现) =====
            # 参考: MimicKit/mimickit/learning/amp_agent.py 第199-201行
            # 公式: reward = -log(1 - D(s))
            # 其中 D(s) = sigmoid(logit) 是discriminator认为agent是expert的概率
            # 
            # 直觉：
            # - D(s)→1 (agent像expert): 1-D(s)→0, -log(0)→+∞ → 高奖励!
            # - D(s)→0 (agent不像expert): 1-D(s)→1, -log(1)→0 → 无奖励
            # 
            # 计算sigmoid概率
            prob = 1 / (1 + torch.exp(-disc_logits))
            
            # 添加数值稳定性：防止log(0)
            disc_rewards = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=prob.device)))
            
            # 应用奖励缩放
            disc_rewards = disc_rewards * self.disc_reward_scale
        
        return disc_rewards
    
    def _update_discriminator(self):
        """
        更新discriminator
        
        目标：让discriminator区分agent数据和expert数据
        - agent数据：从storage中获取
        - expert数据：从motion library采样
        
        损失函数：
        - agent loss: BCE(D(s_agent), 0)  # agent数据应该被判为fake
        - demo loss: BCE(D(s_demo), 1)    # expert数据应该被判为real
        - grad penalty: 梯度惩罚（WGAN-GP风格）
        """
        # 获取agent数据
        disc_obs_agent = self.storage.disc_obs  # [num_steps, num_envs, disc_obs_size]
        num_steps, num_envs = disc_obs_agent.shape[:2]
        disc_obs_agent = disc_obs_agent.reshape(-1, disc_obs_agent.shape[-1])
        
        # 采样expert数据（从环境的motion library）
        num_demo_samples = disc_obs_agent.shape[0]
        
        # 需要从环境获取expert demonstrations
        # 注意：这里需要PPODoubleRewardAMP在初始化时保存环境引用
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'fetch_disc_obs_demo'):
            # 从环境的motion library采样
            disc_obs_demo = self.vec_env.fetch_disc_obs_demo(num_demo_samples)
        else:
            # Fallback: 使用agent数据的随机排列（不推荐，仅用于测试）
            print("[AMP] Warning: 环境没有fetch_disc_obs_demo方法，使用随机排列作为fallback")
            perm = torch.randperm(disc_obs_agent.shape[0], device=self.device)
            disc_obs_demo = disc_obs_agent[perm].clone()
        
        disc_obs_demo.requires_grad_(True)
        
        # 前向传播
        disc_logit_agent = self.actor_critic.eval_disc(disc_obs_agent).squeeze(-1)
        disc_logit_demo = self.actor_critic.eval_disc(disc_obs_demo).squeeze(-1)
        
        # 计算BCE损失
        bce_loss = nn.BCEWithLogitsLoss()
        loss_agent = bce_loss(disc_logit_agent, torch.zeros_like(disc_logit_agent))
        loss_demo = bce_loss(disc_logit_demo, torch.ones_like(disc_logit_demo))
        disc_loss = 0.5 * (loss_agent + loss_demo)
        
        # 梯度惩罚 (WGAN-GP风格)
        disc_demo_grad = torch.autograd.grad(
            outputs=disc_logit_demo.sum(),
            inputs=disc_obs_demo,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_penalty = torch.mean(torch.sum(disc_demo_grad ** 2, dim=-1))
        disc_loss = disc_loss + self.disc_grad_penalty * grad_penalty
        
        # Logit正则化
        if self.disc_logit_reg > 0:
            logit_weights = self.actor_critic.get_disc_logit_weights()
            logit_reg_loss = torch.sum(logit_weights ** 2)
            disc_loss = disc_loss + self.disc_logit_reg * logit_reg_loss
        
        # 权重衰减
        if self.disc_weight_decay > 0:
            disc_weights = self.actor_critic.get_disc_weights()
            if len(disc_weights) > 0:
                disc_weights = torch.cat(disc_weights, dim=-1)
                weight_decay_loss = torch.sum(disc_weights ** 2)
                disc_loss = disc_loss + self.disc_weight_decay * weight_decay_loss
        
        # 反向传播和优化
        self.disc_optimizer.zero_grad()
        (self.disc_loss_weight * disc_loss).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.discriminator.parameters(), self.max_grad_norm)
        self.disc_optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            agent_acc = (disc_logit_agent < 0).float().mean()
            demo_acc = (disc_logit_demo > 0).float().mean()
        
        # 记录统计信息
        self.disc_loss_history.append(disc_loss.item())
        self.disc_agent_acc_history.append(agent_acc.item())
        self.disc_demo_acc_history.append(demo_acc.item())
        
        return {
            'disc_loss': disc_loss.item(),
            'disc_grad_penalty': grad_penalty.item(),
            'disc_agent_acc': agent_acc.item(),
            'disc_demo_acc': demo_acc.item(),
            'disc_agent_logit': disc_logit_agent.mean().item(),
            'disc_demo_logit': disc_logit_demo.mean().item()
        }

