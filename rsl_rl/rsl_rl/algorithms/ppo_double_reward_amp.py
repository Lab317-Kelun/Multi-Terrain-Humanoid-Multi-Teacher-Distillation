# PPO with Double Reward and AMP
# 结合双Critic (BeamDojo) 和 AMP Discriminator的PPO算法

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils.normalizer import Normalizer


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
                 debug_amp=False,  # 调试标志
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
        
        # ===== 归一化器更新控制（参考MimicKit）=====
        # 只在样本数小于normalizer_samples时更新归一化器
        self.normalizer_samples = 100000000  # 默认值，与MimicKit一致
        self._sample_count = 0  # 用于跟踪样本数量
        
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
        
        # ===== Discriminator观测归一化器（关键！MimicKit的核心特性）=====
        # 注意：normalizer会在init_storage时初始化，因为需要知道disc_obs_shape
        self.disc_obs_norm = None
        
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
        
        # 调试标志（从参数中获取）
        self._debug_amp = debug_amp
    
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
        
        # ===== 初始化Discriminator观测归一化器 =====
        if self.enable_amp_storage and disc_obs_shape is not None:
            # disc_obs_shape是一个列表，例如 [disc_obs_size]
            disc_obs_size = disc_obs_shape[0] if isinstance(disc_obs_shape, list) else disc_obs_shape
            self.disc_obs_norm = Normalizer(
                shape=(disc_obs_size,),
                device=self.device,
                min_std=1e-4,
                clip=np.inf,  # 与MimicKit默认值保持一致（可以后续在配置中调整）
                dtype=torch.float32
            )
            print(f"[PPO AMP] Discriminator观测归一化器已初始化，形状: ({disc_obs_size},)")
        else:
            self.disc_obs_norm = None
        
        print(f"[PPO AMP] Storage初始化完成，disc_obs_shape={disc_obs_shape}, AMP={self.enable_amp_storage}")
    
    def process_env_step(self, rewards, dones, infos):
        """处理环境步骤，包含AMP disc_obs的存储和归一化器更新"""
        # 调用父类方法处理奖励
        rewards_total = super().process_env_step(rewards, dones, infos)
        
        if self.enable_amp_storage and 'disc_obs' in infos:
            disc_obs = infos['disc_obs'].clone()
            self.transition.disc_obs = disc_obs
            
            # 更新归一化器统计信息（仅在需要时，参考MimicKit）
            # MimicKit: 只在_need_normalizer_update()为True时记录
            if self.disc_obs_norm is not None and self._need_normalizer_update():
                self.disc_obs_norm.record(disc_obs)
        
        return rewards_total
    
    def _need_normalizer_update(self):
        """检查是否需要更新归一化器（参考MimicKit）"""
        return self._sample_count < self.normalizer_samples
    
    def compute_returns(self, last_critic_obs):
        """
        重写compute_returns，在计算returns之前先合并AMP reward
        
        这是关键：必须在compute_returns之前合并reward，因为returns的计算依赖于rewards
        
        对于双Critic：
        - rewards_dense: 密集奖励（任务相关）
        - rewards_sparse: 稀疏奖励（目标达成等）
        - AMP reward应该合并到rewards_dense中（因为它是风格奖励，属于密集奖励）
        """
        # ===== Step 1: 计算并合并AMP reward =====
        if self.disc_optimizer is not None and self.storage.disc_obs is not None:
            disc_obs = self.storage.disc_obs
            disc_rewards = self._compute_disc_rewards(disc_obs)
            
            if self.use_double_critic:
                # 双Critic模式：合并到rewards_dense
                # rewards_dense中已经包含了密集任务奖励
                task_rewards_dense = self.storage.rewards_dense.clone()
                combined_rewards_dense = (
                    self.task_reward_weight * task_rewards_dense + 
                    self.disc_reward_weight * disc_rewards.unsqueeze(-1)
                )
                self.storage.rewards_dense = combined_rewards_dense
                
                # 更新总的rewards（用于统计）
                task_rewards_total = task_rewards_dense + self.storage.rewards_sparse
                self.storage.rewards = combined_rewards_dense + self.storage.rewards_sparse
                
                # 更新指标
                self.amp_metrics['task_reward_mean'] = task_rewards_total.mean().item()
                self.amp_metrics['amp_reward_mean'] = disc_rewards.mean().item()
                
                # ===== 调试信息：奖励合并（每N次迭代打印一次）=====
                if hasattr(self, '_debug_amp') and self._debug_amp:
                    # 初始化计数器（如果不存在）
                    if not hasattr(self, '_compute_returns_call_count'):
                        self._compute_returns_call_count = 0
                    self._compute_returns_call_count += 1
                    
                    # 每10次调用打印一次（避免日志过多）
                    if self._compute_returns_call_count % 10 == 1:
                        print(f"[AMP DEBUG] 奖励合并 (call={self._compute_returns_call_count}):")
                        print(f"  - Task reward (dense+sparse): mean={task_rewards_total.mean().item():.4f}, "
                              f"min={task_rewards_total.min().item():.4f}, max={task_rewards_total.max().item():.4f}")
                        print(f"  - AMP reward: mean={disc_rewards.mean().item():.4f}, "
                              f"min={disc_rewards.min().item():.4f}, max={disc_rewards.max().item():.4f}")
                        print(f"  - 权重: task={self.task_reward_weight}, amp={self.disc_reward_weight}")
                        print(f"  - 合并后 (dense): mean={combined_rewards_dense.mean().item():.4f}, "
                              f"min={combined_rewards_dense.min().item():.4f}, max={combined_rewards_dense.max().item():.4f}")
            else:
                # 单Critic模式：合并到rewards
                task_rewards = self.storage.rewards.clone()
                combined_rewards = (
                    self.task_reward_weight * task_rewards + 
                    self.disc_reward_weight * disc_rewards.unsqueeze(-1)
                )
                self.storage.rewards = combined_rewards
                
                # 更新指标
                self.amp_metrics['task_reward_mean'] = task_rewards.mean().item()
                self.amp_metrics['amp_reward_mean'] = disc_rewards.mean().item()
                
                # ===== 调试信息：奖励合并（单Critic模式，每10次调用打印一次）=====
                if hasattr(self, '_debug_amp') and self._debug_amp:
                    if not hasattr(self, '_compute_returns_call_count'):
                        self._compute_returns_call_count = 0
                    self._compute_returns_call_count += 1
                    
                    if self._compute_returns_call_count % 10 == 1:
                        print(f"[AMP DEBUG] 奖励合并 (单Critic模式, call={self._compute_returns_call_count}):")
                        print(f"  - Task reward: mean={task_rewards.mean().item():.4f}, "
                              f"min={task_rewards.min().item():.4f}, max={task_rewards.max().item():.4f}")
                        print(f"  - AMP reward: mean={disc_rewards.mean().item():.4f}, "
                              f"min={disc_rewards.min().item():.4f}, max={disc_rewards.max().item():.4f}")
                        print(f"  - 权重: task={self.task_reward_weight}, amp={self.disc_reward_weight}")
        
        # ===== Step 2: 调用父类的compute_returns，使用合并后的rewards =====
        super().compute_returns(last_critic_obs)
    
    def update(self):
        """
        更新策略和discriminator（参考MimicKit流程）
        
        1. 更新actor和双critic (继承自父类)
        2. 更新discriminator
        3. 更新归一化器（如果需要）
        """
        # ===== Step 1: 更新actor和双critic (父类方法) =====
        # 父类返回: mean_value_loss, mean_surrogate_loss, mean_estimator_loss, 
        #          mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, 
        #          priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse
        update_results = super().update()
        (mean_value_loss, mean_surrogate_loss, mean_estimator_loss, 
         mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, 
         priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse) = update_results
        
        # ===== Step 2: 更新discriminator =====
        disc_info = {}
        if self.disc_optimizer is not None and self.storage.disc_obs is not None:
            try:
                disc_info = self._update_discriminator()
                # 覆盖discriminator相关的指标（使用AMP的discriminator）
                if disc_info:
                    mean_discriminator_loss = disc_info.get('disc_loss', 0.0)
                    mean_discriminator_acc = (disc_info.get('disc_agent_acc', 0.0) + disc_info.get('disc_demo_acc', 0.0)) / 2.0
                    # 更新AMP指标
                    self.amp_metrics.update(disc_info)
            except Exception as e:
                print(f"[AMP ERROR] Discriminator更新失败: {e}")
                import traceback
                traceback.print_exc()
                disc_info = {}
        
        # ===== Step 3: 更新归一化器（参考MimicKit：在_update_model之后）=====
        if self._need_normalizer_update() and self.disc_obs_norm is not None:
            old_count = self.disc_obs_norm.get_count().item()
            self.disc_obs_norm.update()
            new_count = self.disc_obs_norm.get_count().item()
            
            # ===== 调试信息：归一化器更新 =====
            if hasattr(self, '_debug_amp') and self._debug_amp:
                print(f"[AMP DEBUG] Normalizer更新: count {old_count} -> {new_count}, "
                      f"新增样本: {new_count - old_count}")
                print(f"[AMP DEBUG] Normalizer统计更新后: "
                      f"mean_range=[{self.disc_obs_norm.get_mean().min().item():.4f}, "
                      f"{self.disc_obs_norm.get_mean().max().item():.4f}], "
                      f"std_range=[{self.disc_obs_norm.get_std().min().item():.4f}, "
                      f"{self.disc_obs_norm.get_std().max().item():.4f}]")
        
        # 更新样本计数（用于判断是否需要继续更新归一化器）
        if self.storage.disc_obs is not None:
            num_steps, num_envs = self.storage.disc_obs.shape[:2]
            self._sample_count += num_steps * num_envs
        
        # 返回与父类相同格式的值，供runner使用
        return (mean_value_loss, mean_surrogate_loss, mean_estimator_loss, 
                mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, 
                priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse)
    
    def _compute_disc_rewards(self, disc_obs):
        """
        使用discriminator计算AMP reward（使用归一化的disc_obs）
        
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
            
            # ===== 关键：归一化disc_obs（MimicKit标准实现）=====
            if self.disc_obs_norm is not None:
                # ===== 调试信息：归一化前 =====
                if hasattr(self, '_debug_amp') and self._debug_amp:
                    print(f"[AMP DEBUG] 归一化前 disc_obs: shape={disc_obs_flat.shape}, "
                          f"min={disc_obs_flat.min().item():.4f}, max={disc_obs_flat.max().item():.4f}, "
                          f"mean={disc_obs_flat.mean().item():.4f}, std={disc_obs_flat.std().item():.4f}")
                    print(f"[AMP DEBUG] Normalizer统计: mean.shape={self.disc_obs_norm.get_mean().shape}, "
                          f"std.shape={self.disc_obs_norm.get_std().shape}, "
                          f"count={self.disc_obs_norm.get_count().item()}")
                
                norm_disc_obs_flat = self.disc_obs_norm.normalize(disc_obs_flat)
                
                # ===== 调试信息：归一化后 =====
                if hasattr(self, '_debug_amp') and self._debug_amp:
                    print(f"[AMP DEBUG] 归一化后 disc_obs: shape={norm_disc_obs_flat.shape}, "
                          f"min={norm_disc_obs_flat.min().item():.4f}, max={norm_disc_obs_flat.max().item():.4f}, "
                          f"mean={norm_disc_obs_flat.mean().item():.4f}, std={norm_disc_obs_flat.std().item():.4f}")
            else:
                norm_disc_obs_flat = disc_obs_flat
            
            # 分批评估（避免OOM，参考MimicKit的eval_minibatch）
            disc_logits = []
            batch_size = self.disc_eval_batch_size if self.disc_eval_batch_size > 0 else norm_disc_obs_flat.shape[0]
            
            for i in range(0, norm_disc_obs_flat.shape[0], batch_size):
                batch = norm_disc_obs_flat[i:i+batch_size]
                logits = self.actor_critic.eval_disc(batch)
                disc_logits.append(logits)
            
            disc_logits = torch.cat(disc_logits, dim=0)
            disc_logits = disc_logits.squeeze(-1)  # [num_steps * num_envs]
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
            
            # ===== 调试信息：AMP奖励 =====
            if hasattr(self, '_debug_amp') and self._debug_amp:
                print(f"[AMP DEBUG] Discriminator输出: logits min={disc_logits.min().item():.4f}, "
                      f"max={disc_logits.max().item():.4f}, mean={disc_logits.mean().item():.4f}")
                print(f"[AMP DEBUG] AMP奖励: min={disc_rewards.min().item():.4f}, "
                      f"max={disc_rewards.max().item():.4f}, mean={disc_rewards.mean().item():.4f}, "
                      f"scale={self.disc_reward_scale}")
        
        return disc_rewards
    
    def _update_discriminator(self):
        """
        更新discriminator（使用归一化的disc_obs，参考MimicKit实现）
        
        目标：让discriminator区分agent数据和expert数据
        - agent数据：从storage中获取
        - expert数据：从motion library采样
        
        损失函数：
        - agent loss: BCE(D(s_agent), 0)  # agent数据应该被判为fake
        - demo loss: BCE(D(s_demo), 1)    # expert数据应该被判为real
        - grad penalty: 梯度惩罚（WGAN-GP风格）
        """
        # ===== Step 1: 获取agent数据并归一化 =====
        disc_obs_agent = self.storage.disc_obs  # [num_steps, num_envs, disc_obs_size]
        num_steps, num_envs = disc_obs_agent.shape[:2]
        disc_obs_agent = disc_obs_agent.reshape(-1, disc_obs_agent.shape[-1])
        
        # ===== Step 2: 采样expert数据（从环境的motion library）=====
        num_demo_samples = disc_obs_agent.shape[0]
        
        if not (hasattr(self, 'vec_env') and hasattr(self.vec_env, 'fetch_disc_obs_demo')):
            raise RuntimeError("环境没有fetch_disc_obs_demo方法，无法采样expert数据")
        
        # 从环境的motion library采样
        disc_obs_demo = self.vec_env.fetch_disc_obs_demo(num_demo_samples)
        
        # 确保demo数据是tensor格式
        if not isinstance(disc_obs_demo, torch.Tensor):
            disc_obs_demo = torch.tensor(disc_obs_demo, device=disc_obs_agent.device, dtype=disc_obs_agent.dtype)
        
        # 确保维度匹配
        if disc_obs_demo.shape != disc_obs_agent.shape:
            raise RuntimeError(f"Expert数据维度不匹配: agent={disc_obs_agent.shape}, demo={disc_obs_demo.shape}")
        
        # ===== Step 3: 更新归一化器统计信息（在归一化之前）=====
        # 关键：MimicKit在_record_disc_demo_data()中直接record(disc_obs_demo)，不检查_need_normalizer_update()
        # 参考: MimicKit/mimickit/learning/amp_agent.py 第87行
        # 但agent数据已经在process_env_step中记录了（只在_need_normalizer_update()为True时记录）
        if self.disc_obs_norm is not None:
            # 直接记录demo数据，不检查_need_normalizer_update()（与MimicKit一致）
            self.disc_obs_norm.record(disc_obs_demo)
        
        # ===== Step 4: 归一化观测（关键！MimicKit的核心特性）=====
        if self.disc_obs_norm is not None:
            # ===== 调试信息：归一化前 =====
            if hasattr(self, '_debug_amp') and self._debug_amp:
                print(f"[AMP DEBUG] Discriminator更新 - 归一化前:")
                print(f"  - Agent obs: shape={disc_obs_agent.shape}, "
                      f"min={disc_obs_agent.min().item():.4f}, max={disc_obs_agent.max().item():.4f}, "
                      f"mean={disc_obs_agent.mean().item():.4f}, std={disc_obs_agent.std().item():.4f}")
                print(f"  - Demo obs: shape={disc_obs_demo.shape}, "
                      f"min={disc_obs_demo.min().item():.4f}, max={disc_obs_demo.max().item():.4f}, "
                      f"mean={disc_obs_demo.mean().item():.4f}, std={disc_obs_demo.std().item():.4f}")
                print(f"  - Normalizer: count={self.disc_obs_norm.get_count().item()}, "
                      f"mean_range=[{self.disc_obs_norm.get_mean().min().item():.4f}, "
                      f"{self.disc_obs_norm.get_mean().max().item():.4f}], "
                      f"std_range=[{self.disc_obs_norm.get_std().min().item():.4f}, "
                      f"{self.disc_obs_norm.get_std().max().item():.4f}]")
            
            norm_disc_obs_agent = self.disc_obs_norm.normalize(disc_obs_agent)
            norm_disc_obs_demo = self.disc_obs_norm.normalize(disc_obs_demo)
            
            # ===== 调试信息：归一化后 =====
            if hasattr(self, '_debug_amp') and self._debug_amp:
                print(f"[AMP DEBUG] Discriminator更新 - 归一化后:")
                print(f"  - Agent obs: shape={norm_disc_obs_agent.shape}, "
                      f"min={norm_disc_obs_agent.min().item():.4f}, max={norm_disc_obs_agent.max().item():.4f}, "
                      f"mean={norm_disc_obs_agent.mean().item():.4f}, std={norm_disc_obs_agent.std().item():.4f}")
                print(f"  - Demo obs: shape={norm_disc_obs_demo.shape}, "
                      f"min={norm_disc_obs_demo.min().item():.4f}, max={norm_disc_obs_demo.max().item():.4f}, "
                      f"mean={norm_disc_obs_demo.mean().item():.4f}, std={norm_disc_obs_demo.std().item():.4f}")
        else:
            norm_disc_obs_agent = disc_obs_agent
            norm_disc_obs_demo = disc_obs_demo
        
        # 需要梯度用于梯度惩罚
        norm_disc_obs_demo.requires_grad_(True)
        
        # ===== Step 5: 前向传播 =====
        disc_logit_agent = self.actor_critic.eval_disc(norm_disc_obs_agent).squeeze(-1)
        disc_logit_demo = self.actor_critic.eval_disc(norm_disc_obs_demo).squeeze(-1)
        
        # ===== Step 6: 计算BCE损失（参考MimicKit）=====
        bce_loss = nn.BCEWithLogitsLoss()
        loss_agent = bce_loss(disc_logit_agent, torch.zeros_like(disc_logit_agent))
        loss_demo = bce_loss(disc_logit_demo, torch.ones_like(disc_logit_demo))
        disc_loss = 0.5 * (loss_agent + loss_demo)
        
        # ===== Step 7: 梯度惩罚 (WGAN-GP风格，参考MimicKit) =====
        # 参考: MimicKit/mimickit/learning/amp_agent.py 第139-143行
        disc_demo_grad = torch.autograd.grad(
            outputs=disc_logit_demo,
            inputs=norm_disc_obs_demo,
            grad_outputs=torch.ones_like(disc_logit_demo),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_penalty = torch.mean(torch.sum(disc_demo_grad ** 2, dim=-1))
        disc_loss = disc_loss + self.disc_grad_penalty * grad_penalty
        
        # ===== Step 8: Logit正则化 =====
        if self.disc_logit_reg > 0:
            logit_weights = self.actor_critic.get_disc_logit_weights()
            logit_reg_loss = torch.sum(logit_weights ** 2)
            disc_loss = disc_loss + self.disc_logit_reg * logit_reg_loss
        
        # ===== Step 9: 权重衰减 =====
        if self.disc_weight_decay > 0:
            disc_weights = self.actor_critic.get_disc_weights()
            if len(disc_weights) > 0:
                disc_weights = torch.cat(disc_weights, dim=-1)
                weight_decay_loss = torch.sum(disc_weights ** 2)
                disc_loss = disc_loss + self.disc_weight_decay * weight_decay_loss
        
        # ===== Step 10: 反向传播和优化 =====
        self.disc_optimizer.zero_grad()
        (self.disc_loss_weight * disc_loss).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.discriminator.parameters(), self.max_grad_norm)
        self.disc_optimizer.step()
        
        # ===== Step 11: 计算准确率 =====
        with torch.no_grad():
            agent_acc = (disc_logit_agent < 0).float().mean()
            demo_acc = (disc_logit_demo > 0).float().mean()
        
        # ===== 调试信息：Discriminator更新结果 =====
        if hasattr(self, '_debug_amp') and self._debug_amp:
            # 计算sigmoid概率（用于理解discriminator的输出）
            with torch.no_grad():
                agent_prob = torch.sigmoid(disc_logit_agent)
                demo_prob = torch.sigmoid(disc_logit_demo)
            
            print(f"[AMP DEBUG] Discriminator更新完成:")
            print(f"  - Agent logits: min={disc_logit_agent.min().item():.4f}, "
                  f"max={disc_logit_agent.max().item():.4f}, mean={disc_logit_agent.mean().item():.4f}")
            print(f"  - Agent prob (sigmoid): min={agent_prob.min().item():.4f}, "
                  f"max={agent_prob.max().item():.4f}, mean={agent_prob.mean().item():.4f} "
                  f"(应该接近0，表示agent不像expert)")
            print(f"  - Demo logits: min={disc_logit_demo.min().item():.4f}, "
                  f"max={disc_logit_demo.max().item():.4f}, mean={disc_logit_demo.mean().item():.4f}")
            print(f"  - Demo prob (sigmoid): min={demo_prob.min().item():.4f}, "
                  f"max={demo_prob.max().item():.4f}, mean={demo_prob.mean().item():.4f} "
                  f"(应该接近1，表示demo像expert)")
            print(f"  - Loss: total={disc_loss.item():.6f}, agent={loss_agent.item():.6f}, "
                  f"demo={loss_demo.item():.6f}, grad_penalty={grad_penalty.item():.6f}")
            print(f"  - 准确率: agent={agent_acc.item():.4f} (应该接近1.0，表示大部分agent被判为fake), "
                  f"demo={demo_acc.item():.4f} (应该接近1.0，表示大部分demo被判为real)")
            
            # 检查梯度
            disc_grad_norm = 0.0
            for param in self.actor_critic.discriminator.parameters():
                if param.grad is not None:
                    disc_grad_norm += param.grad.data.norm(2).item() ** 2
            disc_grad_norm = disc_grad_norm ** 0.5
            print(f"  - Discriminator梯度范数: {disc_grad_norm:.6f}")
        
        # 注意：归一化器更新在update()方法中统一进行（参考MimicKit流程）
        # 这里不调用normalizer.update()，而是在update()方法的最后调用
        
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

