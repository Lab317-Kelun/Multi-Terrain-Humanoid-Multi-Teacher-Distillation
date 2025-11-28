# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Modified for BEAMDOJO Double Critic PPO implementation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.modules import ActorCriticRMADoubleReward
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer_multi import ReplayBufferMulti
import wandb
from rsl_rl.utils import unpad_trajectories


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class PPODoubleReward:
    """
    BEAMDOJO双Critic PPO实现
    支持密集奖励和稀疏奖励的分离学习
    """
    actor_critic: ActorCriticRMADoubleReward
    
    def __init__(self,
                 actor_critic,
                 estimator=None,
                 estimator_paras=None,
                 depth_encoder=None,
                 depth_encoder_paras=None,
                 depth_actor=None,
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
                 # BEAMDOJO双Critic相关参数
                 dense_reward_weight=1.0,
                 sparse_reward_weight=0.25,
                 use_separate_value_loss=True,
                 adam_epsilon=1e-8,
                 use_double_critic=True,
                 dense_value_loss_coef=1.0,
                 sparse_value_loss_coef=1.0,
                 advantage_merge_weight=0.5,
                 # ===== AMP 相关（可选） =====
                 use_amp=False,
                 discriminator=None,
                 amp_data=None,
                 amp_normalizer=None,
                 num_amp_frames=None,
                 amp_loader_type='lafan_16dof_multi',
                 disc_learning_rate=2e-5,
                 amp_replay_buffer_size=100000,
                 amp_reward_mode='quadratic',
                 amp_reward_coef=0.0,
                 **kwargs):

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=adam_epsilon)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # BEAMDOJO双Critic参数
        self.dense_reward_weight = dense_reward_weight
        self.sparse_reward_weight = sparse_reward_weight
        self.use_separate_value_loss = use_separate_value_loss
        self.use_double_critic = actor_critic.use_double_critic
        self.dense_value_loss_coef = dense_value_loss_coef
        self.sparse_value_loss_coef = sparse_value_loss_coef
        self.advantage_merge_weight = advantage_merge_weight
        self.use_double_critic = actor_critic.use_double_critic

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate, eps=adam_epsilon)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        
         # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.num_hist = estimator_paras["num_hist"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"], eps=adam_epsilon)
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
        # ===== AMP =====
        self.discriminator = discriminator
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.num_amp_frames = num_amp_frames
        self.use_amp = use_amp
        self.amp_reward_mode = amp_reward_mode
        # 兼容旧参数名 amp_reward_weight（若存在则覆盖）
        legacy_weight = kwargs.get('amp_reward_weight', None)
        self.amp_reward_coef = legacy_weight if legacy_weight is not None else amp_reward_coef
        self.last_amp_reward_mean = 0.0
        if self.use_amp:
            if self.discriminator is None:
                print("[PPODoubleReward] use_amp=True 但未提供discriminator，自动禁用AMP")
                self.use_amp = False
            else:
                self.discriminator.to(self.device)
                self.amp_transition = RolloutStorage.Transition()
                # amp_storage 保存策略生成的 AMP 状态序列（policy）
                self.amp_storage = ReplayBufferMulti(
                    self.discriminator.state_dim, amp_replay_buffer_size, self.num_amp_frames, device=self.device)
                self.amp_loader_type = amp_loader_type
                # 判别器优化器（同时提供别名，Runner保存时更稳健）
                self.optimizer_disc = optim.AdamW(self.discriminator.parameters(), lr=disc_learning_rate, weight_decay=1e-2)
                self.disc_optimizer = self.optimizer_disc

        # AMP 统计指标（用于 wandb 记录）
        self.disc_policy_score_mean = 0.0
        self.disc_policy_score_std = 0.0
        self.disc_expert_score_mean = 0.0
        self.disc_expert_score_std = 0.0
        self.demo_acc_last = 0.0
        
        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"], eps=adam_epsilon)
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"], eps=adam_epsilon)

        # 读取与 AMP 插值相关的参数（从 kwargs 注入）
        self.use_lerp = bool(kwargs.get('use_lerp', False))
        self.amp_task_reward_lerp = float(kwargs.get('amp_task_reward_lerp', 0.0))


        print(f"PPODoubleReward initialized with use_double_critic={self.use_double_critic}")
        if self.use_double_critic:
            print(f"Dense reward weight: {self.dense_reward_weight}")
            print(f"Sparse reward weight: {self.sparse_reward_weight}")

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # 使用 estimator 估计隐式特权信息（priv_explicit）
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            # 历史观测位于观测的最后 history_len * n_proprio 维
            hist_obs = obs_est[:, -self.num_hist * self.num_prop:]
            # 使用 estimator 估计 priv_explicit
            priv_explicit_estimated = self.estimator(hist_obs)
            # 将估计的 priv_explicit 替换到观测中的相应位置
            # priv_explicit 的位置：n_proprio + n_scan 到 n_proprio + n_scan + priv_states_dim
            obs_est[:, self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.priv_states_dim] = priv_explicit_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()
        
        if self.use_double_critic:
            # 双Critic评估
            value1, value2 = self.actor_critic.evaluate(critic_obs)
            self.transition.values = value1.detach()  # 主要使用密集奖励的价值
            self.transition.values_sparse = value2.detach()  # 存储稀疏奖励的价值
        else:
            # 单Critic评估
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs_frames=None):
        """处理环境步骤，支持密集和稀疏奖励分离，并可选地接收 AMP 观测序列以训练判别器。"""
        num_envs = self.transition.observations.shape[0]
        if isinstance(rewards, dict) and self.use_double_critic:
            # 如果奖励是字典格式，分离密集和稀疏奖励（不在此处混入判别器奖励）
            rewards_dense = rewards.get('dense', None)
            rewards_sparse = rewards.get('sparse', None)
            # 标准化形状到 [N,1]
            if isinstance(rewards_dense, torch.Tensor):
                rewards_dense = rewards_dense.view(-1, 1)
            else:
                rewards_dense = torch.zeros((num_envs, 1), device=self.device)
            if isinstance(rewards_sparse, torch.Tensor):
                rewards_sparse = rewards_sparse.view(-1, 1)
            else:
                rewards_sparse = torch.zeros((num_envs, 1), device=self.device)
            rewards_total = rewards_dense + rewards_sparse
            
            self.transition.rewards_dense = rewards_dense.clone()
            self.transition.rewards_sparse = rewards_sparse.clone()

        else:
            # 如果是单一奖励，作为密集奖励处理
            rewards_total = rewards.clone().view(-1, 1)
            self.transition.rewards_dense = rewards_total.clone()
            if self.use_double_critic:
                self.transition.rewards_sparse = torch.zeros_like(rewards_total)

        # ===== 模仿奖励并入密集奖励（若启用 AMP 且设置了系数）=====
        if self.use_amp and self.amp_reward_coef != 0.0 and self.discriminator is not None:
            # 1) 准备 AMP 观测序列：优先使用传入的 amp_obs_frames；否则从 obs 构造
            if amp_obs_frames is None:
                try:
                    amp_obs_frames = self.build_amp_policy_frames(self.transition.observations)
                except Exception as e:
                    amp_obs_frames = None
                    print(f"[AMP] Warning: failed to build amp policy frames: {e}")
            # 2) 计算判别器输出与模仿奖励
            if amp_obs_frames is not None:
                with torch.no_grad():
                    d_out = self.discriminator(amp_obs_frames.flatten(1))
                    if self.amp_reward_mode == 'quadratic':
                        # 与 AMPDiscriminatorMulti.predict_amp_reward 保持一致的判别器奖励形式
                        disc_reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * torch.square(d_out - 1.0), min=0.0)
                    elif self.amp_reward_mode == 'gail':
                        disc_reward = self.amp_reward_coef * (-torch.log(torch.clamp(1.0 - torch.sigmoid(d_out), min=1e-6)))
                    elif self.amp_reward_mode == 'tanh':
                        disc_reward = self.amp_reward_coef * torch.tanh(d_out)
                    elif self.amp_reward_mode == 'sigmoid':
                        disc_reward = self.amp_reward_coef * torch.sigmoid(d_out)
                    else:
                        disc_reward = self.amp_reward_coef * (-torch.log(torch.clamp(1.0 - torch.sigmoid(d_out), min=1e-6)))
                # 标准化形状到 [N,1]
                disc_reward = disc_reward.view(-1, 1)

                # 3) 与密集奖励合并：支持 lerp 或直接相加（参考 AMPDiscriminatorMulti.predict_amp_reward）
                if isinstance(rewards, dict) and self.use_double_critic:
                    if getattr(self, 'use_lerp', False) and getattr(self, 'amp_task_reward_lerp', 0.0) > 0.0:
                        lerp = float(getattr(self, 'amp_task_reward_lerp', 0.0))
                        self.transition.rewards_dense = (1.0 - lerp) * disc_reward + lerp * self.transition.rewards_dense
                    else:
                        # 非 lerp 情况下，轻度缩放判别器奖励再相加（与 amp_discriminator_multi 的非 lerp 分支保持一致的风格）
                        self.transition.rewards_dense = self.transition.rewards_dense + disc_reward * 0.02
                    rewards_total = self.transition.rewards_dense + self.transition.rewards_sparse
                else:
                    if getattr(self, 'use_lerp', False) and getattr(self, 'amp_task_reward_lerp', 0.0) > 0.0:
                        lerp = float(getattr(self, 'amp_task_reward_lerp', 0.0))
                        self.transition.rewards_dense = (1.0 - lerp) * disc_reward + lerp * self.transition.rewards_dense
                    else:
                        self.transition.rewards_dense = self.transition.rewards_dense + disc_reward * 0.02
                    rewards_total = self.transition.rewards_dense
                # 记录最近一次 AMP 奖励均值（供 Runner 日志使用）
                self.last_amp_reward_mean = float(disc_reward.mean().item())

        # 确保奖励形状为 [num_envs, 1]，避免后续广播错误
        self.transition.rewards = rewards_total.clone().view(-1, 1)
        self.transition.dones = dones

        # 写入 AMP 状态序列（若启用）
        if self.use_amp and amp_obs_frames is not None:
            self.amp_storage.insert(amp_obs_frames)
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            # 保持形状一致为 [num_envs, 1]
            time_outs_raw = infos['time_outs']
            if isinstance(time_outs_raw, torch.Tensor):
                # 先展平为 [N]，再扩展为 [N,1]
                time_outs = time_outs_raw.view(-1,).unsqueeze(1).to(self.device)
            else:
                time_outs = torch.zeros((num_envs, 1), device=self.device)
            # 统一 values 为 [N,1]
            values_bootstrap = self.transition.values.view(num_envs, 1)
            self.transition.rewards += self.gamma * (values_bootstrap * time_outs)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        # 返回给Runner的一维奖励向量，避免与其 [N] 累加时发生广播错误
        return rewards_total.view(-1)

    def compute_returns(self, last_critic_obs):
        """计算returns和advantages, 支持双Critic"""
        if self.use_double_critic:
            last_values1, last_values2 = self.actor_critic.evaluate(last_critic_obs)
            last_values1 = last_values1.detach()
            last_values2 = last_values2.detach()
            
            # 为密集和稀疏奖励分别计算returns
            self.storage.compute_returns_double(last_values1, last_values2, self.gamma, self.lam)
        else:
            last_values = self.actor_critic.evaluate(last_critic_obs).detach()
            self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_value_loss_dense = 0
        mean_value_loss_sparse = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_priv_reg_loss = 0
        mean_discriminator_loss = 0
        mean_discriminator_acc = 0
        # ===== 先训练判别器（若启用 AMP）=====
        if self.use_amp:
            pol_score_mean_acc = 0.0
            pol_score_std_acc = 0.0
            exp_score_mean_acc = 0.0
            exp_score_std_acc = 0.0
            demo_acc_acc = 0.0
            num_disc_updates = 0
            if self.amp_storage is not None and self.amp_data is not None:
                amp_policy_generator = self.amp_storage.feed_forward_generator(
                    self.num_learning_epochs * self.num_mini_batches,
                    self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
                if self.amp_loader_type in ('lafan_16dof_multi', 'lafan_16dof'):
                    amp_expert_generator = self.amp_data.feed_forward_generator_lafan_16dof_multi(
                        self.num_learning_epochs * self.num_mini_batches,
                        self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
                else:
                    raise NotImplementedError(f"Unsupported amp_loader_type: {self.amp_loader_type}")
            else:
                amp_policy_generator = []
                amp_expert_generator = []
            for sample_amp_policy, sample_amp_expert in zip(amp_policy_generator, amp_expert_generator):
                expert_states = sample_amp_expert.to(self.device)
                policy_states = sample_amp_policy
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        expert_states = self.amp_normalizer.normalize_torch(expert_states, self.device)
                        policy_states = self.amp_normalizer.normalize_torch(policy_states, self.device)
                policy_d = self.discriminator(policy_states.flatten(1))
                expert_d = self.discriminator(expert_states.flatten(1))
                agent_acc = (policy_d < 0).float().mean()
                demo_acc = (expert_d > 0).float().mean()
                # MSE 目标与正则（与 amp_ppo_multi 保持一致）
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d, device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d, device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(expert_states, lambda_=5)
                logit_weights = self.discriminator.get_disc_logit_weights()
                disc_logit_loss = 0.01 * torch.sum(torch.square(logit_weights))
                disc_weights = torch.cat(self.discriminator.get_disc_weights(), dim=-1)
                disc_weight_decay = 0.0001 * torch.sum(torch.square(disc_weights))
                disc_loss = amp_loss + grad_pen_loss + disc_logit_loss + disc_weight_decay
                self.optimizer_disc.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.optimizer_disc.step()
                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_states.cpu().numpy())
                    self.amp_normalizer.update(expert_states.cpu().numpy())
                mean_discriminator_loss += amp_loss.item()
                mean_discriminator_acc += agent_acc.mean().item()

                # 累计分数统计用于 wandb
                pol_score_mean_acc += policy_d.mean().item()
                pol_score_std_acc += policy_d.std().item()
                exp_score_mean_acc += expert_d.mean().item()
                exp_score_std_acc += expert_d.std().item()
                demo_acc_acc += demo_acc.item()
                num_disc_updates += 1

            # 更新实例属性供 Runner 记录
            if num_disc_updates > 0:
                self.disc_policy_score_mean = pol_score_mean_acc / num_disc_updates
                self.disc_policy_score_std = pol_score_std_acc / num_disc_updates
                self.disc_expert_score_mean = exp_score_mean_acc / num_disc_updates
                self.disc_expert_score_std = exp_score_std_acc / num_disc_updates
                self.demo_acc_last = demo_acc_acc / num_disc_updates
        # 根据是否使用双Critic选择合适的generator
        if self.use_double_critic and hasattr(self.storage, 'mini_batch_generator_double'):
            generator = self.storage.mini_batch_generator_double(self.num_mini_batches, self.num_learning_epochs)
        elif self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            # 使用双Critic的数据
            (obs_batch, critic_obs_batch, actions_batch, target_values_batch, 
            target_values_dense_batch, target_values_sparse_batch,
            advantages_batch, advantages_dense_batch, advantages_sparse_batch,
            returns_batch, returns_dense_batch, returns_sparse_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
            hid_states_batch, masks_batch) = batch


            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch else None)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            if self.use_double_critic:
                value_batch1, value_batch2 = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,hidden_states=hid_states_batch[1] if hid_states_batch else None)
            else:
                value_batch = self.actor_critic.evaluate(critic_obs_batch,masks=masks_batch,hidden_states=hid_states_batch[1] if hid_states_batch else None)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # Adaptation module update
            priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
            priv_reg_coef = self.priv_reg_coef_schedual[0] + (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) * priv_reg_stage

            # Estimator
            # priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # obs in batch is with true priv_states
            hist = obs_batch[:, -self.num_hist*self.num_prop:]  # 展平的历史观测: (batch, num_hist*num_prop)
            priv_states_predicted = self.estimator(hist)  # 直接输入展平的历史观测
            estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim]).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()
            
            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
                            
            # Surrogate loss (policy loss)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            
            if self.use_double_critic:
                # 合并优势：归一化后加权
                advantages_dense_norm = (advantages_dense_batch - advantages_dense_batch.mean()) / (advantages_dense_batch.std() + 1e-8)
                advantages_sparse_norm = (advantages_sparse_batch - advantages_sparse_batch.mean()) / (advantages_sparse_batch.std() + 1e-8)
                combined_advantages = (self.dense_reward_weight * advantages_dense_norm + 
                                     self.sparse_reward_weight * advantages_sparse_norm)
                surrogate = -torch.squeeze(combined_advantages) * ratio
            else:
                surrogate = -torch.squeeze(advantages_batch) * ratio

            surrogate_clipped = -torch.squeeze(combined_advantages if self.use_double_critic else advantages_batch) * \
                              torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_double_critic:
                # 分别计算两个Critic的损失
                if self.use_clipped_value_loss:
                    value_clipped1 = target_values_dense_batch + (value_batch1 - target_values_dense_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses1 = (value_batch1 - returns_dense_batch).pow(2)
                    value_losses_clipped1 = (value_clipped1 - returns_dense_batch).pow(2)
                    value_loss1 = torch.max(value_losses1, value_losses_clipped1).mean()

                    value_clipped2 = target_values_sparse_batch + (value_batch2 - target_values_sparse_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses2 = (value_batch2 - returns_sparse_batch).pow(2)
                    value_losses_clipped2 = (value_clipped2 - returns_sparse_batch).pow(2)
                    value_loss2 = torch.max(value_losses2, value_losses_clipped2).mean()
                else:
                    value_loss1 = (returns_dense_batch - value_batch1).pow(2).mean()
                    value_loss2 = (returns_sparse_batch - value_batch2).pow(2).mean()

                value_loss = value_loss1 + value_loss2
                mean_value_loss_dense += value_loss1.item()
                mean_value_loss_sparse += value_loss2.item()
            else:
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

            # Total loss
            # 总损失包含4部分：
            # 1. surrogate_loss: 策略损失，影响Actor的actor_backbone（通过actions_log_prob_batch）
            # 2. value_loss: 价值函数损失，影响Critic
            # 3. entropy_batch.mean(): 熵损失，影响Actor的actor_backbone（鼓励探索）
            # 4. priv_reg_loss: privileged信息正则化损失，影响Actor的priv_encoder
            # 
            # 注意：priv_encoder是Actor的一部分，它：
            # - 在Actor.forward()中被使用（计算latent，影响surrogate_loss）
            # - 同时通过priv_reg_loss进行正则化
            # - 通过self.optimizer（包含整个actor_critic的参数）一起优化
            loss = surrogate_loss + \
                   self.value_loss_coef * value_loss - \
                   self.entropy_coef * entropy_batch.mean() + \
                   priv_reg_coef * priv_reg_loss

            # Gradient step
            # loss.backward()会将梯度传播到所有参与计算的参数：
            # - surrogate_loss的梯度 -> actor_backbone, priv_encoder等
            # - value_loss的梯度 -> critic
            # - priv_reg_loss的梯度 -> priv_encoder
            # 然后self.optimizer.step()会更新所有参数（包括priv_encoder）
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_discriminator_loss += 0
            mean_discriminator_acc += 0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_discriminator_loss /= max(1, num_updates)  # 若 AMP 关闭，不影响
        mean_discriminator_acc /= max(1, num_updates)
        
        if self.use_double_critic:
            mean_value_loss_dense /= num_updates
            mean_value_loss_sparse /= num_updates

        self.storage.clear()
        self.update_counter()

        if self.use_double_critic: 
            return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse
        else:
            return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef

    def build_amp_policy_frames(self, obs):
        """根据当前观测构建 AMP 观测序列 [N, num_amp_frames, state_dim]。
        默认使用 12 维动作历史（与 num_actions 对齐）作为判别器状态；
        历史帧来自观测尾部的展平历史，每帧取 proprio 的末尾 12 维（action_history）。
        """
        if not self.use_amp or self.discriminator is None:
            return None
        num_envs = obs.shape[0]
        state_dim = self.discriminator.state_dim
        # 当前帧的 12 维动作历史（proprio 的末尾 12 维）
        current_actions_12 = obs[:, :self.num_prop][:, -12:]
        # 历史观测（展平）重塑为 [N, num_hist, num_prop]，并提取每帧末尾 12 维动作历史
        hist_flat = obs[:, -self.num_hist * self.num_prop:]
        hist = hist_flat.view(num_envs, self.num_hist, self.num_prop)
        hist_actions_12 = hist[:, :, -12:]

        # 组装最近的 num_amp_frames：使用 (num_amp_frames-1) 个历史帧 + 当前帧
        frames_needed_hist = max(self.num_amp_frames - 1, 0)
        if frames_needed_hist > 0:
            take = min(frames_needed_hist, self.num_hist)
            selected_hist = hist_actions_12[:, -take:, :]
        else:
            selected_hist = current_actions_12.new_zeros((num_envs, 0, 12))

        frames = torch.cat([selected_hist, current_actions_12.unsqueeze(1)], dim=1)
        # 若不足 num_amp_frames，前面用零帧填充
        if frames.shape[1] < self.num_amp_frames:
            pad_len = self.num_amp_frames - frames.shape[1]
            pad = current_actions_12.new_zeros((num_envs, pad_len, 12))
            frames = torch.cat([pad, frames], dim=1)

        # 与判别器的 state_dim 对齐（通常为 12）
        cur_dim = frames.shape[-1]
        if cur_dim == state_dim:
            return frames
        elif cur_dim > state_dim:
            return frames[:, :, :state_dim]
        else:
            pad_feat = current_actions_12.new_zeros((num_envs, self.num_amp_frames, state_dim - cur_dim))
            return torch.cat([frames, pad_feat], dim=-1)

    def update_counter(self):
        self.counter += 1
     
    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        pass

    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):
        pass

    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        pass
    def compute_apt_reward(self, source, target):
        pass
