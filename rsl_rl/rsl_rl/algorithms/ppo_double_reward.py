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

        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"], eps=adam_epsilon)
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"], eps=adam_epsilon)


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

    def process_env_step(self, rewards, dones, infos):
        """处理环境步骤，支持密集和稀疏奖励分离"""
        if isinstance(rewards, dict) and self.use_double_critic:
            # 如果奖励是字典格式，分离密集和稀疏奖励
            rewards_dense = rewards.get('dense', torch.zeros_like(rewards.get('total', rewards.get('sparse', torch.zeros(1)))))
            rewards_sparse = rewards.get('sparse', torch.zeros_like(rewards_dense))
            rewards_total = rewards_dense + rewards_sparse
            
            self.transition.rewards_dense = rewards_dense.clone()
            self.transition.rewards_sparse = rewards_sparse.clone()
        else:
            # 如果是单一奖励，作为密集奖励处理
            rewards_total = rewards.clone()
            self.transition.rewards_dense = rewards_total.clone()
            if self.use_double_critic:
                self.transition.rewards_sparse = torch.zeros_like(rewards_total)

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total

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
            loss = surrogate_loss + \
                   self.value_loss_coef * value_loss - \
                   self.entropy_coef * entropy_batch.mean() + \
                   priv_reg_coef * priv_reg_loss

            # Gradient step
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
        mean_discriminator_loss /= num_updates
        mean_discriminator_acc /= num_updates
        
        if self.use_double_critic:
            mean_value_loss_dense /= num_updates
            mean_value_loss_sparse /= num_updates

        self.storage.clear()
        self.update_counter()

        if self.use_double_critic: 
            return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse
        else:
            return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef

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