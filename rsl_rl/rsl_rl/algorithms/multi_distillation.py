# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules import MultiStudentTeacher
from rsl_rl.storage import RolloutStorage
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

class Distillation:
    """Multi-teacher -> student distillation with estimator update.

    这里保留原来的类名 `PPO`，但不再做策略梯度更新，
    而是参考 `distillation.py`，对 `MultiStudentTeacher` 做教师-学生
    MSE 蒸馏；同时保留对 `estimator` 的 MSE 更新。
    """

    policy: MultiStudentTeacher

    def __init__(
        self,
        actor_critic,  # 为保持接口兼容，这里仍然接收，但实际应传入 MultiStudentTeacher
        estimator,
        estimator_paras,
        depth_encoder,
        depth_encoder_paras,
        depth_actor,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,  # 已不再使用，仅为兼容
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,  # 不再使用
        entropy_coef: float = 0.0,  # 不再使用
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,  # 不再使用
        schedule: str = "fixed",  # 不再使用
        desired_kl: float | None = 0.01,  # 不再使用
        device: str = "cpu",
        dagger_update_freq: int = 20,
        # RMA-style regularization (priv vs hist latent alignment)
        # 为与 `distillation.py` 保持一致，扩展为 [coef_start, coef_end, start_step, duration]
        priv_reg_coef_schedual = [0.0, 0.0, 0.0, 1.0],
        adam_epsilon: float = 1e-8,
        # 蒸馏损失类型
        loss_type: str = "mse",
        **kwargs,
    ):

        self.device = device

        # Distillation policy
        # 语义上这里应该传入 MultiStudentTeacher，为兼容旧接口变量名保留为 actor_critic
        self.policy: MultiStudentTeacher = actor_critic
        self.policy.to(self.device)

        # 存储 / 过渡
        self.storage = None  # initialized later
        self.transition = RolloutStorage.Transition()

        # 蒸馏 / 训练参数
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.dagger_update_freq = dagger_update_freq

        # RMA priv reg schedule and counter
        # 期望格式: [coef_start, coef_end, start_step, duration]
        if len(priv_reg_coef_schedual) == 3:
            # 兼容旧格式 [start, end, start_step]，默认 duration=1
            priv_reg_coef_schedual = [
                priv_reg_coef_schedual[0],
                priv_reg_coef_schedual[1],
                priv_reg_coef_schedual[2],
                1.0,
            ]
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0
        self.hist_encoder_optimizer = optim.Adam(self.policy.student.history_encoder.parameters(), lr=learning_rate, eps=adam_epsilon)

        # 蒸馏优化器（只优化学生参数）
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate, eps=adam_epsilon)

        # 损失函数
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(
            self.estimator.parameters(),
            lr=estimator_paras["learning_rate"],
            eps=adam_epsilon,
        )
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Depth encoder（暂时保留接口，不修改逻辑）
        self.if_depth = depth_encoder is not None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(
                self.depth_encoder.parameters(),
                lr=depth_encoder_paras["learning_rate"],
            )
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam(
                [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                lr=depth_encoder_paras["learning_rate"],
            )

        # RNN hidden states（与 distillation.py 对齐）
        self.last_hidden_states = (None, None)

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # 为了兼容原调用接口，这里仍然接收 actor/critic_obs_shape，
        # 实际上只需要与 distillation 对齐的 obs / action_shape。
        # 当与 distillation_runner 打通时，可切换到新的 init_storage 接口。
        self.storage = RolloutStorage(
            "on_policy",  # 占位 training_type
            num_envs,
            num_transitions_per_env,
            TensorDict({}, batch_size=[num_envs]),
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs=None, info=None, hist_encoding: bool = False):
        """与旧接口兼容的 act。

        这里的 obs 视为 MultiStudentTeacher 所需的 TensorDict 或张量，
        由上层环境保证格式正确。
        """
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # 仍然支持 "用估计的 priv_states 行为" 的选项
        if self.train_with_estimated_states and isinstance(obs, torch.Tensor):
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, : self.num_prop])
            obs_est[
                :,
                self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.priv_states_dim,
            ] = priv_states_estimated
            actions = self.policy.act(obs_est).detach()
        else:
            actions = self.policy.act(obs).detach()

        # teacher 动作（带梯度，用于更新学生）
        privileged_actions = self.policy.evaluate(obs).detach()

        self.transition.actions = actions
        self.transition.privileged_actions = privileged_actions
        self.transition.observations = obs

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # 保留 rewards / dones 以兼容原接口，但蒸馏不再使用 value/advantages
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

        # Reset policy hidden states on done
        if hasattr(self.policy, "reset"):
            self.policy.reset(dones)

        return rewards
    

    def update(self):
        """教师-学生蒸馏更新 + estimator 更新。

        - 学生策略: `self.policy.student`
        - 教师策略: `self.policy` 中的 teacher 分支（通过 `evaluate` 提供动作）
        - 行为损失: MSE(actions_student, actions_teacher)
        - 估计器损失: MSE(estimated_priv, true_priv)
        - RMA 特权正则: 对齐学生 priv/hist latent，与 `distillation.py` 一致
        """

        mean_behavior_loss = 0.0
        mean_priv_reg_loss = 0.0
        mean_estimator_loss = 0.0
        cnt = 0

        # 这里依然使用旧的 storage 结构（PPO 风格），需要从中构造 obs / privileged_actions
        if self.policy.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # -------------------------
            #  Estimator update (保持原逻辑)
            # -------------------------
            priv_states_predicted = self.estimator(obs_batch[:, : self.num_prop])
            true_priv_states = obs_batch[
                :,
                self.num_prop
                + self.num_scan : self.num_prop
                + self.num_scan
                + self.priv_states_dim,
            ]
            estimator_loss = (priv_states_predicted - true_priv_states).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            # -------------------------
            #  学生-教师动作蒸馏
            # -------------------------
            # 用 obs_batch 做学生推理（与 distillation.py 中 act_inference 对齐）
            if hasattr(self.policy, "act_inference"):
                actions_student = self.policy.act_inference(obs_batch)
            else:
                actions_student = self.policy.act(obs_batch)

            # 教师动作（使用 evaluate，理论上是带 priv obs 的 teacher）
            with torch.inference_mode():
                actions_teacher = self.policy.evaluate(obs_batch)

            behavior_loss = self.loss_fn(actions_student, actions_teacher)

            # -------------------------
            #  RMA-style priv-vs-hist latent 正则（学生）
            # -------------------------
            priv_reg_loss = 0.0
            try:
                if hasattr(self.policy, "get_student_obs"):
                    student_obs = self.policy.get_student_obs(obs_batch)
                else:
                    student_obs = obs_batch

                student = self.policy.student
                priv_latent = student.infer_priv_latent(student_obs)
                with torch.inference_mode():
                    hist_latent = student.infer_hist_latent(student_obs)
                priv_reg_loss = (priv_latent - hist_latent.detach()).norm(p=2, dim=1).mean()
            except Exception:
                priv_reg_loss = 0.0

            # 计算调度系数，与 distillation.py 对齐
            if self.priv_reg_coef_schedual is not None and len(self.priv_reg_coef_schedual) == 4:
                s0, s1, t0, dur = self.priv_reg_coef_schedual
                stage = 0.0
                if dur > 0:
                    stage = min(max((self.counter - t0), 0.0) / dur, 1.0)
                priv_reg_coef = stage * (s1 - s0) + s0
            else:
                priv_reg_coef = 0.0

            # -------------------------
            #  总损失 & 学生更新
            # -------------------------
            loss = behavior_loss + priv_reg_coef * (
                priv_reg_loss if isinstance(priv_reg_loss, torch.Tensor) else 0.0
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_behavior_loss += behavior_loss.item()
            if isinstance(priv_reg_loss, torch.Tensor):
                mean_priv_reg_loss += priv_reg_loss.item()
            else:
                mean_priv_reg_loss += float(priv_reg_loss)
            mean_estimator_loss += estimator_loss.item()
            cnt += 1

        cnt = max(cnt, 1)
        mean_behavior_loss /= cnt
        mean_priv_reg_loss /= cnt
        mean_estimator_loss /= cnt

        self.storage.clear()
        self.update_counter()

        return {
            "behavior": mean_behavior_loss,
            "priv_reg": mean_priv_reg_loss,
            "estimator": mean_estimator_loss,
        }

    def update_dagger(self):
        """DAgger 风格更新：只更新学生的 history encoder，使其 hist latent 贴近 priv latent。

        参考原始 PPO 里的 DAgger 思路，但作用于 `self.policy.student`。
        """

        mean_hist_latent_loss = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        student = self.policy.student
        history_encoder = getattr(student, "history_encoder", None)
        if history_encoder is None:
            # 如果学生没有 history_encoder，直接返回 0
            self.storage.clear()
            self.update_counter()
            return 0.0

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            with torch.inference_mode():
                # 让学生在 hist_encoding=True 下 rollout 一步，以确保 RNN 维度等一致
                if hasattr(self.policy, "act"):
                    self.policy.act(obs_batch, critic_obs_batch, info=None, hist_encoding=True)

            # 使用学生网络的 priv/hist encoder
            with torch.inference_mode():
                priv_latent_batch = student.infer_priv_latent(obs_batch)
            hist_latent_batch = student.infer_hist_latent(obs_batch)

            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()

            # 仅更新 history_encoder 参数
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= max(num_updates, 1)
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Depth encoder ditillation
        if self.if_depth:
            # TODO: needs to save hidden states
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()

            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
            self.depth_encoder_optimizer.step()
            return depth_encoder_loss.item()
    
    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):
        if self.if_depth:
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

            loss = depth_actor_loss + yaw_loss

            self.depth_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()
            
            return depth_actor_loss.item(), yaw_loss.item()
    
    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()
    
    def update_counter(self):
        self.counter += 1
    
    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward