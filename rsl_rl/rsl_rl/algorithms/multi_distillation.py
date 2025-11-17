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

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules import MultiStudentTeacher
from rsl_rl.storage import RolloutStorage


class DistillationStorage:
    """Lightweight buffer tailored for teacher-student supervision."""

    def __init__(self, num_envs: int, num_steps: int, student_obs_dim: int, action_dim: int, device: str) -> None:
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.student_obs = torch.zeros(num_steps, num_envs, student_obs_dim, device=device)
        self.student_actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.teacher_actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.dones = torch.zeros(num_steps, num_envs, 1, device=device)
        self.step = 0

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError("DistillationStorage overflow")
        self.student_obs[self.step].copy_(transition.observations)
        self.student_actions[self.step].copy_(transition.actions)
        self.teacher_actions[self.step].copy_(transition.privileged_actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def mini_batch_generator(self, num_mini_batches: int, num_learning_epochs: int):
        total_samples = self.step * self.num_envs
        if total_samples == 0:
            return
        student_obs = self.student_obs[: self.step].reshape(-1, self.student_obs.shape[-1])
        teacher_actions = self.teacher_actions[: self.step].reshape(-1, self.teacher_actions.shape[-1])

        batch_size = max(total_samples // max(num_mini_batches, 1), 1)
        for _ in range(num_learning_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                batch_idx = indices[start:end]
                yield student_obs[batch_idx], teacher_actions[batch_idx]

    def clear(self) -> None:
        self.step = 0


class Distillation:
    """Multi-teacher -> student distillation with estimator and latent regularization."""

    policy: MultiStudentTeacher

    def __init__(
        self,
        actor_critic,
        estimator,
        estimator_paras,
        depth_encoder=None,
        depth_encoder_paras=None,
        depth_actor=None,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        priv_reg_coef_schedual: list[float] | None = None,
        adam_epsilon: float = 1e-8,
        loss_type: str = "mse",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del depth_encoder, depth_encoder_paras, depth_actor, kwargs
        self.device = device
        self.policy: MultiStudentTeacher = actor_critic.to(device)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.priv_reg_coef_schedual = priv_reg_coef_schedual or [0.0, 0.0, 0.0, 1.0]
        self.counter = 0

        # Storage
        self.storage: DistillationStorage | None = None
        self.transition = RolloutStorage.Transition()

        # Student optimizer (only student parameters are trained)
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate, eps=adam_epsilon)

        # Loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type not in loss_fn_dict:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")
        self.loss_fn = loss_fn_dict[loss_type]

        # Estimator
        self.estimator = estimator.to(device)
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.estimator_optimizer = optim.Adam(
            self.estimator.parameters(),
            lr=estimator_paras["learning_rate"],
            eps=adam_epsilon,
        )
        self.train_with_estimated_states = estimator_paras.get("train_with_estimated_states", False)

    def init_storage(
        self,
        student_obs_dim: int,
        num_envs: int,
        num_transitions_per_env: int,
        action_dim: int,
        teacher_obs_dim: int | None = None,
    ) -> None:
        del teacher_obs_dim
        self.storage = DistillationStorage(num_envs, num_transitions_per_env, student_obs_dim, action_dim, self.device)

    def train_mode(self) -> None:
        self.policy.train()

    def test_mode(self) -> None:
        self.policy.eval()

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        actions = self.policy.act(obs).detach()
        teacher_actions = self.policy.evaluate(obs).detach()
        student_obs = self.policy.get_student_obs(obs).detach()
        self.transition.actions = actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = student_obs
        return actions

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos=None):
        del infos
        if self.storage is None:
            raise RuntimeError("Storage must be initialized before collecting transitions")
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()
        self.storage.add_transition(self.transition)
        self.transition.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset(dones)
        return rewards

    def update(self):
        if self.storage is None or self.storage.step == 0:
            return {"behavior": 0.0, "priv_reg": 0.0, "estimator": 0.0}

        mean_behavior_loss = 0.0
        mean_priv_reg_loss = 0.0
        mean_estimator_loss = 0.0
        batches = 0

        priv_start = self.num_prop + self.num_scan

        for student_obs_batch, teacher_actions_batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            batches += 1

            estimator_input = student_obs_batch[:, : self.num_prop]
            true_priv_states = student_obs_batch[:, priv_start : priv_start + self.priv_states_dim]
            priv_states_pred = self.estimator(estimator_input)
            estimator_loss = nn.functional.mse_loss(priv_states_pred, true_priv_states)
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            if self.train_with_estimated_states:
                student_inputs = student_obs_batch.clone()
                student_inputs[:, priv_start : priv_start + self.priv_states_dim] = priv_states_pred.detach()
            else:
                student_inputs = student_obs_batch

            student_inputs = self.policy.student_obs_normalizer(student_inputs)
            student_actions = self.policy.student(student_inputs, hist_encoding=self.policy.student_hist_encoding)
            behavior_loss = self.loss_fn(student_actions, teacher_actions_batch)

            priv_reg_loss = self._compute_priv_reg(student_inputs)
            priv_reg_coef = self._resolve_priv_reg_coef()
            total_loss = behavior_loss + priv_reg_coef * priv_reg_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_behavior_loss += behavior_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_estimator_loss += estimator_loss.item()

        denom = max(batches, 1)
        stats = {
            "behavior": mean_behavior_loss / denom,
            "priv_reg": mean_priv_reg_loss / denom,
            "estimator": mean_estimator_loss / denom,
        }

        self.storage.clear()
        self.update_counter()
        return stats

    def _compute_priv_reg(self, student_inputs: torch.Tensor) -> torch.Tensor:
        student = self.policy.student
        try:
            priv_latent = student.infer_priv_latent(student_inputs)
            with torch.inference_mode():
                hist_latent = student.infer_hist_latent(student_inputs)
            return (priv_latent - hist_latent.detach()).norm(p=2, dim=1).mean()
        except AttributeError:
            return torch.tensor(0.0, device=student_inputs.device)

    def _resolve_priv_reg_coef(self) -> float:
        coef_start, coef_end, start_step, duration = self.priv_reg_coef_schedual
        if duration <= 0:
            return coef_end
        phase = max(self.counter - start_step, 0.0) / duration
        phase = max(0.0, min(phase, 1.0))
        return coef_start + phase * (coef_end - coef_start)

    def update_counter(self):
        self.counter += 1
