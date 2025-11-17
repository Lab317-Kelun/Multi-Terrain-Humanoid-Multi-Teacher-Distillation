# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MultiStudentTeacher
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class old_Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: MultiStudentTeacher 
    """The student teacher model."""

    def __init__(
        self,
        policy: MultiStudentTeacher ,
        estimator,
        estimator_paras,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        # RMA-style regularization (privileged vs history latent alignment)
        # Format: [coef_start, coef_end, start_step, duration]
        priv_reg_coef_schedual: list[float] = [0.0, 0.0, 0.0, 1.0],
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # Initialized later

        # Initialize the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # Initialize the transition
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        # RMA priv reg schedule and counter
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        # Initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)

        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_priv_reg_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                # Inference of the student for gradient computation
                actions = self.policy.act_inference(obs)

                # Behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # RMA-style privileged-vs-history latent regularization on student
                priv_reg_loss = 0.0
                try:
                    student_obs = self.policy.get_student_obs(obs)
                    # Compute priv and hist latents from student actor
                    # Match PPO semantics: stop gradients through history branch
                    priv_latent = self.policy.student.infer_priv_latent(student_obs)
                    with torch.inference_mode():
                        hist_latent = self.policy.student.infer_hist_latent(student_obs)
                    priv_reg_loss = (priv_latent - hist_latent.detach()).norm(p=2, dim=1).mean()
                except Exception:
                    # If student has no priv/hist configured, skip the term gracefully
                    priv_reg_loss = 0.0

                # Compute scheduled coefficient
                if self.priv_reg_coef_schedual is not None and len(self.priv_reg_coef_schedual) == 4:
                    s0, s1, t0, dur = self.priv_reg_coef_schedual
                    stage = 0.0
                    if dur > 0:
                        stage = min(max((self.counter - t0), 0.0) / dur, 1.0)
                    priv_reg_coef = stage * (s1 - s0) + s0
                else:
                    priv_reg_coef = 0.0

                # Total loss
                loss = loss + behavior_loss + priv_reg_coef * priv_reg_loss
                mean_behavior_loss += behavior_loss.item()
                if isinstance(priv_reg_loss, torch.Tensor):
                    mean_priv_reg_loss += priv_reg_loss.item()
                else:
                    mean_priv_reg_loss += float(priv_reg_loss)
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # Reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= max(cnt, 1)
        mean_priv_reg_loss /= max(cnt, 1)
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        # Advance global counter (used for scheduling)
        self.counter += 1

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss, "priv_reg": mean_priv_reg_loss}

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel