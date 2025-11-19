# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Union,  Dict, List

import torch
import torch.nn as nn
from tensordict import TensorDict


from rsl_rl.modules.teacher_student import MultiStudentTeacher
from rsl_rl.utils import resolve_optimizer


@dataclass
class DistillationTransition:
    observations: Union[TensorDict, None] = None
    privileged_actions: Union[torch.Tensor, None] = None
    actions: Union[torch.Tensor, None] = None
    rewards: Union[torch.Tensor, None] = None
    dones: Union[torch.Tensor, None] = None

    def clear(self) -> None:
        self.observations = None
        self.privileged_actions = None
        self.actions = None
        self.rewards = None
        self.dones = None


class DistillationStorage:
    def __init__(self, num_envs: int, num_transitions_per_env: int, device: str) -> None:
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        self.clear()

    def add_transition(self, transition: DistillationTransition) -> None:
        if transition.observations is None or transition.privileged_actions is None or transition.dones is None:
            raise RuntimeError("Incomplete transition passed to DistillationStorage.")
        self.observations.append(transition.observations.clone())
        self.teacher_actions.append(transition.privileged_actions.clone())
        self.dones.append(transition.dones.clone())

    def generator(self):
        for obs, teacher_act, dones in zip(self.observations, self.teacher_actions, self.dones):
            yield obs, None, teacher_act, dones

    def clear(self) -> None:
        self.observations: list[TensorDict] = []
        self.teacher_actions: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: MultiStudentTeacher 
    """The student teacher model."""

    def __init__(
        self,
        policy: MultiStudentTeacher ,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: Union[float, None] = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: Union[Dict, None] = None,
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
        self.storage: Union[DistillationStorage, None] = None

        # Initialize the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # Initialize the transition
        self.transition = DistillationTransition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

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
        actions_shape,
    ) -> None:
        # Create rollout storage
        self.storage = DistillationStorage(num_envs, num_transitions_per_env, self.device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: Dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)

        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage before collecting data.")
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self) -> Dict[str, float]:
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            if self.storage is None:
                raise RuntimeError("Storage not initialized. Call init_storage before calling update().")
            for obs, _, privileged_actions, dones in self.storage.generator():
                obs = obs.clone()
                privileged_actions = privileged_actions.clone()
                # Inference of the student for gradient computation
                actions = self.policy.act_inference(obs)

                # Behavior cloning loss
                
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
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

        mean_behavior_loss /= cnt
        if self.storage is not None:
            self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

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