# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None

import rsl_rl
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import MultiStudentTeacher
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state, tensor_to_obs_groups


class DistillationRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of teacher-student training."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = self.cfg.get("logger_type", "none")

        # Query observations from environment for algorithm construction
        raw_obs = self.env.get_observations()
        obs, self.cfg["obs_groups"] = resolve_obs_groups(raw_obs, self.cfg["obs_groups"], default_sets=["teacher"])
        obs = obs.to(self.device)

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _configure_multi_gpu(self) -> None:
        if dist.is_available() and dist.is_initialized():
            self.is_distributed = False
            self.gpu_global_rank = dist.get_rank()
            self.gpu_world_size = dist.get_world_size()
            self.multi_gpu_cfg = {"global_rank": self.gpu_global_rank, "world_size": self.gpu_world_size}
        else:
            self.is_distributed = False
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
            self.multi_gpu_cfg = None

    def _prepare_logging_writer(self) -> None:
        if self.disable_logs or self.writer is not None:
            return
        if self.logger_type == "tensorboard" and self.log_dir and SummaryWriter is not None:
            self.writer = SummaryWriter(self.log_dir, flush_secs=10)
        elif self.logger_type in {"wandb", "neptune"}:
            print("⚠️  Wandb/Neptune logging is not implemented for DistillationRunner; disabling remote logger.")
            self.logger_type = "none"

    def train_mode(self) -> None:
        self.alg.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def save(self, path: str, infos=None) -> None:  # noqa: D401 (compat signature)
        state = {
            "policy_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def log(self, locs, width: int = 80, pad: int = 35) -> None:  # noqa: D401 (compat signature)
        behavior_loss = locs["loss_dict"].get("behavior", 0.0)
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_time += iteration_time
        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6))

        mean_reward = statistics.mean(locs["rewbuffer"]) if locs["rewbuffer"] else 0.0
        mean_ep_len = statistics.mean(locs["lenbuffer"]) if locs["lenbuffer"] else 0.0

        if self.writer and self.logger_type == "tensorboard":
            step = locs["it"]
            self.writer.add_scalar("loss/behavior", behavior_loss, step)
            self.writer.add_scalar("reward/mean", mean_reward, step)
            self.writer.add_scalar("episode/length", mean_ep_len, step)
            self.writer.add_scalar("perf/fps", fps, step)

        ep_string = ""
        if locs["ep_infos"]:
            sample = locs["ep_infos"][0]
            for key in sample:
                values = torch.tensor([info[key] for info in locs["ep_infos"]], dtype=torch.float32)
                value = float(values.mean())
                ep_string += f"{('Mean episode ' + key + ':'):>{pad}} {value:.4f}\n"

        header = f" \033[1m Learning iteration {locs['it']} / {locs['num_learning_iterations']} \033[0m "
        log_lines = ["#" * width, header.center(width), "", f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)", f"{'Behavior loss:':>{pad}} {behavior_loss:.4f}", f"{'Mean reward:':>{pad}} {mean_reward:.2f}", f"{'Mean episode length:':>{pad}} {mean_ep_len:.2f}", "-" * width, ep_string, f"{'Total timesteps:':>{pad}} {self.tot_timesteps}", f"{'Iteration time:':>{pad}} {iteration_time:.2f}s", f"{'Total time:':>{pad}} {self.tot_time:.2f}s"]
        print("\n".join([line for line in log_lines if line]))

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()
        # Check if teacher is loaded
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = tensor_to_obs_groups(self.env.get_observations(), self.cfg["obs_groups"]).to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)
        
        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, 1, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, 1, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs_tensor, _, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device and wrap observations
                    obs = tensor_to_obs_groups(obs_tensor, self.cfg["obs_groups"]).to(self.device)
                    rewards = rewards.to(self.device).unsqueeze(-1)
                    dones = dones.to(self.device).unsqueeze(-1)
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        done_envs = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[done_envs][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_envs][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[done_envs] = 0
                        cur_episode_length[done_envs] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """Construct the distillation algorithm."""
        # Initialize the policy
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: MultiStudentTeacher = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        alg.init_storage(
            "distillation",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg