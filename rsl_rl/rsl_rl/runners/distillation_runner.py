# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from copy import deepcopy

import torch
from tensordict import TensorDict

import rsl_rl
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import Estimator, MultiStudentTeacher
from rsl_rl.utils import resolve_obs_groups, store_code_state


class DistillationRunner:
    """On-policy runner that performs teacher-student policy distillation."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        init_wandb: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del init_wandb  # unused but kept for API parity
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.cfg = train_cfg
        self.policy_cfg = deepcopy(train_cfg.get("policy", {}))
        self.alg_cfg = deepcopy(train_cfg.get("algorithm", {}))
        self.estimator_cfg = deepcopy(train_cfg.get("estimator", {}))
        self.runner_cfg = deepcopy(train_cfg.get("runner", {}))

        self.num_steps_per_env = self.runner_cfg.get("num_steps_per_env", 24)
        self.save_interval = self.runner_cfg.get("save_interval", 200)

        # Sample observations to build policy and obs-groups metadata
        obs_td = self._build_obs_tensordict(
            self.env.get_observations(),
            self.env.get_privileged_observations(),
        )
        self.runner_cfg["obs_groups"] = resolve_obs_groups(
            obs_td,
            self.runner_cfg.get("obs_groups"),
            default_sets=["policy", "teacher"],
        )

        # Construct algorithm
        self.alg = self._construct_algorithm(obs_td)

        # Logging helpers
        self.disable_logs = False
        self.logger_type = None
        self.writer = None
        self.git_status_repos = [rsl_rl.__file__]
        self.current_learning_iteration = 0
        self.tot_time = 0.0

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _build_obs_tensordict(self, policy_obs: torch.Tensor, privileged_obs: torch.Tensor | None) -> TensorDict:
        policy_obs = policy_obs.to(self.device)
        if privileged_obs is None:
            privileged_obs = policy_obs
        else:
            privileged_obs = privileged_obs.to(self.device)
        return TensorDict(
            {
                "policy_obs": policy_obs,
                "teacher_obs": privileged_obs,
            },
            batch_size=[policy_obs.shape[0]],
        )

    def _construct_algorithm(self, obs_td: TensorDict) -> Distillation:
        """Instantiate policy, estimator, and algorithm components."""

        policy_cfg = deepcopy(self.policy_cfg)
        policy_class_name = policy_cfg.pop("class_name", "MultiStudentTeacher")
        policy_cls = eval(policy_class_name) if isinstance(policy_class_name, str) else policy_class_name
        student_teacher: MultiStudentTeacher = policy_cls(
            obs_td,
            self.runner_cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        estimator_cfg = deepcopy(self.estimator_cfg)
        estimator_hidden = estimator_cfg.get("hidden_dims", [256, 128, 64])
        estimator_input_dim = estimator_cfg.get("num_prop", self.env.cfg.env.n_proprio)
        estimator_output_dim = estimator_cfg.get("priv_states_dim", self.env.cfg.env.n_priv)
        estimator = Estimator(
            input_dim=estimator_input_dim,
            output_dim=estimator_output_dim,
            hidden_dims=estimator_hidden,
            activation=estimator_cfg.get("activation", "elu"),
        ).to(self.device)
        estimator_paras = {
            "priv_states_dim": estimator_output_dim,
            "num_prop": estimator_cfg.get("num_prop", self.env.cfg.env.n_proprio),
            "num_scan": estimator_cfg.get("num_scan", self.env.cfg.env.n_scan),
            "learning_rate": estimator_cfg.get("learning_rate", 1e-4),
            "train_with_estimated_states": estimator_cfg.get("train_with_estimated_states", False),
        }

        alg_cfg = deepcopy(self.alg_cfg)
        alg_class_name = alg_cfg.pop("class_name", "Distillation")
        alg_cls = eval(alg_class_name) if isinstance(alg_class_name, str) else alg_class_name
        alg: Distillation = alg_cls(
            student_teacher,
            estimator=estimator,
            estimator_paras=estimator_paras,
            depth_encoder=None,
            depth_encoder_paras={},
            depth_actor=None,
            device=self.device,
            **alg_cfg,
        )

        student_obs_dim = obs_td["policy_obs"].shape[-1]
        teacher_obs_dim = obs_td["teacher_obs"].shape[-1]
        alg.init_storage(
            student_obs_dim,
            self.env.num_envs,
            self.num_steps_per_env,
            self.env.num_actions,
            teacher_obs_dim,
        )
        return alg

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_td = self._build_obs_tensordict(
            self.env.get_observations(),
            self.env.get_privileged_observations(),
        )
        self.alg.train_mode()

        ep_infos: list[dict] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        total_iters = start_iter + num_learning_iterations

        for it in range(start_iter, total_iters):
            rollout_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_td)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    rewards_tensor = self._to_reward_tensor(rewards).to(self.device)
                    dones_tensor = dones.to(self.device)
                    obs_td = self._build_obs_tensordict(obs, privileged_obs)
                    self.alg.process_env_step(rewards_tensor, dones_tensor, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards_tensor.squeeze(-1)
                        cur_episode_length += 1
                        finished = (dones_tensor > 0).nonzero(as_tuple=False)
                        if finished.numel() > 0:
                            rewbuffer.extend(cur_reward_sum[finished[:, 0]].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[finished[:, 0]].cpu().numpy().tolist())
                            cur_reward_sum[finished[:, 0]] = 0
                            cur_episode_length[finished[:, 0]] = 0

            collection_time = time.time() - rollout_start
            learn_start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start
            self.current_learning_iteration = it + 1
            self.tot_time += collection_time + learn_time

            if self.log_dir is not None and not self.disable_logs:
                self.log(
                    {
                        "it": self.current_learning_iteration,
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "ep_infos": ep_infos.copy(),
                        "rewbuffer": rewbuffer.copy(),
                        "lenbuffer": lenbuffer.copy(),
                        **loss_dict,
                    }
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                store_code_state(self.log_dir, self.git_status_repos)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _to_reward_tensor(self, rewards) -> torch.Tensor:
        if isinstance(rewards, dict):
            total = None
            for value in rewards.values():
                total = value if total is None else total + value
            rewards = total
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        return rewards

    def log(self, stats: dict, width: int = 80, pad: int = 32) -> None:
        fps = int(self.num_steps_per_env * self.env.num_envs / (stats['collection_time'] + stats['learn_time']))
        log_lines = ["#" * width]
        header = f" Distillation iteration {stats['it']} "
        log_lines.append(header.center(width, ' '))
        log_lines.append("")
        log_lines.append(f"{'FPS:':>{pad}} {fps}")
        log_lines.append(f"{'Collection time:':>{pad}} {stats['collection_time']:.3f}s")
        log_lines.append(f"{'Learning time:':>{pad}} {stats['learn_time']:.3f}s")
        for key in ["behavior", "priv_reg", "estimator"]:
            if key in stats:
                log_lines.append(f"{(key + ' loss:'):>{pad}} {stats[key]:.6f}")
        if stats.get('rewbuffer'):
            log_lines.append(f"{'Mean reward:':>{pad}} {statistics.mean(stats['rewbuffer']):.3f}")
        if stats.get('lenbuffer'):
            log_lines.append(f"{'Mean episode length:':>{pad}} {statistics.mean(stats['lenbuffer']):.1f}")
        log_lines.append("-" * width)
        print("\n".join(log_lines))

    def save(self, path: str, infos: dict | None = None) -> None:
        state = {
            "policy_state_dict": self.alg.policy.state_dict(),
            "student_state_dict": self.alg.policy.student.state_dict(),
            "teacher_state_dict": self.alg.policy.teacher.state_dict(),
            "estimator_state_dict": self.alg.estimator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(state, path)

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        if "policy_state_dict" in checkpoint:
            self.alg.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        elif "student_state_dict" in checkpoint:
            self.alg.policy.student.load_state_dict(checkpoint["student_state_dict"], strict=False)
            if "teacher_state_dict" in checkpoint:
                self.alg.policy.teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=False)
        if "estimator_state_dict" in checkpoint:
            self.alg.estimator.load_state_dict(checkpoint["estimator_state_dict"], strict=False)
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "estimator_optimizer_state_dict" in checkpoint:
            self.alg.estimator_optimizer.load_state_dict(checkpoint["estimator_optimizer_state_dict"])
        self.current_learning_iteration = checkpoint.get("iter", 0)
        return checkpoint.get("infos")

    def train_mode(self) -> None:
        self.alg.train_mode()

    def eval_mode(self) -> None:
        self.alg.test_mode()