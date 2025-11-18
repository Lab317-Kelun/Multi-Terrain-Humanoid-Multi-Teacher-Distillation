#!/usr/bin/env python3
"""Entry point for teacher-student distillation on the BEAMDOJO humanoid task."""

from __future__ import annotations

import os
from datetime import datetime

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.humanoid.humanoid_beamdojo_config import HumanoidBEAMDOJOCfgPPO
from legged_gym.utils import get_args, task_registry
from rsl_rl.runners.distillation_runner import DistillationRunner


def build_distillation_cfg(env_cfg) -> dict:
    """Assemble the runner/algorithm configuration for distillation."""
    num_prop = getattr(env_cfg.env, "n_proprio", 0)
    num_scan = getattr(env_cfg.env, "n_scan", 0)
    num_priv_explicit = getattr(env_cfg.env, "n_priv", 0)
    num_priv_latent = getattr(env_cfg.env, "n_priv_latent", 0)
    num_hist = getattr(env_cfg.env, "history_len", 0)
    history_dim = num_hist * num_prop

    obs_groups = {
        "groups": {
            "proprio": {"start": 0, "length": num_prop},
            "scan": {"start": num_prop, "length": num_scan},
            "priv_explicit": {"start": num_prop + num_scan, "length": num_priv_explicit},
            "priv_latent": {
                "start": num_prop + num_scan + num_priv_explicit,
                "length": num_priv_latent,
            },
            "history": {
                "start": num_prop + num_scan + num_priv_explicit + num_priv_latent,
                "length": history_dim,
            },
        },
        "policy": ["proprio", "scan", "history"],
        "teacher": ["proprio", "scan", "priv_explicit", "priv_latent", "history"],
    }

    policy_defaults = HumanoidBEAMDOJOCfgPPO.policy
    runner_defaults = HumanoidBEAMDOJOCfgPPO.runner

    policy_cfg = {
        "class_name": "MultiStudentTeacher",
        "activation": policy_defaults.activation,
        "scan_encoder_dims": list(policy_defaults.scan_encoder_dims),
        "priv_encoder_dims": list(policy_defaults.priv_encoder_dims),
        "tanh_encoder_output": policy_defaults.tanh_encoder_output,
        # Student layout (no privileged channels)
        "student_num_prop": num_prop,
        "student_num_scan": num_scan,
        "student_num_priv_latent": 0,
        "student_num_priv_explicit": 0,
        "student_num_hist": num_hist,
        "student_actor_hidden_dims": list(policy_defaults.actor_hidden_dims),
        "student_hist_encoding": True,
        "student_obs_normalization": True,
        # Teacher layout (full information)
        "teacher_num_prop": num_prop,
        "teacher_num_scan": num_scan,
        "teacher_num_priv_latent": num_priv_latent,
        "teacher_num_priv_explicit": num_priv_explicit,
        "teacher_num_hist": num_hist,
        "teacher_actor_hidden_dims": list(policy_defaults.actor_hidden_dims),
        "teacher_hist_encoding": True,
        "teacher_obs_normalization": True,
        # Noise configuration
        "init_noise_std": 0.1,
        "noise_std_type": "scalar",
    }

    algorithm_cfg = {
        "class_name": "Distillation",
        "num_learning_epochs": 1,
        "gradient_length": 8,
        "learning_rate": 5e-4,
        "max_grad_norm": 1.0,
        "loss_type": "mse",
        "optimizer": "adam",
    }

    cfg = {
        "num_steps_per_env": getattr(runner_defaults, "num_steps_per_env", 24),
        "save_interval": getattr(runner_defaults, "save_interval", 200),
        "max_iterations": getattr(runner_defaults, "max_iterations", 10000),
        "logger_type": "tensorboard",
        "obs_groups": obs_groups,
        "policy": policy_cfg,
        "algorithm": algorithm_cfg,
    }
    return cfg


def prepare_log_dir(args) -> str:
    if not getattr(args, "proj_name", None):
        args.proj_name = "beamdojo"
    if not getattr(args, "exptid", None):
        args.exptid = f"distill_{datetime.now().strftime('%m%d_%H%M')}"
    stamp = datetime.now().strftime("%b%d_%H-%M-%S--") + args.exptid
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, stamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def load_teacher_policy(runner: DistillationRunner, checkpoint: str, device: str) -> None:
    if not checkpoint:
        raise ValueError("Please provide --teacher_checkpoint pointing to a trained PPO model.")
    state = torch.load(checkpoint, map_location=device)
    # PPO checkpoints wrap the actor parameters inside 'model_state_dict'
    teacher_state = state.get("model_state_dict", state)
    runner.alg.policy.load_state_dict(teacher_state, strict=False)


def maybe_resume_student(runner: DistillationRunner, checkpoint: str | None, device: str) -> None:
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    policy_state = state.get("policy_state_dict")
    if policy_state:
        runner.alg.policy.load_state_dict(policy_state, strict=False)
    optimizer_state = state.get("optimizer_state_dict")
    if optimizer_state:
        runner.alg.optimizer.load_state_dict(optimizer_state)


def main():
    args = get_args()
    if getattr(args, "task", None) in (None, "h1_2_fix"):
        args.task = "humanoid_beamdojo"

    log_dir = prepare_log_dir(args)

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    train_cfg = build_distillation_cfg(env_cfg)

    runner = DistillationRunner(env, train_cfg, log_dir=log_dir, device=args.rl_device)
    load_teacher_policy(runner, args.teacher_checkpoint, args.rl_device)
    maybe_resume_student(runner, getattr(args, "student_checkpoint", None), args.rl_device)

    num_iterations = args.distill_iters or train_cfg.get("max_iterations", 1000)
    print(f"🚀 Starting distillation for {num_iterations} iterations. Logs: {log_dir}")
    runner.learn(num_iterations, init_at_random_ep_len=True)
    print("✅ Distillation finished.")


if __name__ == "__main__":
    main()
