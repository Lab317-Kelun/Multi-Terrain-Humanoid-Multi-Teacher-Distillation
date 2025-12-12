#!/usr/bin/env python3
"""Entry point for teacher-student distillation on the BEAMDOJO humanoid task."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.humanoid.humanoid_beamdojo_config import HumanoidBEAMDOJOCfgPPO
from legged_gym.utils import get_args, task_registry
from rsl_rl.runners.distillation_runner import DistillationRunner
import torch

# Configuration for Multi-Teacher Distillation
# Map terrain_id (int) to checkpoint path (str)
# Please update these paths with actual checkpoints corresponding to each terrain type
TEACHER_CHECKPOINTS = {
     14: "/home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/legged_gym/logs/teachers/14model_43000.pt",
     #8: "/home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/legged_gym/logs/teachers/8model_9000.pt",
     #3: "/home/cft/yanzhe/Multi-Terrain-Humanoid-Multi-Teacher-Distillation/model_7000.pt",
}

def build_distillation_cfg(env_cfg) -> Dict:
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
        "num_teachers": max(len(TEACHER_CHECKPOINTS), 1),
        "teacher_terrain_ids": sorted(TEACHER_CHECKPOINTS.keys()) if TEACHER_CHECKPOINTS else None,
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
        "logger_type": "wandb",
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
    if TEACHER_CHECKPOINTS:
        print(f"Loading {len(TEACHER_CHECKPOINTS)} teachers from configuration...")
        # Sort keys to ensure consistent order with 'teacher_terrain_ids' passed to policy
        sorted_ids = sorted(TEACHER_CHECKPOINTS.keys())
        
        for i, t_id in enumerate(sorted_ids):
            ckpt_path = TEACHER_CHECKPOINTS[t_id]
            if not os.path.exists(ckpt_path):
                print(f"Warning: Teacher checkpoint for terrain {t_id} not found at {ckpt_path}")
                continue
            
            print(f"Loading teacher for terrain {t_id} (index {i}) from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            # PPO checkpoints wrap the actor parameters inside 'model_state_dict'
            teacher_state = state.get("model_state_dict", state)
            
            # Fix: Prepare state dict for Actor module by removing 'actor.' prefix
            actor_state_dict = {}
            for k, v in teacher_state.items():
                if k.startswith("actor."):
                    actor_state_dict[k.replace("actor.", "")] = v
                elif not k.startswith("critic.") and not k.startswith("std"):
                    # Fallback for keys that might already be correct or belong to other parts
                    actor_state_dict[k] = v

            if i < len(runner.alg.policy.teachers):
                # Load and verify
                missing, unexpected = runner.alg.policy.teachers[i].load_state_dict(actor_state_dict, strict=False)
                
                # Check if critical weights are missing
                critical_missing = [k for k in missing if "actor_backbone" in k or "encoder" in k]
                if critical_missing:
                    print(f"⚠️  WARNING: Teacher {i} (Terrain {t_id}) seems to have failed loading critical weights!")
                    print(f"    Missing keys example: {critical_missing[:3]}")
                else:
                    print(f"✅ Teacher {i} (Terrain {t_id}) loaded successfully.")
                
            else:
                print(f"Error: Teacher index {i} out of range (num_teachers={len(runner.alg.policy.teachers)}).")
        
        # Attempt to load observation normalizer from the first teacher checkpoint (best effort)
        if len(sorted_ids) > 0:
            first_ckpt = TEACHER_CHECKPOINTS[sorted_ids[0]]
            state = torch.load(first_ckpt, map_location=device)
            teacher_state = state.get("model_state_dict", state)
            norm_state_dict = {}
            for k, v in teacher_state.items():
                if "actor_obs_normalizer." in k:
                    norm_state_dict[k.replace("actor_obs_normalizer.", "")] = v
            
            if norm_state_dict and hasattr(runner.alg.policy, "teacher_obs_normalizer"):
                print(f"Loading teacher observation normalizer from first checkpoint: {first_ckpt}")
                runner.alg.policy.teacher_obs_normalizer.load_state_dict(norm_state_dict, strict=False)

        runner.alg.policy.loaded_teacher = True
    else:
        if not checkpoint:
            raise ValueError("Please provide --teacher_checkpoint pointing to a trained PPO model or configure TEACHER_CHECKPOINTS.")
        print(f"Loading single teacher from {checkpoint} into all teacher slots...")
        state = torch.load(checkpoint, map_location=device)
        # PPO checkpoints wrap the actor parameters inside 'model_state_dict'
        teacher_state = state.get("model_state_dict", state)
        
        # Fix: Prepare state dict
        actor_state_dict = {}
        norm_state_dict = {}
        for k, v in teacher_state.items():
            if k.startswith("actor."):
                actor_state_dict[k.replace("actor.", "")] = v
            elif "actor_obs_normalizer." in k:
                norm_state_dict[k.replace("actor_obs_normalizer.", "")] = v
            elif not k.startswith("critic.") and not k.startswith("std"):
                actor_state_dict[k] = v

        for idx, teacher in enumerate(runner.alg.policy.teachers):
            missing, unexpected = teacher.load_state_dict(actor_state_dict, strict=False)
            critical_missing = [k for k in missing if "actor_backbone" in k]
            if critical_missing:
                 print(f"⚠️  WARNING: Teacher {idx} failed to load weights properly.")
            else:
                 print(f"✅ Teacher {idx} loaded successfully.")
        
        if norm_state_dict and hasattr(runner.alg.policy, "teacher_obs_normalizer"):
            print("Loading teacher observation normalizer.")
            runner.alg.policy.teacher_obs_normalizer.load_state_dict(norm_state_dict, strict=False)


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
