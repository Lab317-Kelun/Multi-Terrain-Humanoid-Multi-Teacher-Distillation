#!/usr/bin/env python3
"""Play script for distilled student policy on the BEAMDOJO humanoid task."""

from __future__ import annotations

import os

from typing import Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.humanoid.humanoid_beamdojo_config import HumanoidBEAMDOJOCfgPPO
from legged_gym.utils import get_args, task_registry
from legged_gym.envs import HumanoidRobot

from rsl_rl.modules.teacher_student import MultiStudentTeacher
from rsl_rl.utils import resolve_obs_groups, tensor_to_obs_groups
import torch

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

    cfg = {
        "obs_groups": obs_groups,
        "policy": policy_cfg,
    }
    return cfg

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.exists(root):
        print(f"Logging directory {root} does not exist.")
        return None, None
        
    if load_run == -1:
        # Find the latest run
        runs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)))
        if not runs:
            print(f"No runs found in {root}")
            return None, None
        load_run = runs[-1]
    
    run_dir = os.path.join(root, load_run)
    
    if checkpoint == -1:
        models = [file for file in os.listdir(run_dir) if model_name_include in file and file.endswith(".pt")]
        models.sort(key=lambda m: int(m.split("_")[-1].split(".")[0]))
        if not models:
            print(f"No models found in {run_dir}")
            return None, None
        checkpoint = models[-1]
    else:
        # If checkpoint is a number, construct the filename
        if isinstance(checkpoint, int) or (isinstance(checkpoint, str) and checkpoint.isdigit()):
             checkpoint = f"model_{checkpoint}.pt"
    
    return os.path.join(run_dir, checkpoint), load_run

def play(args):
    # 1. Prepare Environment Configuration
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    
    # Override parameters for play
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50) # Play with fewer envs
    env_cfg.env.episode_length_s = 20 # Longer episodes for observation
    env_cfg.commands.resampling_time = 4
    env_cfg.rewards.is_play = True
    
    # Terrain config for play
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False # Usually disable noise for play
    #env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # 2. Create Environment
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 3. Build Distillation Config & Initialize Policy
    distill_cfg = build_distillation_cfg(env_cfg)
    
    # Resolve observation groups
    raw_obs = env.get_observations()
    obs, distill_cfg["obs_groups"] = resolve_obs_groups(raw_obs, distill_cfg["obs_groups"], default_sets=["teacher"])
    obs = obs.to(env.device)
    
    # Initialize MultiStudentTeacher
    policy_cfg = distill_cfg["policy"]
    policy = MultiStudentTeacher(
        obs, 
        distill_cfg["obs_groups"], 
        env.num_actions, 
        **policy_cfg
    ).to(env.device)
    policy.eval()

    # 4. Load Model
    if args.checkpoint_path:
        load_path = args.checkpoint_path
    else:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name)
        load_path, _ = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    
    if not load_path or not os.path.exists(load_path):
        print(f"Could not load model from: {load_path}")
        return

    print(f"Loading model from: {load_path}")
    state_dict = torch.load(load_path, map_location=env.device)
    
    # Handle different checkpoint formats
    if "policy_state_dict" in state_dict:
        policy.load_state_dict(state_dict["policy_state_dict"], strict=False)
    elif "model_state_dict" in state_dict:
        policy.load_state_dict(state_dict["model_state_dict"], strict=False)
    else:
        policy.load_state_dict(state_dict, strict=False)

    # 5. Play Loop
    print("Starting play loop...")
    obs = env.get_observations()
    
    # Reset environment to get initial state
    # env.reset() # make_env usually resets
    actions = torch.zeros(env.num_envs, 19, device=env.device, requires_grad=False)
    
    
    for i in range(10*int(env.max_episode_length)):
        # Convert raw obs to obs_groups for the policy
        obs_groups = tensor_to_obs_groups(obs, distill_cfg["obs_groups"])
        
        with torch.no_grad():
            # Use act_inference to get student actions
            #print("INFO:", obs_groups)
            actions = policy.act_inference(obs_groups)
        
        obs, _, rews, dones, infos = env.step(actions)
        
        # Optional: Render if supported/enabled in env
        # env.render() 

if __name__ == "__main__":
    args = get_args()
    if getattr(args, "task", None) in (None, "h1_2_fix"):
        args.task = "humanoid_beamdojo"
    if not getattr(args, "proj_name", None):
        args.proj_name = "beamdojo"
        
    play(args)
