from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch
import h5py
from datetime import datetime
import faulthandler
from terrain_base.config import terrain_config

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def record_replay(args):
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1000
    env_cfg.commands.resampling_time = 60
    env_cfg.rewards.is_play = False

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = 1

    env_cfg.terrain.height = [0.01, 0.02]
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False


    depth_latent_buffer = []
    # prepare environment
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    actions = torch.zeros(env.num_envs, 19, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    # for i in range(10*int(env.max_episode_length)):
    if not args.replay:
        for i in range(3*int(env.max_episode_length)): #dt =  0.02, max_episode_length_s = 20, then there must be 1000
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            if(i % 1000 == 0):
                print(f"current step:{i}")
    else: # replay the dataset directly
        load_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'datasets', 'Apr15_14-29-52_h1_2_terrain.h5')
        with h5py.File(load_path, "r") as f:
            all_actions = []
            max_length = 0 # only for replay
            for env_id in range(env.num_envs):
                actions_data = f[f'env_{env_id}']['episode_0']['actions'][:]
                actions_data = actions_data.astype(np.float32)
                actions_data = actions_data[:, :12]
                all_actions.append(actions_data)
                if actions_data.shape[0] > max_length:
                    max_length = actions_data.shape[0]

            padded_actions = []
            for actions_data in all_actions:
                pad_length = max_length - actions_data.shape[0]
                padded_data = np.pad(actions_data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
                padded_actions.append(padded_data)
            all_actions = np.array(padded_actions)

        for i in range(int(env.max_episode_length)):
            actions = all_actions[:, i, :]
            actions = torch.tensor(actions, device=env.device, dtype=torch.float32)
            obs, _, rews, dones, infos = env.step(actions)
            if i < all_actions.shape[1] - 1:
                actions = all_actions[:, i + 1, :]
            else:
                actions = all_actions[:, -1, :]

    if args.save:
        save_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'flat', 
                        f"{datetime.now().strftime('%b%d_%H-%M-%S')}_h1_2_flat.h5")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with h5py.File(save_path, "w") as f:
            for env_id in range(env.num_envs):
                # 获取该环境的所有episode
                episodes = env.episode_data['observations'][env_id]

                if len(episodes) == 0 :
                    continue
                
                # 创建环境组
                env_group = f.create_group(f"env_{env_id}")
                
                # 保存每个episode
                for ep_idx, (obs, act, rew, hei, body, rews, dof) in enumerate(zip(
                    env.episode_data['observations'][env_id],
                    env.episode_data['actions'][env_id],
                    env.episode_data['rewards'][env_id],
                    env.episode_data['height_map'][env_id],
                    env.episode_data['rigid_body_state'][env_id],
                    env.episode_data['dof_state'][env_id],
                )):
                    ep_group = env_group.create_group(f"episode_{ep_idx}")
                    ep_group.create_dataset("observations", data=obs, compression="gzip")
                    ep_group.create_dataset("actions", data=act, compression="gzip")
                    ep_group.create_dataset("rewards", data=rew, compression="gzip")
                    ep_group.create_dataset("height_map", data=hei, compression="gzip")
                    ep_group.create_dataset("rigid_body_state", data=body, compression="gzip")
                    ep_group.create_dataset("dof_state", data=dof, compression="gzip")
                    
                    # 保存privileged观测（如果存在）
                    if len(env.episode_data['privileged_obs'][env_id]) > ep_idx:
                        priv_obs = env.episode_data['privileged_obs'][env_id][ep_idx]
                        ep_group.create_dataset("privileged_observations", data=priv_obs, compression="gzip")
        
        print(f"dataset save path: {save_path}")
        print("dataset informations: ", env.get_data_stats())
        
if __name__ == '__main__':
    args = get_args()
    record_replay(args)