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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args,  task_registry
from terrain_base.config import terrain_config

import torch
import faulthandler

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 6
    env_cfg.commands.resampling_time = 6
    env_cfg.rewards.is_play = True

    env_cfg.curriculum_config.success_mode = 'survival_time'
    env_cfg.curriculum_config.survival_time_threshold = 6.0
    env_cfg.curriculum_config.survival_success_threshold = 1  # 连续存活成功次数阈值
    env_cfg.curriculum_config.survival_failure_threshold = 2   # 连续存活失败次数阈值

    env_cfg.commands.ranges.lin_vel_x = [0.5, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.0, 0.0]
    env_cfg.commands.ranges.heading = [-0.0, 0.0]
    env_cfg.commands.ranges.height = [-0.0, 0.0]
    
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 0

    env_cfg.terrain.height = [0.00, 0.02]
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = True

    depth_latent_buffer = []
    # prepare environment
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 19, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    # 获取配置参数
    n_proprio = env.cfg.env.n_proprio
    n_scan = env.cfg.env.n_scan
    history_len = env.cfg.env.history_len
    n_priv_explicit = env.cfg.env.n_priv  # priv_explicit 维度

    for i in range(10*int(env.max_episode_length)):
        # 使用 estimator 估计隐式特权信息
        # 历史观测位于观测的最后 history_len * n_proprio 维
        hist_obs = obs[:, -history_len * n_proprio:]
        # 使用 estimator 估计 priv_explicit
        priv_explicit_estimated = estimator(hist_obs)
        # 将估计的 priv_explicit 替换到观测中的相应位置
        # priv_explicit 的位置：n_proprio + n_scan 到 n_proprio + n_scan + n_priv_explicit
        obs[:, n_proprio + n_scan : n_proprio + n_scan + n_priv_explicit] = priv_explicit_estimated
       
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5*yaw
                
        else:
            depth_latent = None
        
        if hasattr(ppo_runner.alg, "depth_actor"):
            actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
