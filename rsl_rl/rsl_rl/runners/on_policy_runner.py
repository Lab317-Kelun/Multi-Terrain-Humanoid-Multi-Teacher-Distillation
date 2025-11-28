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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.algorithms import PPOMirror
from rsl_rl.algorithms.ppo_double_reward import PPODoubleReward
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator_multi import AMPDiscriminatorMulti
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
from rsl_rl.utils.normalizer import Normalizer
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        
        # 动态选择policy类
        policy_class_name = self.cfg.get("policy_class_name", "ActorCriticRMADoubleReward")
        if policy_class_name == "ActorCriticRMADoubleReward":
            from rsl_rl.modules.actor_critic import ActorCriticRMADoubleReward
            actor_critic_class = ActorCriticRMADoubleReward
            print(f"Using {policy_class_name} for BEAMDOJO double critic")
        else:
            actor_critic_class = ActorCriticRMA
            print(f"Using default {policy_class_name}")
                    
        if policy_class_name == "ActorCriticRMADoubleReward":
            actor_critic_kwargs = {
                'num_prop': self.env.cfg.env.n_proprio,
                'num_scan': self.env.cfg.env.n_scan,
                'num_critic_obs': self.env.num_obs,
                'num_priv_latent': self.env.cfg.env.n_priv_latent,
                'num_priv_explicit': self.env.cfg.env.n_priv,
                'num_hist': self.env.cfg.env.history_len,
                'num_actions': self.env.num_actions,
                **self.policy_cfg
            }
        else:
            actor_critic_kwargs = {
                'num_actor_obs': self.env.cfg.env.n_proprio,
                'num_critic_obs': self.env.num_obs,
                'num_actions': self.env.num_actions,
                **self.policy_cfg
            }
            
        actor_critic = actor_critic_class(**actor_critic_kwargs).to(self.device)
        #estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        estimator = Estimator(input_dim=env.cfg.env.history_len * env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # Depth encoder
        scan_encoder_type = self.policy_cfg.get("scan_encoder_type", "mlp").lower()
        if scan_encoder_type == "cnn":
            scan_cnn_channels = self.policy_cfg.get("scan_cnn_channels") or []
            scan_encoder_output_dim = scan_cnn_channels[-1] if scan_cnn_channels else self.env.cfg.env.n_scan
        elif scan_encoder_type == "mlp":
            scan_dims = self.policy_cfg.get("scan_encoder_dims") or []
            scan_encoder_output_dim = scan_dims[-1] if scan_dims else self.env.cfg.env.n_scan
        else:
            scan_encoder_output_dim = self.env.cfg.env.n_scan

        if self.policy_cfg.get("scan_encoder_debug", False):
            print("[ScanEncoderDebug] Runner summary -> "
                  f"type: {scan_encoder_type.upper()}, "
                  f"num_prop: {self.env.cfg.env.n_proprio}, "
                  f"num_scan: {self.env.cfg.env.n_scan}, "
                  f"latent_dim: {scan_encoder_output_dim}, "
                  f"critic_obs_dim: {self.env.num_obs}, "
                  f"num_actions: {self.env.num_actions}")

        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                    scan_encoder_output_dim, 
                                                    self.depth_encoder_cfg["hidden_dims"],
                                                    )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None
        # self.depth_encoder = depth_encoder
        # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
        # self.depth_encoder_paras = self.depth_encoder_cfg
        # self.depth_encoder_criterion = nn.MSELoss()
        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        
        discriminator = None
        amp_data = None
        amp_normalizer = None
        if self.alg_cfg.get("use_amp", False):
            num_amp_obs = getattr(self.env.cfg.env, 'num_amp_obs', None)
            num_amp_frames = self.cfg.get("num_amp_frames")
            amp_reward_coef = self.cfg.get("amp_reward_coef")
            amp_discr_hidden_dims = self.cfg.get("amp_discr_hidden_dims")
            amp_task_reward_lerp = self.cfg.get("amp_task_reward_lerp")
            use_lerp = self.cfg.get("use_lerp")
            motion_dir = getattr(self.env.cfg.env, 'amp_motion_files', None)
            amp_loader_type = self.alg_cfg.get("amp_loader_type")
            amp_loader_class_name = self.alg_cfg.get("amp_loader_class_name", "G1_AMPLoader")
            if num_amp_obs is not None and num_amp_frames is not None and motion_dir is not None and amp_loader_type is not None:
                amp_data = eval(amp_loader_class_name)(
                    self.device,
                    time_between_frames=self.env.dt,
                    motion_dir=motion_dir,
                    preload_transitions=True,
                    num_preload_transitions=self.cfg.get("amp_num_preload_transitions"),
                    num_frames=num_amp_frames,
                )
                amp_normalizer = Normalizer(num_amp_obs, self.device)
                discriminator = AMPDiscriminatorMulti(
                    num_amp_obs,
                    amp_reward_coef,
                    amp_discr_hidden_dims,
                    self.device,
                    num_amp_frames,
                    amp_task_reward_lerp,
                    use_lerp,
                )
                # 与AMP多帧runner保持一致：避免无关键传入算法构造
                if "amp_loader_class_name" in self.alg_cfg:
                    del self.alg_cfg["amp_loader_class_name"]
        
        alg_cfg_filtered = dict(self.alg_cfg)
        for k in ["amp_loader_class_name", "amp_loader_type", "use_amp"]:
            if k in alg_cfg_filtered:
                del alg_cfg_filtered[k]

        self.alg: PPO = alg_class(actor_critic, 
                                  estimator=estimator,
                                  estimator_paras=self.estimator_cfg,
                                  depth_encoder=depth_encoder,
                                  depth_encoder_paras=self.depth_encoder_cfg,
                                  depth_actor=depth_actor,
                                  discriminator=discriminator,
                                  amp_data=amp_data,
                                  amp_normalizer=amp_normalizer,
                                  num_amp_frames=self.cfg.get("num_amp_frames"),
                                  amp_loader_type=self.alg_cfg.get("amp_loader_type"),
                                  use_amp=self.alg_cfg.get("use_amp", False),
                                  device=self.device, **alg_cfg_filtered)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        # ===== AMP runner侧缓冲初始化 =====
        # 依据参考实现，若环境提供AMP观测维度且算法配置包含num_amp_frames，则在runner中维护一个循环缓冲
        self.use_amp = getattr(self.alg, 'use_amp', False)
        self.num_amp_frames = getattr(self.alg, 'num_amp_frames', None)
        self.num_amp_obs = getattr(self.env.cfg.env, 'num_amp_obs', None)
        if self.use_amp and self.num_amp_frames is not None and self.num_amp_obs is not None:
            # 形状: (num_envs, num_amp_frames, num_amp_obs)
            self.amp_obs_buffer = torch.zeros(self.env.num_envs, self.num_amp_frames, self.num_amp_obs, device=self.device)
            self.amp_frame_cursor = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        else:
            self.amp_obs_buffer = None
            self.amp_frame_cursor = None

        # 准备init_storage参数
        storage_kwargs = {
            'num_envs': self.env.num_envs,
            'num_transitions_per_env': self.num_steps_per_env,
            'actor_obs_shape': [self.env.num_obs],
            'critic_obs_shape': [self.env.num_privileged_obs],
            'action_shape': [self.env.num_actions],
        }
            
        # 针对不同算法的存储初始化：AMPPPOMulti 需要历史与深度参数
        if type(self.alg).__name__ == 'AMPPPOMulti':
            history_len = getattr(self.env.cfg.env, 'history_len', 0)
            history_dim = getattr(self.env.cfg.env, 'n_proprio', self.env.num_obs)
            depth_shape = None
            depth_buffer_len = None
            if hasattr(self.env.cfg, 'depth') and getattr(self.env.cfg.depth, 'use_camera', False):
                depth_shape = (self.env.cfg.depth.resized[1], self.env.cfg.depth.resized[0])
                depth_buffer_len = self.env.cfg.depth.buffer_len
            self.alg.init_storage(
                storage_kwargs['num_envs'],
                storage_kwargs['num_transitions_per_env'],
                storage_kwargs['actor_obs_shape'],
                storage_kwargs['critic_obs_shape'],
                storage_kwargs['action_shape'],
                history_len=history_len,
                history_dim=history_dim,
                depth_shape=depth_shape,
                depth_buffer_len=depth_buffer_len
            )
        else:
            self.alg.init_storage(**storage_kwargs)

        self.learn = self.learn_RL if not self.if_depth else self.learn_vision
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 根据算法类型适配 act() 签名
                    if type(self.alg).__name__ == 'AMPPPOMulti':
                        history = self.env.get_history_observations().to(self.device) if hasattr(self.env, 'get_history_observations') else None
                        depth_image = None
                        if hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'depth') and getattr(self.env.cfg.depth, 'use_camera', False):
                            depth_image = self.env.depth_buffer.clone().to(self.device)
                        obs_input = (obs, depth_image) if depth_image is not None else obs
                        actions = self.alg.act(obs_input, critic_obs, history)
                    else:
                        actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    # 兼容LeggedRobot.step的扩展返回值
                    step_outputs = self.env.step(actions)
                    if isinstance(step_outputs, tuple) and len(step_outputs) >= 5:
                        obs, privileged_obs, rewards, dones, infos = step_outputs[:5]
                        # 如果环境返回终止相关的AMP观测，则拼接到amp缓冲
                        termination_ids = step_outputs[5] if len(step_outputs) > 5 else None
                        termination_privileged_obs = step_outputs[6] if len(step_outputs) > 6 else None
                    else:
                        obs, privileged_obs, rewards, dones, infos = step_outputs
                        termination_ids, termination_privileged_obs = None, None
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    infos = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in infos.items()}
                    rewards = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in rewards.items()} if isinstance(rewards, dict) else rewards.to(self.device)
                    obs, critic_obs, dones = obs.to(self.device), critic_obs.to(self.device), dones.to(self.device)

                    # ===== AMP帧收集：参考g1_16dof_moe_residual_env.get_amp_observations =====
                    amp_obs_frames = None
                    if self.use_amp and self.amp_obs_buffer is not None:
                        # 获取当前帧AMP观测：若环境实现了get_amp_observations()
                        if hasattr(self.env, 'get_amp_observations'):
                            cur_amp_obs = self.env.get_amp_observations().to(self.device)
                        elif hasattr(self.env, 'env') and hasattr(self.env.env, 'get_amp_observations'):
                            # 某些VecEnv包装器可能在self.env.env下持有真实环境
                            cur_amp_obs = self.env.env.get_amp_observations().to(self.device)
                        else:
                            cur_amp_obs = None
                        if cur_amp_obs is not None and cur_amp_obs.shape[-1] == self.num_amp_obs:
                            # 将当前帧写入循环缓冲：左移，最新帧放在最后
                            self.amp_obs_buffer = torch.cat([self.amp_obs_buffer[:, 1:], cur_amp_obs.unsqueeze(1)], dim=1)
                            amp_obs_frames = self.amp_obs_buffer.clone()
                        else:
                            amp_obs_frames = None

                    # 根据算法类型适配 process_env_step() 签名
                    if type(self.alg).__name__ == 'AMPPPOMulti':
                        # AMPPPOMulti 需要 next_obs/next_critic_obs，且不返回值；用于日志的总奖励自行计算
                        if isinstance(rewards, dict):
                            total_rew = rewards.get('total', rewards.get('dense', 0)) + rewards.get('sparse', 0)
                            if not isinstance(total_rew, torch.Tensor):
                                total_rew = torch.tensor(total_rew, device=self.device).unsqueeze(1).repeat(self.env.num_envs, 1)
                        else:
                            total_rew = rewards
                        self.alg.process_env_step(total_rew, dones, infos, next_obs=obs, next_critic_obs=critic_obs, amp_obs_frames=amp_obs_frames)
                    else:
                        total_rew = self.alg.process_env_step(rewards, dones, infos, amp_obs_frames=amp_obs_frames)

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # 根据算法类型适配 compute_returns() 签名
                if type(self.alg).__name__ == 'AMPPPOMulti':
                    history = self.env.get_history_observations().to(self.device) if hasattr(self.env, 'get_history_observations') else None
                    self.alg.compute_returns(critic_obs, history)
                else:
                    self.alg.compute_returns(critic_obs)

            if self.log_dir is not None:
                # Book keeping
                if 'episode' in infos:
                    ep_infos.append(infos['episode'])
                cur_reward_sum += total_rew
                cur_reward_explr_sum += 0
                cur_reward_entropy_sum += 0
                cur_episode_length += 1
                
                new_ids = (dones > 0).nonzero(as_tuple=False)
                
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                
                cur_reward_sum[new_ids] = 0
                cur_reward_explr_sum[new_ids] = 0
                cur_reward_entropy_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            # Learning step - 适配不同的算法返回值
            if self.alg.use_double_critic:
                mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef, mean_value_loss_dense, mean_value_loss_sparse = self.alg.update()
            else:
                mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        num_pretrain_iter = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_buffer_student = []
            yaw_buffer_teacher = []
            delta_yaw_ok_buffer = []
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                if infos["depth"] != None:
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                    
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = 1.5*depth_latent_and_yaw[:, -2:]
                    
                    depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(yaw)
                    yaw_buffer_teacher.append(obs[:, 6:8])
                
                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = obs.clone()
                # obs_student[:, 6:8] = yaw.detach()
                obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
                actions_student = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # detach actions before feeding the env
                if it < num_pretrain_iter:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_teacher.detach())  # obs has changed to next_obs !! if done obs has been reset
                else:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())  # obs has changed to next_obs !! if done obs has been reset
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
            stop = time.time()
            collection_time = stop - start
            start = stop

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            depth_encoder_loss = 0
            # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)
            depth_actor_loss, yaw_loss = self.alg.update_depth_actor(actions_student_buffer, actions_teacher_buffer, yaw_buffer_student, yaw_buffer_teacher)
            
            # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_hidden_states()

            if self.log_dir is not None:
                self.log_vision(locals())
            if (it-self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
               (it-self.start_learning_iteration < 5000 and it % (2*self.save_interval) == 0) or \
               (it-self.start_learning_iteration >= 5000 and it % (5*self.save_interval) == 0):
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
    
    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        if hasattr(self.env, 'total_times') and hasattr(self.env, 'success_times') and hasattr(self.env, 'complete_times'):
            if self.env.total_times > 0:
                # 获取成功率计算模式配置
                curriculum_config = getattr(self.env.cfg, 'curriculum_config', object())
                success_rate_mode = getattr(curriculum_config, 'success_rate_mode', 'survival_time')
                
                # 根据配置选择成功率计算方式
                if success_rate_mode == 'survival_time':
                    # 方式1: 基于存活时间计算成功率
                    if len(locs.get('lenbuffer', [])) > 0:
                        mean_ep_len_steps = statistics.mean(locs['lenbuffer'])
                        survival_time = float(mean_ep_len_steps) * float(getattr(self.env, 'dt', 0.02))
                        threshold = float(getattr(curriculum_config, 'survival_time_threshold', getattr(self.env, 'max_episode_length_s', 1.0)))
                        success_rate = survival_time / threshold if threshold > 0 else 0.0
                    else:
                        # 如果没有episode长度数据,回退到目标计算方式
                        success_rate = self.env.success_times / self.env.total_times
                elif success_rate_mode == 'goal_based':
                    # 方式2: 基于目标完成度计算成功率
                    success_rate = self.env.success_times / self.env.total_times
                else:
                    # 未知模式,使用默认(目标模式)
                    print(f"Warning: Unknown success_rate_mode '{success_rate_mode}', using 'goal_based'")
                    success_rate = self.env.success_times / self.env.total_times
                
                completion_rate = self.env.complete_times / self.env.total_times
                wandb_dict['Episode_rew/success_rate'] = success_rate
                wandb_dict['Episode_rew/completion_rate'] = completion_rate
                wandb_dict['Episode_rew/terrain_level'] = torch.mean(self.env.terrain_levels.float()) if hasattr(self.env, 'terrain_levels') else 0
        
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        
        # Discriminator loss
        if 'mean_discriminator_loss' in locs:
            wandb_dict['Loss/discriminator'] = locs['mean_discriminator_loss']
            wandb_dict['Loss/discriminator_accuracy'] = locs['mean_discriminator_acc']
        elif 'mean_disc_loss' in locs:
            wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
            wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']
        
        # 双Critic的额外loss
        if 'mean_value_loss_dense' in locs:
            wandb_dict['Loss/value_function_dense'] = locs['mean_value_loss_dense']
        if 'mean_value_loss_sparse' in locs:
            wandb_dict['Loss/value_function_sparse'] = locs['mean_value_loss_sparse']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            
            # 添加自定义参数记录示例
            # 你可以在这里添加任何你想记录的参数
            # wandb_dict['Custom/terrain_complexity'] = self.env.get_terrain_complexity_mean() if hasattr(self.env, 'get_terrain_complexity_mean') else 0
            # wandb_dict['Custom/gap_success_rate'] = self.env.get_gap_success_rate() if hasattr(self.env, 'get_gap_success_rate') else 0
            # wandb_dict['Custom/foot_contact_frequency'] = self.env.get_foot_contact_frequency() if hasattr(self.env, 'get_foot_contact_frequency') else 0
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        disc_loss_print = locs['mean_discriminator_loss'] if 'mean_discriminator_loss' in locs else locs.get('mean_disc_loss', 0.0)
        disc_acc_print = locs['mean_discriminator_acc'] if 'mean_discriminator_acc' in locs else locs.get('mean_disc_acc', 0.0)

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {disc_loss_print:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {disc_acc_print:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {disc_loss_print:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {disc_acc_print:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        
        # 保存历史编码器优化器
        if hasattr(self.alg, 'hist_encoder_optimizer') and self.alg.hist_encoder_optimizer is not None:
            state_dict['hist_encoder_optimizer_state_dict'] = self.alg.hist_encoder_optimizer.state_dict()
        
        # 保存Estimator优化器
        if hasattr(self.alg, 'estimator_optimizer') and self.alg.estimator_optimizer is not None:
            state_dict['estimator_optimizer_state_dict'] = self.alg.estimator_optimizer.state_dict()
        
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        
        # AMP 判别器权重与优化器
        if getattr(self.alg, 'use_amp', False) and getattr(self.alg, 'discriminator', None) is not None:
            state_dict['discriminator_state_dict'] = self.alg.discriminator.state_dict()
            if hasattr(self.alg, 'optimizer_disc') and self.alg.optimizer_disc is not None:
                state_dict['optimizer_disc_state_dict'] = self.alg.optimizer_disc.state_dict()
                print("[Save] AMP启用：已保存判别器权重与优化器状态")
            elif hasattr(self.alg, 'disc_optimizer') and self.alg.disc_optimizer is not None:
                # 兼容旧字段名
                state_dict['optimizer_disc_state_dict'] = self.alg.disc_optimizer.state_dict()
                print("[Save] AMP启用：检测到旧优化器字段名，已保存判别器优化器状态")
        else:
            print("[Save] AMP未启用或未检测到判别器：不保存判别器相关状态")
        
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        
        # 加载actor_critic，允许缺少某些键（如discriminator）
        model_state = loaded_dict['model_state_dict']
        current_model_state = self.alg.actor_critic.state_dict()
        
        # 检查哪些键缺失
        missing_keys = set(current_model_state.keys()) - set(model_state.keys())
        unexpected_keys = set(model_state.keys()) - set(current_model_state.keys())
        
        if missing_keys:
            print(f"[Load] 警告: 模型中有但checkpoint中缺失的键: {missing_keys}")
            print(f"[Load] 这些键将保持初始化值（例如：discriminator将重新开始训练）")
        
        if unexpected_keys:
            print(f"[Load] 警告: checkpoint中有但模型中不存在的键（将被忽略）: {unexpected_keys}")
        
        # 使用strict=False允许缺少某些键
        load_result = self.alg.actor_critic.load_state_dict(model_state, strict=False)
        
        if load_result.missing_keys:
            print(f"[Load] 实际缺失的键: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"[Load] 实际多余的键: {load_result.unexpected_keys}")
        
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        
        # 加载优化器（包括所有优化器）
        if load_optimizer:
            try:
                # 加载主优化器（Actor-Critic）
                self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
                print("[Load] 成功加载主优化器状态（Actor-Critic）")
            except (ValueError, KeyError) as e:
                print(f"[Load] 警告: 无法加载主优化器状态: {e}")
                print("[Load] 优化器将使用初始状态")
            
            # 加载历史编码器优化器
            if hasattr(self.alg, 'hist_encoder_optimizer') and self.alg.hist_encoder_optimizer is not None:
                if 'hist_encoder_optimizer_state_dict' in loaded_dict:
                    try:
                        self.alg.hist_encoder_optimizer.load_state_dict(loaded_dict['hist_encoder_optimizer_state_dict'])
                        print("[Load] 成功加载历史编码器优化器状态")
                    except (ValueError, KeyError) as e:
                        print(f"[Load] 警告: 无法加载历史编码器优化器状态: {e}")
                else:
                    print("[Load] checkpoint中没有历史编码器优化器状态，将使用初始状态")
            
            # 加载Estimator优化器
            if hasattr(self.alg, 'estimator_optimizer') and self.alg.estimator_optimizer is not None:
                if 'estimator_optimizer_state_dict' in loaded_dict:
                    try:
                        self.alg.estimator_optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
                        print("[Load] 成功加载Estimator优化器状态")
                    except (ValueError, KeyError) as e:
                        print(f"[Load] 警告: 无法加载Estimator优化器状态: {e}")
                else:
                    print("[Load] checkpoint中没有Estimator优化器状态，将使用初始状态")

        # AMP 判别器加载逻辑
        if getattr(self.alg, 'use_amp', False):
            if getattr(self.alg, 'discriminator', None) is not None:
                print("[Load] AMP启用：尝试加载判别器状态")
                if 'discriminator_state_dict' in loaded_dict:
                    try:
                        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
                        print("[Load] 成功加载判别器权重")
                    except (ValueError, KeyError) as e:
                        print(f"[Load] 警告: 无法加载判别器权重: {e}")
                else:
                    print("[Load] checkpoint中不存在判别器权重，保持当前初始化状态")

                if load_optimizer:
                    opt_loaded = False
                    if 'optimizer_disc_state_dict' in loaded_dict:
                        try:
                            if hasattr(self.alg, 'optimizer_disc') and self.alg.optimizer_disc is not None:
                                self.alg.optimizer_disc.load_state_dict(loaded_dict['optimizer_disc_state_dict'])
                                print("[Load] 成功加载判别器优化器状态")
                                opt_loaded = True
                            elif hasattr(self.alg, 'disc_optimizer') and self.alg.disc_optimizer is not None:
                                self.alg.disc_optimizer.load_state_dict(loaded_dict['optimizer_disc_state_dict'])
                                print("[Load] 成功加载判别器优化器状态（兼容旧字段）")
                                opt_loaded = True
                        except (ValueError, KeyError) as e:
                            print(f"[Load] 警告: 无法加载判别器优化器状态: {e}")
                    if not opt_loaded and 'disc_optimizer_state_dict' in loaded_dict:
                        try:
                            if hasattr(self.alg, 'optimizer_disc') and self.alg.optimizer_disc is not None:
                                self.alg.optimizer_disc.load_state_dict(loaded_dict['disc_optimizer_state_dict'])
                                print("[Load] 成功加载判别器优化器状态（来自旧key）")
                                opt_loaded = True
                            elif hasattr(self.alg, 'disc_optimizer') and self.alg.disc_optimizer is not None:
                                self.alg.disc_optimizer.load_state_dict(loaded_dict['disc_optimizer_state_dict'])
                                print("[Load] 成功加载判别器优化器状态（旧字段与旧key）")
                                opt_loaded = True
                        except (ValueError, KeyError) as e:
                            print(f"[Load] 警告: 无法加载判别器优化器状态（旧key）: {e}")
                    if not opt_loaded:
                        print("[Load] checkpoint中没有判别器优化器状态，将使用初始状态")
            else:
                print("[Load] AMP启用但未检测到判别器实例，跳过判别器加载")
        else:
            if 'discriminator_state_dict' in loaded_dict or 'optimizer_disc_state_dict' in loaded_dict or 'disc_optimizer_state_dict' in loaded_dict:
                print("[Load] AMP未启用：检测到判别器相关状态但将跳过加载")
        
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder
    
