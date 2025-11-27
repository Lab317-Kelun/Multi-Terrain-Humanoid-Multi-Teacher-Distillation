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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask

from terrain_base.terrain import Terrain
from terrain_base.config import terrain_config

from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from .legged_robot_config import LeggedRobotCfg

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


class HumanoidRobot(BaseTask):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, save):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self.save = save
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        self.num_lower_dof = self.cfg.env.num_actions
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
    
        if self.save:
            self.episode_data = {
                'observations': [[] for _ in range(self.num_envs)],
                'actions': [[] for _ in range(self.num_envs)],
                'rewards': [[] for _ in range(self.num_envs)],
                'height_map': [[] for _ in range(self.num_envs)],
                'privileged_obs': [[] for _ in range(self.num_envs)],
                'rigid_body_state': [[] for _ in range(self.num_envs)],
                'dof_state': [[] for _ in range(self.num_envs)]
            }
            self.current_episode_buffer = {
                'observations': [[] for _ in range(self.num_envs)],
                'actions': [[] for _ in range(self.num_envs)],
                'rewards': [[] for _ in range(self.num_envs)],
                'height_map': [[] for _ in range(self.num_envs)],
                'privileged_obs': [[] for _ in range(self.num_envs)],
                'rigid_body_state': [[] for _ in range(self.num_envs)],
                'dof_state': [[] for _ in range(self.num_envs)]
            }
        # init data save buffer
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.time_stamp = 0

        self.total_times = 0
        self.last_times = -1
        self.success_times = 0
        self.complete_times = 0.

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def get_data_stats(self):
        """get dataset information"""
        stats = {
            'total_episodes': 0,
            'total_samples': 0,
            'avg_episode_length': 0
        }
        for env_data in self.episode_data['observations']:
            stats['total_episodes'] += len(env_data)
            for ep in env_data:
                stats['total_samples'] += ep.shape[0]
        if stats['total_episodes'] > 0:
            stats['avg_episode_length'] = stats['total_samples'] / stats['total_episodes']
        return stats

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions.to(self.device)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.origin_actions[:] = self.actions[:]
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None

        if self.save:
            for env_idx in range(self.num_envs):
                self.current_episode_buffer['observations'][env_idx].append(
                    self.obs_buf[env_idx].cpu().numpy().copy())  
                self.current_episode_buffer['actions'][env_idx].append(
                    self.actions[env_idx].cpu().numpy().copy())      
                
                self.current_episode_buffer['rewards'][env_idx].append(
                    self.rew_buf[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['height_map'][env_idx].append(
                    self.measured_heights_data[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['rigid_body_state'][env_idx].append(
                    self.rigid_body_states[env_idx].cpu().numpy().copy()) 
                
                self.current_episode_buffer['dof_state'][env_idx].append(
                    self.dof_state[env_idx].cpu().numpy().copy())  

                if self.privileged_obs_buf is not None:
                    self.current_episode_buffer['privileged_obs'][env_idx].append(
                        self.privileged_obs_buf[env_idx].cpu().numpy().copy())      

        if(self.cfg.rewards.is_play):
            if(self.total_times > 0):
                if(self.total_times > self.last_times):
                    print("total_times=",self.total_times)
                    print("success_rate=",self.success_times / self.total_times)
                    print("complete_rate=",(self.complete_times / self.total_times).cpu().numpy().copy())
                    self.last_times = self.total_times
                    

        if self.cfg.env.use_double_critic:
            rewards = {
                'dense': self.dense_rew_buf,
                'sparse': self.sparse_rew_buf
            }
        else:
            rewards = self.rew_buf

        return self.obs_buf, self.privileged_obs_buf, rewards, self.reset_buf, self.extras

    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var = self._get_feet_heights()
        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)
        
        # self._update_jump_schedule()
        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.update_depth_buffer()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if(self.time_stamp ==5):
            self.last_foot_action = self.rigid_body_states[:, self.feet_indices, :]
            self.time_stamp=0
        else :
            self.time_stamp=self.time_stamp+1
        
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_height_samples()
            self._draw_goals()
            self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def reindex(self, vec):
        return vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def check_termination(self):
        """ Check if environments need to be reset"""
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 0.8
        pitch_cutoff = torch.abs(self.pitch) > 0.8
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < 0.5
        
        # 检查机器人是否超出地形边界
        # length = (self.cfg.terrain.terrain_length / 2) - 0.2
        # width = (self.cfg.terrain.terrain_width - 1) / 2 - 0.2
        # length = self.cfg.terrain.terrain_length- 0.2
        # width = self.cfg.terrain.terrain_width - 0.2
        # relative_pos = self.root_states[:, :2] - self.env_origins[:, :2]
        # x_out_of_bounds = (relative_pos[:, 0] < -length) | (relative_pos[:, 0] > length) 
        # y_out_of_bounds = (relative_pos[:, 1] < -width) | (relative_pos[:, 1] > width)
        
        # boundary_cutoff = x_out_of_bounds | y_out_of_bounds

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= reach_goal_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        # self.reset_buf |= boundary_cutoff  # 超出地形边界也终止

        self.total_times += len(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.success_times += len(reach_goal_cutoff.nonzero(as_tuple=False).flatten())
        self.complete_times += (self.cur_goal_idx[self.reset_buf.nonzero(as_tuple=False).flatten()] / self.cfg.terrain.num_goals).sum()

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        if self.save:
            for env_id in env_ids:
                try:
                    if len(self.current_episode_buffer['observations'][env_id]) > 750:
                        # 转换为numpy数组
                        episode_obs = np.stack(self.current_episode_buffer['observations'][env_id])  # [T,*]
                        episode_act = np.stack(self.current_episode_buffer['actions'][env_id])       # [T,*]
                        episode_rew = np.stack(self.current_episode_buffer['rewards'][env_id])      # [T]
                        episode_hei = np.stack(self.current_episode_buffer['height_map'][env_id])      # [T, 396]
                        episode_body = np.stack(self.current_episode_buffer['rigid_body_state'][env_id]) # [T,13,13] first is root
                        episode_dof = np.stack(self.current_episode_buffer['dof_state'][env_id])
                      
                        # 存入主数据存储
                        self.episode_data['observations'][env_id].append(episode_obs)
                        self.episode_data['actions'][env_id].append(episode_act)
                        self.episode_data['rewards'][env_id].append(episode_rew)
                        self.episode_data['height_map'][env_id].append(episode_hei)
                        self.episode_data['rigid_body_state'][env_id].append(episode_body)
                        self.episode_data['dof_state'][env_id].append(episode_dof)

                        
                        # 处理privileged观测
                        if self.privileged_obs_buf is not None:
                            episode_priv = np.stack(self.current_episode_buffer['privileged_obs'][env_id]) # [T,*]
                            self.episode_data['privileged_obs'][env_id].append(episode_priv)
                        
                        # 清空当前buffer
                        self.current_episode_buffer['observations'][env_id] = []
                        self.current_episode_buffer['actions'][env_id] = []
                        self.current_episode_buffer['rewards'][env_id] = []
                        self.current_episode_buffer['height_map'][env_id] = []
                        self.current_episode_buffer['privileged_obs'][env_id] = []
                        self.current_episode_buffer['rigid_body_state'][env_id] = []
                        self.current_episode_buffer['dof_state'][env_id] = []
                        
                        print(f"Env {env_id} have saved {episode_obs.shape[0]} step data")
                except Exception as e:
                    print(f"An error occured when saving env {env_id}: {str(e)}")
        
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_foot_action[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        
        # reset goal distance tracking for reach_goal reward
        distance_to_goal = torch.norm(self.root_states[env_ids, :2] - self.cur_goals[env_ids, :2], dim=1)
        self.last_distance_to_goal[env_ids] = distance_to_goal

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        
        if self.cfg.env.use_double_critic:
            # 初始化密集和稀疏奖励缓冲区
            if not hasattr(self, 'dense_rew_buf'):
                self.dense_rew_buf = torch.zeros_like(self.rew_buf)
                self.sparse_rew_buf = torch.zeros_like(self.rew_buf)
            
            self.dense_rew_buf[:] = 0.
            self.sparse_rew_buf[:] = 0.
            
            # 获取稠密和稀疏奖励列表
            reward_config = getattr(self.cfg, 'reward_config', None)
            if reward_config:
                dense_rewards = getattr(reward_config, 'dense_rewards', [])
                sparse_rewards = getattr(reward_config, 'sparse_rewards', [])
            
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew = self.reward_functions[i]() * self.reward_scales[name]
                if torch.isnan(rew).any():
                    import ipdb; ipdb.set_trace()
                
                # 根据奖励类型分配到不同的缓冲区
                if name in sparse_rewards:
                    self.sparse_rew_buf += rew
                elif name in dense_rewards or (len(dense_rewards) == 0 and len(sparse_rewards) == 0):
                    # 如果在dense_rewards列表中，或者两个列表都为空（默认所有奖励为稠密）
                    self.dense_rew_buf += rew
                else:
                    # 如果dense_rewards不为空但当前奖励不在列表中，默认为稠密奖励
                    self.dense_rew_buf += rew
                
                self.rew_buf += rew
                self.episode_sums[name] += rew
                
            # 处理termination奖励
            if "termination" in self.reward_scales:
                rew = self._reward_termination() * self.reward_scales["termination"]
                self.dense_rew_buf += rew  # termination通常是密集奖励
                self.rew_buf += rew
                self.episode_sums["termination"] += rew
                
            if self.cfg.rewards.only_positive_rewards:
                self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
                self.dense_rew_buf[:] = torch.clip(self.dense_rew_buf[:], min=0.)
                # self.sparse_rew_buf[:] = torch.clip(self.sparse_rew_buf[:], min=0.)
        else:
            # 原有的单一奖励逻辑
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew = self.reward_functions[i]() * self.reward_scales[name]
                if torch.isnan(rew).any():
                    import ipdb; ipdb.set_trace()
                self.rew_buf += rew
                self.episode_sums[name] += rew
            if self.cfg.rewards.only_positive_rewards:
                self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            # add termination reward after clipping
            if "termination" in self.reward_scales:
                rew = self._reward_termination() * self.reward_scales["termination"]
                self.rew_buf += rew
                self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.delta_yaw = self.target_yaw - self.yaw
        self.delta_next_yaw = self.next_target_yaw - self.yaw
        
        # if self.global_counter % 5 == 0:
        #     # 添加调试信息
        #     # print("Robot position:", self.root_states[0, :2])  # 机器人位置 - 世界坐标系
        #     # print("Env origin:", self.env_origins[0, :2])      # 环境原点 - 世界坐标系
        #     # print("Base init state:", self.base_init_state[:2]) # 基础初始状态 - 相对环境原点坐标系
        #     # print("Current goal (relative):", self.cur_goals[0, :2])      # 当前目标点 - 相对环境原点坐标系
        #     # print("Next goal (relative):", self.next_goals[0, :2])        # 下一个目标点 - 相对环境原点坐标系
        #     print("Current goal (world):", self.cur_goals[0, :2] + self.env_origins[0, :2])      # 当前目标点 - 世界坐标系
        #     print("Next goal (world):", self.next_goals[0, :2] + self.env_origins[0, :2])        # 下一个目标点 - 世界坐标系
        #     # print("Target pos rel:", self.target_pos_rel[0])   # 相对位置向量 - 机器人本体坐标系
        #     print("Robot yaw:", self.yaw[0])                   # 机器人当前朝向 - 世界坐标系
        #     print("Target yaw:", self.target_yaw[0])           # 目标朝向 - 世界坐标系
        #     print("self.delta_yaw=",self.delta_yaw[0])
        #     print("self.delta_next_yaw=",self.delta_next_yaw[0]) 
            
        #     print("######################################################################")
            
        #     # 添加速度和指令信息
        #     print("Robot linear velocity:", self.base_lin_vel[0])  # 机器人线速度 - 机器人本体坐标系
        #     print("Robot angular velocity:", self.base_ang_vel[0])  # 机器人角速度 - 机器人本体坐标系
        #     print("Linear velocity command X:", self.commands[0, 0])  # X方向线速度指令 - 机器人本体坐标系
        #     print("Angular velocity command Yaw:", self.commands[0, 2])  # Z轴角速度指令 - 机器人本体坐标系
        #     print("Heading command:", self.commands[0, 3])  # 朝向指令 - 世界坐标系
            
        noisy_dof_pos = self.get_noisy_measurement(
            self.dof_pos - self.default_dof_pos_all, 
            self.cfg.noise.noise_scales.dof_pos
        )
        noisy_dof_vel = self.get_noisy_measurement(
            self.dof_vel, 
            self.cfg.noise.noise_scales.dof_vel
        )
        noisy_ang_vel = self.get_noisy_measurement(
            self.base_ang_vel, 
            self.cfg.noise.noise_scales.ang_vel
        )
        noisy_gravity = self.get_noisy_measurement(
            self.projected_gravity, 
            self.cfg.noise.noise_scales.gravity
        )
        noisy_dof_pos = noisy_dof_pos * self.obs_scales.dof_pos
        noisy_dof_vel = noisy_dof_vel * self.obs_scales.dof_vel
        noisy_ang_vel = noisy_ang_vel * self.obs_scales.ang_vel
        noisy_commands = self.commands[:, 0:3] * self.commands_scale

        # print(f"noisy_ang_vel: {noisy_ang_vel}")
        print(f"self.commands[:, 0:3]: {self.commands[:, 0:3]}")
        
        
        
        obs_buf = torch.cat((
                            #skill_vector, 
                            # self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3] # 3
                            # imu_obs,    #[1,2]  2 只包含roll和pitch
                            # 0*self.delta_yaw[:, None], # 1
                            # self.delta_yaw[:, None], # 1
                            # self.delta_next_yaw[:, None],  # 1
                            # 0*self.commands[:, 0:2],  # 2
                            # self.commands[:, 0:1],  #[1,1]  # 1
                            # (self.env_class != 17).float()[:, None],  #1
                            # (self.env_class == 17).float()[:, None], # 1
                            # (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos, # 12
                            # self.dof_vel * self.obs_scales.dof_vel,  # 12
                            # self.action_history_buf[:, -1], # 12
                            # self.contact_filt.float()-0.5, # 2
                            noisy_commands,   #3 x y yaw
                            noisy_ang_vel,           # R^3 (带噪声的角速度)
                            noisy_gravity,           # R^3 (带噪声的重力)
                            noisy_dof_pos,           # R^{n_dof} (带噪声的关节位置)
                            noisy_dof_vel,           # R^{n_dof} (带噪声的关节速度)
                            self.action_history_buf[:, -1, :12], # R^{12}
                            ), dim=-1)
        
        priv_explicit = self.base_lin_vel * self.obs_scales.lin_vel
        
        priv_latent = torch.cat((
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0][:, :12] - 1, 
            self.motor_strength[1][:, :12] - 1
        ), dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = self.root_states[:, 2].unsqueeze(1) - self.measured_heights
            self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )
            
    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            x = x + (2.0 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
        return x

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        start = time()
        print("*"*80)
        mesh_type = terrain_config.mesh_type

        if mesh_type=='None':
            self._create_ground_plane()
        else:
            self.terrain = Terrain(self.num_envs)
            self._create_trimesh()

        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        if self.cfg.terrain.measure_heights:
            # if self.global_counter % self.cfg.depth.update_interval == 0:
            self.measured_heights, self.measured_heights_data  = self._get_heights()
        
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0)
        self._resample_commands(env_ids.nonzero(as_tuple=False).flatten())
        
        if self.cfg.commands.heading_command:
            self.commands[:, 3] = self.target_yaw
            yaw_error = wrap_to_pi(self.commands[:, 3] - self.yaw)
            ang_vel_cmd = 0.8 * yaw_error
            small_command_mask = torch.abs(ang_vel_cmd) <= self.cfg.commands.ang_vel_clip
            self.commands[:, 2] = torch.where(small_command_mask, 
                                            torch.zeros_like(ang_vel_cmd), 
                                            ang_vel_cmd)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)
    
    def _get_forward_height_gradient(self):
        """计算机器人前方的高度梯度，用于判断坡度"""
        front_x_indices = [3, 4, 5, 6]  # x = 0, 0.15, 0.3, 0.45 的索引
        front_point_indices = []
        for x_idx in front_x_indices:
            for y_idx in range(11):  # 所有y方向
                front_point_indices.append(x_idx * 11 + y_idx)

        forward_heights = self.measured_heights[:, front_point_indices]
        # 计算一阶差分
        gradients = torch.diff(forward_heights, n=1, dim=1)
        avg_gradient = torch.mean(gradients, dim=1)
        return avg_gradient  # 返回每个环境的平均坡度指标

    def _analyze_terrain_complexity(self):
        """分析前方地形复杂度"""
        # 提取前方高度采样点（机器人前方0-1.2米区域）
        front_x_indices = [3, 4, 5, 6]  # x = 0, 0.15, 0.3, 0.45 的索引
        front_point_indices = []
        for x_idx in front_x_indices:
            for y_idx in range(11):  # 所有y方向
                front_point_indices.append(x_idx * 11 + y_idx)

        forward_heights = self.measured_heights[:, front_point_indices]
        # 计算地形复杂度指标
        height_variance = torch.var(forward_heights, dim=1)      # 高度方差（起伏程度）
        height_gradient = torch.max(forward_heights, dim=1)[0] - torch.min(forward_heights, dim=1)[0]  # 高度差
        height_roughness = torch.mean(torch.abs(torch.diff(forward_heights, dim=1)), dim=1)  # 粗糙度
        
        # 综合复杂度评分 [0, 1]
        complexity = torch.clamp(
            0.4 * height_variance + 0.4 * height_gradient + 0.2 * height_roughness,
            0.0, 1.0
        )
        return complexity
    
    
    def _generate_adaptive_speed(self, env_ids):
        """基于地形复杂度生成自适应速度
        
        参数:
            env_ids: 环境ID列表
            
        返回:
            adaptive_speeds: 自适应速度张量
        """
        complexity = self._analyze_terrain_complexity()[env_ids]
        
        # 获取配置参数，如果没有设置则使用默认值
        max_speed = getattr(self.cfg, 'max_speed', 1.0)  # 默认最大速度1.5m/s
        min_speed = getattr(self.cfg, 'min_speed', 0.2)  # 默认最小速度0.2m/s
        
        # 计算速度范围和基础速度
        speed_range_ratio = getattr(self.cfg, 'speed_range_ratio', 0.3)  # 速度范围比例
        complexity_sensitivity = getattr(self.cfg, 'complexity_sensitivity', 1.0)  # 复杂度敏感度
        
        # 基础速度从max_speed到min_speed线性下降
        base_speed = max_speed - complexity * complexity_sensitivity * (max_speed - min_speed)
        
        # 速度范围：简单地形变化大，困难地形变化小
        speed_range = speed_range_ratio * (1 - complexity) * (max_speed - min_speed)
        
        # 在基础速度± 范围内随机采样
        min_speed_val = torch.clamp(base_speed - speed_range, min_speed, max_speed - 0.1)
        max_speed_val = torch.clamp(base_speed + speed_range, min_speed + 0.1, max_speed)
        
        # 生成随机速度
        adaptive_speeds = torch.empty((len(env_ids), 1), device=self.device).uniform_(0, 1)
        adaptive_speeds = min_speed_val.unsqueeze(1) + adaptive_speeds * (max_speed_val.unsqueeze(1) - min_speed_val.unsqueeze(1))
        adaptive_speeds = adaptive_speeds.squeeze(1)
        
        return adaptive_speeds
    
    def _resample_commands(self, env_ids):
        if self.cfg.commands.height_adaptive_speed:
            adaptive_speeds = self._generate_adaptive_speed(env_ids)
            self.commands[env_ids, 0] = adaptive_speeds
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(env_ids), 1), device=self.device
            ).squeeze(1)
            
        if not self.cfg.commands.heading_command:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device
            ).squeeze(1)

            small_command_mask = torch.abs(self.commands[env_ids, 2]) <= self.cfg.commands.ang_vel_clip
            self.commands[env_ids, 2] = torch.where(small_command_mask, 
                                                    torch.zeros_like(self.commands[env_ids, 2]), 
                                                    self.commands[env_ids, 2])

        small_lin_vel_x_mask = torch.abs(self.commands[env_ids, 0]) <= self.cfg.commands.lin_vel_clip
        small_lin_vel_y_mask = torch.abs(self.commands[env_ids, 1]) <= self.cfg.commands.lin_vel_clip
        self.commands[env_ids, 0] = torch.where(small_lin_vel_x_mask, 
                                               torch.zeros_like(self.commands[env_ids, 0]), 
                                               self.commands[env_ids, 0])
        self.commands[env_ids, 1] = torch.where(small_lin_vel_y_mask, 
                                               torch.zeros_like(self.commands[env_ids, 1]), 
                                               self.commands[env_ids, 1])

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.9, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        
        # 初始化环境级别的连续成功/失败计数器
        if not hasattr(self, 'env_consecutive_success'):
            self.env_consecutive_success = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
            self.env_consecutive_failure = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        
        # 获取课程学习配置
        curriculum_cfg = self.cfg.curriculum_config
        
        # 检查每个需要重置的环境
        for env_id in env_ids:
            env_id = env_id.item()
            
            # 根据配置判断成功模式
            if curriculum_cfg.success_mode == 'goal_reached':
                # 目标到达模式：判断是否到达所有目标点
                is_success = self.cur_goal_idx[env_id] >= self.cfg.terrain.num_goals
                success_threshold = curriculum_cfg.success_threshold
                failure_threshold = curriculum_cfg.failure_threshold
                
            elif curriculum_cfg.success_mode == 'survival_time':
                # 存活时间模式：判断是否存活超过指定时间
                episode_time = self.episode_length_buf[env_id] * self.dt
                # print(f"episode_time: {episode_time}")
                is_success = episode_time >= curriculum_cfg.survival_time_threshold
                success_threshold = curriculum_cfg.survival_success_threshold
                failure_threshold = curriculum_cfg.survival_failure_threshold
            
            elif curriculum_cfg.success_mode == 'vel_tracking':
                # 速度模式：判断是否达到指定速度
                is_success = (self.episode_sums["tracking_x_vel"][env_id] / self.max_episode_length) > (0.6 * self.reward_scales["tracking_x_vel"])
                success_threshold = curriculum_cfg.velocity_success_threshold
                failure_threshold = curriculum_cfg.velocity_failure_threshold
            
            if is_success:
                # 成功：增加连续成功计数，重置连续失败计数
                self.env_consecutive_success[env_id] += 1
                self.env_consecutive_failure[env_id] = 0
                
                # 检查是否达到升级条件
                if self.env_consecutive_success[env_id] >= success_threshold:
                    self.terrain_levels[env_id] += 1
                    self.env_consecutive_success[env_id] = 0  # 重置计数器
                    # print(f"环境 {env_id} 连续成功{success_threshold}次，升级到等级 {self.terrain_levels[env_id]}")
            else:
                # 失败：增加连续失败计数，重置连续成功计数
                self.env_consecutive_failure[env_id] += 1
                self.env_consecutive_success[env_id] = 0
                
                # 检查是否达到降级条件
                if self.env_consecutive_failure[env_id] >= failure_threshold:
                    self.terrain_levels[env_id] -= 1
                    self.env_consecutive_failure[env_id] = 0  # 重置计数器
                    # print(f"环境 {env_id} 连续失败{failure_threshold}次，降级到等级 {self.terrain_levels[env_id]}")
        
        # 保持难度在合理范围
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        
        # 更新环境类别和目标
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # 动态计算高度采样点数
        num_height_points = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
        self.measured_heights = torch.zeros((self.num_envs, num_height_points), device=self.device)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.origin_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_foot_action = torch.zeros_like(self.rigid_body_states[:, self.feet_indices, :])
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.last_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]

        if self.cfg.terrain.measure_heights:
            self.height_points, self.height_points_data, self.height_points_data_origin = self._init_height_points()
            # 初始化存储上一次高度的变量（用于地图更新延迟）
            self.last_heights = torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device)
            self.last_heights_data = torch.zeros(self.num_envs, self.num_height_points_data, dtype=torch.float, device=self.device)
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dof, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_all[:] = self.default_dof_pos[0]
        
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        
        self.random_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs ,2), device=self.device)

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        self.height_update_interval = 1
        if hasattr(self.cfg.env, "height_update_dt"):
            self.height_update_interval = int(self.cfg.env.height_update_dt / (self.cfg.sim.dt * self.cfg.control.decimation))

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
        
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = torch_rand_float(self.cfg.terrain.static_friction[0], self.cfg.terrain.static_friction[1], (1, 1), device=self.device).item()
        plane_params.dynamic_friction = torch_rand_float(self.cfg.terrain.dynamic_friction[0], self.cfg.terrain.dynamic_friction[1], (1, 1), device=self.device).item()
        plane_params.restitution = torch_rand_float(self.cfg.terrain.restitution[0], self.cfg.terrain.restitution[1], (1, 1), device=self.device).item()
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = torch_rand_float(self.cfg.terrain.static_friction[0], self.cfg.terrain.static_friction[1], (1, 1), device=self.device).item()
        tm_params.dynamic_friction = torch_rand_float(self.cfg.terrain.dynamic_friction[0], self.cfg.terrain.dynamic_friction[1], (1, 1), device=self.device).item()
        tm_params.restitution = torch_rand_float(self.cfg.terrain.restitution[0], self.cfg.terrain.restitution[1], (1, 1), device=self.device).item()
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            # print("rigid_body_names=",self.gym.get_actor_rigid_body_names(env_handle, actor_handle))

            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        # print("rigid_body_names=",self.gym.get_actor_rigid_body_names(env_handle, actor_handle))

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)


        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in self.body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in self.body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_payload = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs, 2), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
        
        self.torso_body_index = self.body_names.index("torso_link")
        self.left_hand_index = self.body_names.index("left_hand_palm_link")
        self.right_hand_index = self.body_names.index("right_hand_palm_link")   
        
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
 
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.domain_rand.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            if self.cfg.domain_rand.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(0., 0., self.cfg.domain_rand.rand_yaw_range*np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            
            dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][12:] = [300., 200., 200., 200., 100.,  20.,  20.,  20., 200., 200., 200., 100.,  20.,  20.,  20.]
            dof_props["damping"][12:] = [5.0000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000,
                                            0.5000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000]
        
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            self.attach_camera(i, env_handle, actor_handle)
            
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
            
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
            
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])
        
        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
      
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
        
        self.left_hip_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
        
        self.hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.hip_pitch_joints)):
            self.hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_pitch_joints[i])
    
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])
            
        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])
            
        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.upper_body_link)
        self.imu_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if terrain_config.mesh_type == "None":
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
        else:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level # 2
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)
            
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
 
    def _draw_height_samples(self):
        """ 
        可视化高度采样点（用于调试，会降低仿真速度）
        显示策略网络实际看到的数据（包括所有域随机化效果）
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(255, 0, 0))
        i = self.lookat_id
        base_pos = (self.root_states[i, :3]).cpu().numpy()
        heights = self.measured_heights[i].cpu().numpy()
        
        # 直接使用已经计算好的世界坐标系采样点（避免重复计算）
        if hasattr(self, 'height_points_world'):
            height_points = self.height_points_world[i].cpu().numpy()
        else:
            # 备用方案：如果世界坐标点不存在，使用原始方法
            height_points_offset = self.height_points[i] + base_pos
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), 
                                          height_points_offset).cpu().numpy()
        
        if self.save:
            # 数据记录：使用已经计算好的世界坐标系数据点
            heights = self.measured_heights_data[i].cpu().numpy()
            if hasattr(self, 'height_points_data_world'):
                height_points = self.height_points_data_world[i].cpu().numpy()
            else:
                # 备用方案
                height_points_offset = self.height_points_data[i] + base_pos
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), 
                                              height_points_offset).cpu().numpy()
        
        for j in range(heights.shape[0]):
            x = height_points[j, 0]
            y = height_points[j, 1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    def _draw_feet(self):
        if not hasattr(self, '_foothold_offsets'):
            foot_length = getattr(self.cfg.rewards, 'foothold_foot_length', 0.12) # x方向采样点数
            foot_width = getattr(self.cfg.rewards, 'foothold_foot_width', 0.06) # y方向采样点数
            
            num_x = int(foot_length * 100)  # 12个点，转换为int
            num_y = int(foot_width * 100)    # 8个点，转换为int
            
            spacing = 0.01  # 采样间距 0.01m
            # 计算采样范围（确保中心对称）
            x_start = - num_x / 2 * spacing + 0.01
            y_start = - num_y / 2 * spacing
            x_end = -x_start + 0.08
            y_end = -y_start 

            x_samples = torch.linspace(x_start, x_end, num_x, device=self.device)
            y_samples = torch.linspace(y_start, y_end, num_y, device=self.device)
             
            offsets_list = []
            for x in x_samples:
                for y in y_samples:
                    offsets_list.append([x.item(), y.item(), 0.0])
            
            self._foothold_offsets = torch.tensor(offsets_list, device=self.device, dtype=torch.float)
        
        # 获取脚部位置和朝向
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]  # [E, n_feet, 3]
        foot_quats = self.rigid_body_states[:, self.feet_indices, 3:7]     # [E, n_feet, 4]
        
        # 扩展采样点到所有脚: [E, n_feet, n_samples, 3]
        n_samples = self._foothold_offsets.shape[0]
        offsets_expanded = self._foothold_offsets.unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.feet_indices), -1, -1
        )
        
        # 1. 先应用偏移到脚部位置
        sample_points_offset = foot_positions.unsqueeze(2) + offsets_expanded  # [E, n_feet, n_samples, 3]
        # 2. 计算每个脚的采样点中心
        sample_center = torch.mean(sample_points_offset, dim=2, keepdim=True)  # [E, n_feet, 1, 3]
        # 3. 围绕采样点中心旋转：平移到原点，旋转，再平移回去
        points_centered = sample_points_offset - sample_center  # 平移到原点
        points_rotated = quat_apply(
            foot_quats.unsqueeze(2).expand(-1, -1, n_samples, -1),
            points_centered
        )  # 围绕原点旋转
        sample_points_world = points_rotated + sample_center  # 平移回去，得到最终世界坐标
        
        # 可视化采样点：第一个环境，两只脚，所有采样点
        # 绘制采样点（第一只脚用黄色，第二只脚用红色）
        left_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, None, color=(1, 1, 0))   # 黄色
        right_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, None, color=(1, 0, 0))    # 红色
        
        for foot_id in range(min(len(self.feet_indices), 2)):  # 两只脚
            sample_points_env_foot = sample_points_world[0, foot_id, :, :]  # 第一个环境
            
            for sample_idx in range(n_samples):  # 所有采样点
                sample_pos = sample_points_env_foot[sample_idx].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(sample_pos[0], sample_pos[1], sample_pos[2]), r=None)
                
                # 根据脚ID选择颜色
                if foot_id == 0:
                    gymutil.draw_lines(left_geom, self.gym, self.viewer, self.envs[0], pose)
                else:
                    gymutil.draw_lines(right_geom, self.gym, self.viewer, self.envs[0], pose)
  
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)

        # only for recording dataset, not for policy
        y_data = torch.tensor(self.cfg.terrain.dataset_points_y, device=self.device, requires_grad=False)
        x_data = torch.tensor(self.cfg.terrain.dataset_points_x, device=self.device, requires_grad=False)
        grid_x_data, grid_y_data = torch.meshgrid(x_data, y_data)
        self.num_height_points_data = grid_x_data.numel()
        points_data = torch.zeros(self.num_envs, self.num_height_points_data, 3, device=self.device, requires_grad=False)

        # 用于roll pitch噪声计算
        y_data_origin = torch.tensor(self.cfg.terrain.measured_points_y_origin, device=self.device, requires_grad=False)
        x_data_origin = torch.tensor(self.cfg.terrain.measured_points_x_origin, device=self.device, requires_grad=False)
        grid_x_data_origin, grid_y_data_origin = torch.meshgrid(x_data_origin, y_data_origin)
        self.num_height_points_data_origin = grid_x_data_origin.numel()
        points_data_origin = torch.zeros(self.num_envs, self.num_height_points_data_origin, 3, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_offset, self.cfg.terrain.measure_horizontal_offset, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]

            # visualize saved height point
            points_data[i, :, 0] = grid_x_data.flatten()
            points_data[i, :, 1] = grid_y_data.flatten()
            
            points_data_origin[i, :, 0] = grid_x_data_origin.flatten() 
            points_data_origin[i, :, 1] = grid_y_data_origin.flatten() 
            
        return points, points_data, points_data_origin

    def get_foot_contacts(self):
        foot_contacts_bool = self.contact_forces[:, self.feet_indices, 2] > 10
        if self.cfg.env.include_foot_contacts:
            return foot_contacts_bool
        else:
            return torch.zeros_like(foot_contacts_bool).to(self.device)

    def _get_heights(self, env_ids=None):
        """
        采样机器人周围的地形高度
        功能：在机器人周围的指定点采样地形高度，用于策略观测和数据记录
        返回：
            heights: 策略观测用的高度数据（含域随机化噪声）
            heights_data: 数据集记录用的高度数据（不含噪声）
            height_points_data_origin: 用于roll ptich 的噪声计算的原始采样点
        """
        # ========================================
        # 第一步：准备域随机化参数
        # ========================================
        
        # 1.1 Yaw旋转噪声：模拟IMU漂移（rad）
        yaw_noise = torch_rand_float(-self.terrain.cfg.measure_map_yaw_noise,
                                     self.terrain.cfg.measure_map_yaw_noise,
                                     (self.num_envs, 1), device=self.device)
        cos_yaw = torch.cos(yaw_noise)
        sin_yaw = torch.sin(yaw_noise)
        
        # 1.2 Roll/Pitch倾斜噪声：模拟姿态估计误差（rad）
        roll_pitch_noise = torch_rand_float(-self.terrain.cfg.measure_map_roll_pitch_noise,
                                           self.terrain.cfg.measure_map_roll_pitch_noise,
                                           (self.num_envs, 2), device=self.device)
        pitch_noise = roll_pitch_noise[:, 0].unsqueeze(1)  # 绕Y轴旋转
        roll_noise = roll_pitch_noise[:, 1].unsqueeze(1)   # 绕X轴旋转
        
        # 1.3 垂直偏移和噪声：模拟高度测量误差（m）
        vertical_offset = torch_rand_float(-self.terrain.cfg.measure_vertical_offset,
                                          self.terrain.cfg.measure_vertical_offset,
                                          (self.num_envs, 1), device=self.device)
        vertical_noise = torch_rand_float(-self.terrain.cfg.measure_vertical_noise,
                                         self.terrain.cfg.measure_vertical_noise,
                                         (self.num_envs, self.num_height_points), device=self.device)
        
        # ========================================
        # 第二步：在机体坐标系应用Yaw旋转噪声
        # ========================================
        
        # 获取需要处理的环境索引
        num_envs_process = len(env_ids) if env_ids else self.num_envs
        env_slice = env_ids if env_ids else slice(None)
        
        # 2.1 对策略观测点应用Yaw旋转（机体坐标系 -> 旋转后的机体坐标系）
        height_points_body = self.height_points[env_slice]  # 原始机体坐标系采样点
        x_body = height_points_body[:, :, 0]
        y_body = height_points_body[:, :, 1]
        
        # 应用2D旋转矩阵：[x', y'] = [cos(θ) -sin(θ); sin(θ) cos(θ)] * [x, y]
        height_points_rotated = height_points_body.clone()
        yaw_slice = env_ids if env_ids else slice(None)
        height_points_rotated[:, :, 0] = cos_yaw[yaw_slice] * x_body - sin_yaw[yaw_slice] * y_body
        height_points_rotated[:, :, 1] = sin_yaw[yaw_slice] * x_body + cos_yaw[yaw_slice] * y_body
        
        # 2.2 对数据记录点不应用噪声（保持原始）
        height_points_data = self.height_points_data[env_slice]
        height_points_data_origin = self.height_points_data_origin[env_slice]
        x_body_origin = height_points_data_origin[:, :, 0]
        y_body_origin = height_points_data_origin[:, :, 1]
        
        # ========================================
        # 第三步：转换到世界坐标系
        # ========================================
        
        # 3.1 策略观测点：先应用偏移，再围绕测量点中心旋转
        base_quat_repeated = self.base_quat[env_slice].repeat(1, self.num_height_points)
        height_points_offset = height_points_rotated + self.root_states[env_slice, :3].unsqueeze(1)
        
        # 围绕测量点中心旋转：计算测量点中心，平移到原点，旋转，再平移回去
        measurement_center = torch.mean(height_points_offset, dim=1, keepdim=True)  # [envs, 1, 3]
        points_centered = height_points_offset - measurement_center  # 平移到原点
        points_rotated = quat_apply_yaw(base_quat_repeated, points_centered)  # 围绕原点旋转
        points_world = points_rotated + measurement_center  # 平移回去
        
        # 3.2 数据记录点：先应用偏移，再围绕测量点中心旋转
        base_quat_data_repeated = self.base_quat[env_slice].repeat(1, self.num_height_points_data)
        height_points_data_offset = height_points_data + self.root_states[env_slice, :3].unsqueeze(1)
        
        # 围绕测量点中心旋转
        measurement_center_data = torch.mean(height_points_data_offset, dim=1, keepdim=True)
        points_data_centered = height_points_data_offset - measurement_center_data
        points_data_rotated = quat_apply_yaw(base_quat_data_repeated, points_data_centered)
        points_data_world = points_data_rotated + measurement_center_data
        
        # 保存世界坐标系的采样点（用于可视化，避免重复计算）
        if env_ids is None:
            self.height_points_world = points_world
            self.height_points_data_world = points_data_world

        # ========================================
        # 第四步：从世界坐标系转换到网格索引，并采样高度
        # ========================================
        
        # 4.1 策略观测点：世界坐标系 -> 网格索引
        points_grid = points_world + self.terrain.cfg.border_size  # 加上边界偏移
        points_grid = (points_grid / self.terrain.cfg.horizontal_scale).long()  # 转换为网格索引
        px = torch.clip(points_grid[:, :, 0].view(-1), 0, self.height_samples.shape[0]-2)
        py = torch.clip(points_grid[:, :, 1].view(-1), 0, self.height_samples.shape[1]-2)
        
        # 4.2 采样高度（三角形插值取最小值，保守估计）
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(torch.min(heights1, heights2), heights3)
        heights = heights.view(num_envs_process, -1) * self.terrain.cfg.vertical_scale  # 转换为米
        
        # ========================================
        # 第五步：应用高度域随机化（在高度值上）
        # ========================================
        
        # 5.1 垂直偏移：系统性高度误差（所有点统一偏移）
        heights += vertical_offset[env_slice]
        
        # 5.2 垂直噪声：每个采样点独立的测量抖动
        heights += vertical_noise[env_slice]
        
        # 5.3 Roll/Pitch倾斜噪声：模拟地图在俯仰、滚转方向的旋转误差
        # 效果：高度随着X/Y位置产生线性偏移
        # 公式：height_offset = pitch × x_position + roll × y_position
        tilt_offset = (pitch_noise[env_slice] * x_body_origin + 
                      roll_noise[env_slice] * y_body_origin)
        heights += tilt_offset
        
        # 5.4 支撑面扩展：LiDAR平滑化（以一定概率触发）
        # TODO: 当前实现理解有误，需要重新实现
        # 正确含义：将邻近有效落脚点随机扩展为有效点，模拟LiDAR数据后处理的平滑效应
        # if torch.rand(1).item() < self.terrain.cfg.foothold_extension_prob:
        #     kernel_size = 3
        #     heights = torch.nn.functional.max_pool1d(
        #         heights.unsqueeze(1), kernel_size=kernel_size, stride=1, 
        #         padding=kernel_size//2).squeeze(1)
        
        # ========================================
        # 第六步：数据集记录用的高度采样（不添加域随机化噪声）
        # ========================================
        
        # 6.1 世界坐标系 -> 网格索引
        points_data_grid = points_data_world + self.terrain.cfg.border_size
        points_data_grid = (points_data_grid / self.terrain.cfg.horizontal_scale).long()
        px_data = torch.clip(points_data_grid[:, :, 0].view(-1), 0, self.height_samples.shape[0]-2)
        py_data = torch.clip(points_data_grid[:, :, 1].view(-1), 0, self.height_samples.shape[1]-2)
        
        # 6.2 采样高度（无噪声）
        heights1_data = self.height_samples[px_data, py_data]
        heights2_data = self.height_samples[px_data+1, py_data]
        heights3_data = self.height_samples[px_data, py_data+1]
        heights_data = torch.min(torch.min(heights1_data, heights2_data), heights3_data)
        heights_data = heights_data.view(num_envs_process, -1) * self.terrain.cfg.vertical_scale

        # ========================================
        # 第七步：地图更新延迟（Map Repeat）
        # ========================================
        # 模拟真实传感器地图刷新滞后，以一定概率返回上一次的高度数据
        if torch.rand(1).item() < self.terrain.cfg.map_repeat_prob:
            # 使用上一次缓存的高度数据（地图未刷新）
            heights_return = self.last_heights
            heights_data_return = self.last_heights_data
        else:
            # 使用当前采样的新数据，并更新缓存
            self.last_heights = heights
            self.last_heights_data = heights_data
            heights_return = heights
            heights_data_return = heights_data
        # print('heights_return', heights_return)  # 调试用
        # print('heights_data_return', heights_data_return)  # 调试用
        return heights_return, heights_data_return

    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        if self.cfg.terrain.mesh_type == 'plane':
            left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            return torch.cat((left_foot_height, right_foot_height), dim=-1), torch.cat((left_foot_height_var, right_foot_height_var), dim=-1)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = left_foot_pos[env_ids].clone()
            right_points = right_foot_pos[env_ids].clone()
        else:
            left_points = left_foot_pos.clone()
            right_points = right_foot_pos.clone()

        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points/self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points/self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0]-2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0]-2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1]-2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1]-2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px+1, left_py]
        left_heights3 = self.height_samples[left_px, left_py+1]
        left_heights = torch.min(left_heights1, left_heights2)
        left_heights = torch.min(left_heights, left_heights3)
        left_heights = left_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        left_foot_heights =  left_foot_pos[:, :, 2] - left_heights

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px+1, right_py]
        right_heights3 = self.height_samples[right_px, right_py+1]
        right_heights = torch.min(right_heights1, right_heights2)
        right_heights = torch.min(right_heights, right_heights3)
        right_heights = right_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        right_foot_heights =  right_foot_pos[:, :, 2] - right_heights

        feet_heights = torch.cat((torch.mean(left_foot_heights, dim=-1, keepdim=True), torch.mean(right_foot_heights, dim=-1, keepdim=True)), dim=-1)
        feet_heights_var = torch.cat((torch.var(left_foot_heights, dim=-1, keepdim=True), torch.var(right_foot_heights, dim=-1, keepdim=True)), dim=-1)

        return torch.clip(feet_heights, min=0.), feet_heights_var


    #------------ reward functions----------------
    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_heading_tracking(self):
        heading_error = wrap_to_pi(self.commands[:, 3] - self.yaw)
        return torch.exp(-torch.abs(heading_error) / self.cfg.rewards.tracking_sigma)  # 朝向越准确奖励越高
    
    def _reward_next_heading_tracking(self):
        next_heading_error = wrap_to_pi(self.next_target_yaw - self.yaw)
        return torch.exp(-torch.abs(next_heading_error) / self.cfg.rewards.tracking_sigma)  # 朝向越准确奖励越高

    def _reward_reach_goal(self):
        """靠近目标奖励,远离目标惩罚"""
        distance_to_goal = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1)
        # 计算距离变化: 负值=靠近(给奖励), 正值=远离(给惩罚)
        distance_change = distance_to_goal - self.last_distance_to_goal
        self.last_distance_to_goal = distance_to_goal
        # 返回负的距离变化: 靠近->正奖励, 远离->负惩罚
        return -distance_change
    
    def _reward_center(self):
        y_offset = torch.square(self.root_states[:, 1] - self.cur_goals[:, 1])
        return y_offset
    
    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        height_error = torch.abs(base_height - self.cfg.rewards.base_height_target + self.cfg.asset.ankle_sole_distance)
        return torch.exp(-height_error / self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
        
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_deviation_hip_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1)
    
    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1)
    
    def _reward_deviation_knee_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.knee_joint_indices], dim=-1)
    
    # def _reward_deviation_knee_joint(self):
    #     height_error = (self.root_states[:, 2] - self.commands[:, 4])
    #     knee_action_min = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
    #     knee_action_max = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
    #     joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (knee_action_max - knee_action_min) # always positive
    #     return torch.sum(torch.abs((joint_deviation-0.5) * height_error.unsqueeze(-1)), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0])[:, :self.num_actions].clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1])[:, :self.num_actions].clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contacts, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 # no reward for zero command
        return rew_airTime
    
    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var = self._get_feet_heights()
        height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * feet_leteral_vel, dim=1)
    
    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(-foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0)
    
    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1])
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(-knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0)
    
    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)
    
    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1, unbiased=False)
        feet_distances_var = torch.nan_to_num(feet_distances_var, nan=0.0, posinf=0.0, neginf=0.0)
        return feet_distances_var
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + 0.2 * torch.square(self.commands[:, 2]), min=0.1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_torques(self):
        # Penalize torques
        # 修复：p_gains已经是[num_envs, num_dof]，不需要unsqueeze
        torques_normalized = (self.torques / self.p_gains)[:, :self.num_lower_dof]
        return torch.sum(torch.square(torques_normalized), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :self.num_lower_dof]), dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit)[:, :self.num_lower_dof].clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit)[:, :self.num_lower_dof].clip(min=0.), dim=1)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, :3], dim=1) < 0.1)) # full reward for zero command
        return rew_no_fly
    
    # def _reward_joint_tracking_error(self):
    #     return torch.sum(torch.square(self.joint_pos_target[:, :self.num_lower_dof] - self.dof_pos[:, :self.num_lower_dof]), dim=-1)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(self.contact_forces[:, self.feet_indices, 2] - 50, min=0)
        return torch.sum(feet_contact_momentum_z, dim=1)
    
    def _reward_action_vanish(self):
        upper_error = torch.clip(self.origin_actions[:, :self.num_lower_dof] - self.action_max[:, :self.num_lower_dof], min=0)
        lower_error = torch.clip(self.action_min[:, :self.num_lower_dof] - self.origin_actions[:, :self.num_lower_dof], min=0)
        return torch.sum(upper_error + lower_error, dim=-1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)
        error_sim = contacts
        return error_sim * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_foothold(self):
        """
        按照BEAMDOJO论文公式实现
        公式: -∑_{i=1}^{2} C_i · ∑_{j=1}^{n} 1{dij < ε}
        
        说明：
        - C_i: 第i只脚的接触状态
        - dij: 第i只脚第j个采样点处的真实地形高度
        - ε: 高度容忍度（threshold）
        - 1{dij < ε}: 指示函数（如果地形高度低于阈值则为1，表示踩空）
        
        惩罚踩空的情况
        """
        # 获取脚的接触状态 C_i
        contact_forces = self.contact_forces[:, self.feet_indices, 2]  # [num_envs, n_feet]
        contact_threshold = getattr(self.cfg.rewards, 'contact_force_threshold', 1.0)
        contact = contact_forces > contact_threshold  # C_i
        contact_float = contact.float()  # [num_envs, n_feet]
        
        # 初始化脚底采样点（首次调用时）
        if not hasattr(self, '_foothold_offsets'):
            foot_length = getattr(self.cfg.rewards, 'foothold_foot_length', 0.12) # x方向采样点数
            foot_width = getattr(self.cfg.rewards, 'foothold_foot_width', 0.06) # y方向采样点数
            
            num_x = int(foot_length * 100)  # 12个点，转换为int
            num_y = int(foot_width * 100)    # 8个点，转换为int
            
            spacing = 0.01  # 采样间距 0.01m
            # 计算采样范围（确保中心对称）
            x_start = - num_x / 2 * spacing + 0.01
            y_start = - num_y / 2 * spacing 
            x_end = -x_start + 0.08
            y_end = -y_start 
            

            x_samples = torch.linspace(x_start, x_end, num_x, device=self.device)
            y_samples = torch.linspace(y_start, y_end, num_y, device=self.device)
             
            offsets_list = []
            for x in x_samples:
                for y in y_samples:
                    offsets_list.append([x.item(), y.item(), 0.0])
            
            self._foothold_offsets = torch.tensor(offsets_list, device=self.device, dtype=torch.float)
    
        # 获取脚部位置和朝向
        foot_positions = self.rigid_body_states[:, self.feet_indices, :3]  # [E, n_feet, 3]
        foot_quats = self.rigid_body_states[:, self.feet_indices, 3:7]     # [E, n_feet, 4]
        
        # 扩展采样点到所有脚: [E, n_feet, n_samples, 3]
        n_samples = self._foothold_offsets.shape[0]
        offsets_expanded = self._foothold_offsets.unsqueeze(0).unsqueeze(0).expand(
            self.num_envs, len(self.feet_indices), -1, -1
        )
        
        # 1. 先应用偏移到脚部位置
        sample_points_offset = foot_positions.unsqueeze(2) + offsets_expanded  # [E, n_feet, n_samples, 3]
        # 2. 计算每个脚的采样点中心
        sample_center = torch.mean(sample_points_offset, dim=2, keepdim=True)  # [E, n_feet, 1, 3]
        # 3. 围绕采样点中心旋转：平移到原点，旋转，再平移回去
        points_centered = sample_points_offset - sample_center  # 平移到原点
        points_rotated = quat_apply(
            foot_quats.unsqueeze(2).expand(-1, -1, n_samples, -1),
            points_centered
        )  # 围绕原点旋转
        sample_points_world = points_rotated + sample_center  # 平移回去，得到最终世界坐标
        
        # === 关键：查询真实地形高度 ===
        # 将世界坐标转换为地形网格索引
        points_grid = sample_points_world + self.terrain.cfg.border_size
        points_grid = (points_grid / self.terrain.cfg.horizontal_scale).long()
        
        # 展平以便批量采样
        E, F, S = self.num_envs, len(self.feet_indices), n_samples
        px = torch.clip(points_grid[:, :, :, 0].reshape(-1), 0, self.height_samples.shape[0]-2)
        py = torch.clip(points_grid[:, :, :, 1].reshape(-1), 0, self.height_samples.shape[1]-2)
        
        # 三角形插值取最小值（保守估计）
        h1 = self.height_samples[px, py]
        h2 = self.height_samples[px+1, py]
        h3 = self.height_samples[px, py+1]
        terrain_heights = torch.min(torch.min(h1, h2), h3)
        terrain_heights = terrain_heights.view(E, F, S) * self.terrain.cfg.vertical_scale  # dij
        
        # 获取高度容忍度 ε
        epsilon = getattr(self.cfg.rewards, 'foothold_height_tolerance', -0.1)
        
        # 指示函数：1{dij < ε} 
        # 含义：如果采样点低于地形高度容忍度，说明踩空了（地形高度不足）
        down = (terrain_heights < epsilon).float()  # [E, F, S]
        # print('terrain_heights:',terrain_heights)
        # print('terrain_heights.shape:',terrain_heights.shape)
        
        # 对采样点求和：∑_{j=1}^{n} 1{dij < ε}
        sum_samples = torch.sum(down, dim=2)  # [E, F]
        # print('sum_samples:', sum_samples)
        # print('contact_float * sum_samples:', contact_float * sum_samples)
        # 对所有脚求和：-∑_{i=1}^{2} C_i · ∑
        return -torch.sum(contact_float * sum_samples, dim=1)  # [E]
    