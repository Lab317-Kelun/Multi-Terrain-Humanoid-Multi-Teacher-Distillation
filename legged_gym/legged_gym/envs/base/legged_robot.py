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

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

from terrain_base.terrain import Terrain
from terrain_base.config import terrain_config

import threading


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x.unsqueeze(1), pitch_y.unsqueeze(1), yaw_z.unsqueeze(1)

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
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
        self.debug_viz = False
        self.init_done = False
        
        self.total_times = 0
        self.last_times = -1
        self.success_times = 0
        self.complete_times = 0.
        
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_actions
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        if (self.common_step_counter % self.cfg.domain_rand.upper_interval == 0):
            # (NOTE) implementation of upper-body curriculum
            self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
            uu = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device)
            self.random_upper_ratio = -1.0 / (20 * (1-self.random_upper_ratio*0.99))*torch.log(1 - uu + uu * np.exp(-20 * (1-self.random_upper_ratio*0.99)))
            self.random_joint_ratio = self.random_upper_ratio * torch.rand(self.num_envs, self.num_actions - self.num_lower_dof).to(self.device)
            rand_pos = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device) - 0.5
            self.random_upper_actions = ((self.action_min[:, self.num_lower_dof:] * (rand_pos >= 0)) + (self.action_max[:, self.num_lower_dof:] * (rand_pos < 0) ))* self.random_joint_ratio
            self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (self.cfg.domain_rand.upper_interval)
        self.current_upper_actions += self.delta_upper_actions
        actions = torch.cat((actions, self.current_upper_actions), dim=-1)
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.origin_actions[:] = self.actions[:]
        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # upper-body with position control; lower-body with force control;
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        # 检查是否使用双critic，如果是则返回奖励字典
        use_double_critic = hasattr(self.cfg, 'algorithm') and hasattr(self.cfg.algorithm, 'use_double_critic') and self.cfg.algorithm.use_double_critic
        if use_double_critic and hasattr(self, 'dense_rew_buf') and hasattr(self, 'sparse_rew_buf'):
            rewards = {
                'dense': self.dense_rew_buf,
                'sparse': self.sparse_rew_buf
            }
        else:
            rewards = self.rew_buf
            
        return self.obs_buf, self.privileged_obs_buf, rewards, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var = self._get_feet_heights()
        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights, self.measured_heights_data  = self._get_heights()

        
        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)

        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt
        
        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)
        #     self._draw_height_samples()
        #     self._draw_goals()
        #     self._draw_feet()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.5, dim=1)
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < 0.0

        # 检查机器人是否超出地形边界
        length = (self.cfg.terrain.terrain_length / 2) - 0.1
        width = (self.cfg.terrain.terrain_width - 1) / 2 - 0.1
        relative_pos = self.root_states[:, :2] - self.env_origins[:, :2]
        x_out_of_bounds = (relative_pos[:, 0] < -length) | (relative_pos[:, 0] > length) 
        y_out_of_bounds = (relative_pos[:, 1] < -width) | (relative_pos[:, 1] > width)
        
        boundary_cutoff = x_out_of_bounds | y_out_of_bounds

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= reach_goal_cutoff
        self.reset_buf |= boundary_cutoff
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf
        self.reset_buf |= height_cutoff

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
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.random_upper_actions[env_ids] = 0. 
        self.current_upper_actions[env_ids] = 0.
        self.delta_upper_actions[env_ids] = 0.
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # self.extras["episode"]["height_curriculum_ratio"] = self.height_curriculum_ratio
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        
        # 检查是否使用双critic
        use_double_critic = hasattr(self.cfg, 'algorithm') and hasattr(self.cfg.algorithm, 'use_double_critic') and self.cfg.algorithm.use_double_critic
        
        if use_double_critic:
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
                self.sparse_rew_buf[:] = torch.clip(self.sparse_rew_buf[:], min=0.)
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
        """ Computes observations
        观测组成详细说明：
        1. 命令信息 (4维)：
           - commands[:, :3]: 线速度x、y和角速度yaw命令 (3维)
           - commands[:, 4]: 高度命令 (1维)
        2. IMU信息 (6维)：
           - imu_ang_vel: IMU角速度 (3维)
           - imu_projected_gravity: 投影重力向量 (3维)
        3. 关节信息 (54维)：
           - dof_pos偏差: 所有27个关节的位置偏差 (27维)
           - dof_vel: 所有27个关节的速度 (27维)
        4. 动作信息 (12维)：
           - actions[:, :12]: 上一步的12个动作值 (12维)
        
        总计单步观测维度: 4 + 6 + 54 + 12 = 76维
        """
        # IMU角速度 (3维) - 机体坐标系下的角速度
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        # IMU投影重力 (3维) - 机体坐标系下的重力方向
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        
        current_obs = torch.cat((   
                                    # 1. 命令信息 (4维)
                                    self.commands[:, :3] * self.commands_scale,  # 线速度x,y + 角速度yaw (3维)
                                    self.commands[:, 4].unsqueeze(1),            # 高度命令 (1维)
                                    
                                    # 2. IMU信息 (6维)
                                    imu_ang_vel  * self.obs_scales.ang_vel,     # IMU角速度 (3维)
                                    imu_projected_gravity,                       # 投影重力向量 (3维)
                                    
                                    # 3. 关节信息 (54维)
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 27个关节位置偏差 (27维)
                                    self.dof_vel * self.obs_scales.dof_vel,                           # 27个关节速度 (27维)
                                    
                                    # 4. 动作信息 (12维)
                                    self.actions[:, :12],                        # 上一步的12个动作 (12维)
                                    ),dim=-1)

        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, -1, 1.)
        current_obs = torch.cat([current_obs, heights], dim=-1)

        current_actor_obs = torch.clone(current_obs)
        # if self.add_noise:
        #     current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof)]           
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_actor_obs[:, :self.num_one_step_obs]), dim=-1)
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions[:, :12],
                                    ),dim=-1)
        
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, -1, 1.)
        current_obs = torch.cat([current_obs, heights], dim=-1)

        # add noise if needed
        # if self.add_noise:
        #     # 动态计算噪声向量长度以匹配观测维度
        #     noise_length = min(current_obs.shape[1], len(self.noise_scale_vec))
        #     current_obs[:, :noise_length] += (2 * torch.rand_like(current_obs[:, :noise_length]) - 1) * self.noise_scale_vec[:noise_length]
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)[env_ids]
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        # if self.cfg.depth.use_camera:
        #     self.graphics_device_id = self.sim_device_id  # required in headless mode
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
        
    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, torso_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)
            
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

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
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

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
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]
            props[self.left_hand_index].mass = self.default_rigid_body_mass[self.left_hand_index] + self.hand_payload[env_id, 0]
            props[self.right_hand_index].mass = self.default_rigid_body_mass[self.right_hand_index] + self.hand_payload[env_id, 1]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2])

        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
                
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        is_height = set_x < 1/3
        is_vel = set_x > 1/2
        self.commands[env_ids, 0] = (torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        self.commands[env_ids, 1] = (torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        else:
            self.commands[env_ids, 2] = (torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        
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
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="M":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            
            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            return torch.cat((torques[..., :self.num_lower_dof], self.joint_pos_target[..., self.num_lower_dof:]), dim=-1)
        
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

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
            if self.cfg.domain_rand.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.domain_rand.randomize_start_yaw:
                rand_yaw = self.cfg.domain_rand.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.domain_rand.randomize_start_pitch:
                    rand_pitch = self.cfg.domain_rand.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.domain_rand.randomize_start_y:
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
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]) and (torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_y_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

    def update_action_curriculum(self, env_ids):
        """ Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

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

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # 计算基础观测维度
        base_obs_dim = 10 + 2*self.num_actions + self.num_lower_dof
        # 如果有高度测量，添加高度维度
        height_dim = self.cfg.env.num_height if hasattr(self.cfg.env, 'num_height') else 0
        total_dim = base_obs_dim + height_dim
        
        noise_vec = torch.zeros(total_dim, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[0:4] = 0. # commands
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:(10 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(10 + self.num_actions):(10 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(10 + 2 * self.num_actions):(10 + 2 * self.num_actions + self.num_lower_dof)] = 0. # previous actions
        
        # 为高度测量添加噪声（如果存在）
        if height_dim > 0:
            height_noise_scale = getattr(noise_scales, 'height_measurement', 0.1) * noise_level
            noise_vec[base_obs_dim:base_obs_dim + height_dim] = height_noise_scale
            
        return noise_vec

    #----------------------------------------
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
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
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
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        if self.cfg.terrain.measure_heights:
            self.height_points, self.height_points_data, self.height_points_data_origin = self._init_height_points()
            # 初始化存储上一次高度的变量（用于地图更新延迟）
            self.last_heights = torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device)
            self.last_heights_data = torch.zeros(self.num_envs, self.num_height_points_data, dtype=torch.float, device=self.device)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
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
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        
        self.random_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
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
        
        print('self.body_names',self.body_names)
        print('self.dof_names',self.dof_names)
        self.torso_body_index = self.body_names.index("torso_link")
        self.left_hand_index = self.body_names.index("left_hand_palm_link")
        self.right_hand_index = self.body_names.index("right_hand_palm_link")    
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
            
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            
            self.actor_handles.append(actor_handle)

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
            
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)

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
        # 使用第一个环境进行可视化调试
        i = getattr(self, 'lookat_id', 0)
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
        lookat_id = getattr(self, 'lookat_id', 0)
        goals = self.terrain_goals[self.terrain_levels[lookat_id], self.terrain_types[lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[lookat_id], pose)
                if self.reached_goal_ids[lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[lookat_id], pose)
        
        # if not self.cfg.depth.use_camera:
        if True:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[lookat_id], pose)
        
    def _draw_feet(self):
        if not hasattr(self, '_foothold_offsets'):
            foot_length = getattr(self.cfg.rewards, 'foothold_foot_length', 0.12) # x方向采样点数
            foot_width = getattr(self.cfg.rewards, 'foothold_foot_width', 0.06) # y方向采样点数
            
            num_x = int(foot_length * 100)  # 12个点，转换为int
            num_y = int(foot_width * 100)    # 8个点，转换为int
            
            spacing = 0.01  # 采样间距 0.01m
            # 计算采样范围（确保中心对称）
            x_start = - num_x / 2 * spacing + 0.01
            y_start = - num_y / 2 * spacing + 9
            x_end = -x_start + 0.08
            y_end = -y_start + 18

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
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        height_error = torch.abs(base_height - self.commands[:, 4] + self.cfg.asset.ankle_sole_distance)
        return torch.exp(-height_error * 4)
    
    def _reward_deviation_hip_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_deviation_knee_joint(self):
        height_error = (self.root_states[:, 2] - self.commands[:, 4])
        knee_action_min = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
        knee_action_max = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
        joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (knee_action_max - knee_action_min) # always positive
        return torch.sum(torch.abs((joint_deviation-0.5) * height_error.unsqueeze(-1)), dim=-1)
    
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
        return torch.sum(height_error * feet_leteral_vel, dim=1) * (self.commands[:, 4]>=0.71)
    
    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(-foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0) * (self.commands[:, 4] >= 0.735)
    
    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1])
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(-knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0) * (self.commands[:, 4] >= 0.735)
    
    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)
    
    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1)
        return feet_distances_var * (self.commands[:, 4] >= 0.735)
    
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
        return torch.sum(torch.square((self.torques / self.p_gains.unsqueeze(0))[:, :self.num_lower_dof]), dim=1)

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
    
    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target[:, :self.num_lower_dof] - self.dof_pos[:, :self.num_lower_dof]), dim=-1)
    
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
    
    # def _reward_action_vanish(self):
    #     upper_error = torch.clip(self.origin_actions[:, :self.num_lower_dof] - self.action_max[:, :self.num_lower_dof], min=0)
    #     lower_error = torch.clip(self.action_min[:, :self.num_lower_dof] - self.origin_actions[:, :self.num_lower_dof], min=0)
    #     return torch.sum(upper_error + lower_error, dim=-1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)
        error_sim = (contacts) * (self.commands[:, 4] >= 0.735)
        return error_sim * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
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
            x_start = - num_x / 2 * spacing +0.01
            y_start = - num_y / 2 * spacing + 9
            x_end = -x_start +0.08
            y_end = -y_start + 18
            

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