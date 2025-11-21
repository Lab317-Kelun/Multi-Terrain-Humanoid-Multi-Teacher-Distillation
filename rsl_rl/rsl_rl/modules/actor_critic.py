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

import numpy as np

from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class TerrainOnehotHistoryEncoder(nn.Module):
    """使用CNN从历史信息预测terrain onehot编码（类似StateHistoryEncoder）"""
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(TerrainOnehotHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        # 直接输出编码后的特征，维度为output_size（与terrain_onehot_encoder输出维度一致）
        # 注意：这里不输出softmax，因为要与terrain_onehot_encoder的输出比较
        return self.linear_output(output)


class CNNScanEncoder(nn.Module):
    def __init__(self, channels, kernel_sizes, strides, activation_fn, output_dim):
        super().__init__()
        if channels is None or len(channels) == 0:
            raise ValueError("channels must be a non-empty list for CNNScanEncoder")
        if kernel_sizes is None:
            kernel_sizes = [3] * len(channels)
        if strides is None:
            strides = [1] * len(channels)
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes and strides must have the same length")

        conv_layers = []
        in_channels = 1
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            padding = max((kernel_size - 1) // 2, 0)
            conv_layers.append(nn.Conv1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
            conv_layers.append(deepcopy(activation_fn))
            in_channels = out_channels
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*conv_layers)

        projection_layers = [nn.Flatten(),
                             nn.Linear(in_channels, output_dim),
                             nn.Tanh()]
        self.projection = nn.Sequential(*projection_layers)

    def forward(self, scan):
        x = scan.unsqueeze(1)
        x = self.conv(x)
        x = self.projection(x)
        return x

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 terrain_onehot_encoder_dims,
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 scan_encoder_type='mlp',
                 scan_cnn_channels=None,
                 scan_cnn_kernel_sizes=None,
                 scan_cnn_strides=None,
                 scan_cnn_output_dim=None,
                 scan_encoder_debug=False,
                 tanh_encoder_output=False,
                 n_terrain_onehot=0) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.n_terrain_onehot = n_terrain_onehot
        self.scan_encoder_type = (scan_encoder_type or 'mlp').lower()
        self.scan_encoder_debug = scan_encoder_debug
        self.if_scan_encode = num_scan > 0 and self.scan_encoder_type != 'none'

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent
            
        # Terrain Onehot Encoder (编码真实的terrain onehot)
        if len(terrain_onehot_encoder_dims) > 0 :
            terrain_onehot_encoder_layers = []
            terrain_onehot_encoder_layers.append(nn.Linear(self.n_terrain_onehot, terrain_onehot_encoder_dims[0]))
            terrain_onehot_encoder_layers.append(activation)
            for l in range(len(terrain_onehot_encoder_dims) - 1):
                terrain_onehot_encoder_layers.append(nn.Linear(terrain_onehot_encoder_dims[l], terrain_onehot_encoder_dims[l + 1]))
                terrain_onehot_encoder_layers.append(activation)
            self.terrain_onehot_encoder = nn.Sequential(*terrain_onehot_encoder_layers)
            terrain_onehot_encoder_output_dim = terrain_onehot_encoder_dims[-1]
        else:
            self.terrain_onehot_encoder = nn.Identity()
            terrain_onehot_encoder_output_dim = self.n_terrain_onehot

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        self.terrain_onehot_history_encoder = TerrainOnehotHistoryEncoder(activation, num_prop, num_hist, terrain_onehot_encoder_output_dim)
   

        if self.if_scan_encode:
            if self.scan_encoder_type == 'cnn':
                if scan_cnn_channels is None or len(scan_cnn_channels) == 0:
                    raise ValueError("scan_cnn_channels must be provided when scan_encoder_type is 'cnn'")
                if scan_cnn_output_dim is None:
                    scan_cnn_output_dim = scan_cnn_channels[-1]
                if scan_cnn_kernel_sizes is None:
                    scan_cnn_kernel_sizes = [3] * len(scan_cnn_channels)
                if scan_cnn_strides is None:
                    scan_cnn_strides = [1] * len(scan_cnn_channels)
                self.scan_encoder = CNNScanEncoder(
                    scan_cnn_channels,
                    scan_cnn_kernel_sizes,
                    scan_cnn_strides,
                    activation,
                    scan_cnn_output_dim
                )
                self.scan_encoder_output_dim = scan_cnn_output_dim
            elif self.scan_encoder_type == 'mlp':
                if scan_encoder_dims is None or len(scan_encoder_dims) == 0:
                    raise ValueError("scan_encoder_dims must be provided when scan_encoder_type is 'mlp'")
                scan_encoder = []
                scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
                scan_encoder.append(activation)
                for l in range(len(scan_encoder_dims) - 1):
                    if l == len(scan_encoder_dims) - 2:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                        scan_encoder.append(nn.Tanh())
                    else:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                        scan_encoder.append(activation)
                self.scan_encoder = nn.Sequential(*scan_encoder)
                self.scan_encoder_output_dim = scan_encoder_dims[-1]
            else:
                raise ValueError(f"Unsupported scan_encoder_type: {self.scan_encoder_type}")
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
        
        self.actor_input_dim = num_prop + self.scan_encoder_output_dim + num_priv_explicit + priv_encoder_output_dim + terrain_onehot_encoder_output_dim

        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim+
                                      terrain_onehot_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

        if self.scan_encoder_debug:
            self._log_scan_encoder_info(scan_encoder_dims, scan_cnn_channels, scan_cnn_kernel_sizes, scan_cnn_strides, num_actions)

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        if not eval:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            obs_terrain_onehot = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent:self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.n_terrain_onehot]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            if hist_encoding:
                hist_terrain_onehot_latent = self.infer_hist_terrain_onehot(obs)
            else:
                hist_terrain_onehot_latent = self.infer_terrain_onehot(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, obs_terrain_onehot, latent, hist_terrain_onehot_latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        else:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            obs_terrain_onehot = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent:self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.n_terrain_onehot]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            if hist_encoding:
                hist_terrain_onehot_latent = self.infer_hist_terrain_onehot(obs)
            else:
                hist_terrain_onehot_latent = self.infer_terrain_onehot(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, obs_terrain_onehot, latent, hist_terrain_onehot_latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_terrain_onehot(self, obs):
        """从obs中获取真实的terrain onehot并编码"""
        terrain_onehot = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent + self.n_terrain_onehot]
        return self.terrain_onehot_encoder(terrain_onehot)
    
    def infer_hist_terrain_onehot(self, obs):
        """从历史信息推断terrain onehot编码（使用CNN），然后编码"""
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.terrain_onehot_history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

    def _log_scan_encoder_info(self, scan_encoder_dims, scan_cnn_channels, scan_cnn_kernel_sizes, scan_cnn_strides, num_actions):
        print("[ScanEncoderDebug] ===== Scan Encoder Summary =====")
        print(f"[ScanEncoderDebug] Type: {self.scan_encoder_type.upper()} | Num scan inputs: {self.num_scan}")
        if self.if_scan_encode:
            if self.scan_encoder_type == 'cnn':
                print(f"[ScanEncoderDebug] CNN channels: {scan_cnn_channels}, kernel_sizes: {scan_cnn_kernel_sizes}, strides: {scan_cnn_strides}")
            else:
                print(f"[ScanEncoderDebug] MLP hidden dims: {scan_encoder_dims}")
            print(f"[ScanEncoderDebug] Latent dim: {self.scan_encoder_output_dim}")
            with torch.no_grad():
                dummy = torch.zeros(1, self.num_scan)
                latent = self.scan_encoder(dummy)
                print(f"[ScanEncoderDebug] Dummy forward output shape: {list(latent.shape)}")
        else:
            print("[ScanEncoderDebug] Scan encoder disabled (passing raw scan).")
        print(f"[ScanEncoderDebug] Actor input dim: {self.actor_input_dim}, num_actions: {num_actions}")
        print("[ScanEncoderDebug] ==================================")

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        supported_kwargs = {
            'priv_encoder_dims',
            'tanh_encoder_output',
            'scan_encoder_type',
            'scan_cnn_channels',
            'scan_cnn_kernel_sizes',
            'scan_cnn_strides',
            'scan_cnn_output_dim',
            'scan_encoder_debug',
        }
        unexpected = [key for key in kwargs.keys() if key not in supported_kwargs]
        if unexpected:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(unexpected))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs.get('priv_encoder_dims', [])
        tanh_encoder_output = kwargs.get('tanh_encoder_output', False)
        scan_encoder_type = kwargs.get('scan_encoder_type', 'mlp')
        scan_cnn_channels = kwargs.get('scan_cnn_channels')
        scan_cnn_kernel_sizes = kwargs.get('scan_cnn_kernel_sizes')
        scan_cnn_strides = kwargs.get('scan_cnn_strides')
        scan_cnn_output_dim = kwargs.get('scan_cnn_output_dim')
        scan_encoder_debug = kwargs.get('scan_encoder_debug', False)
        activation = get_activation(activation)
        
        self.actor = Actor(
            num_prop,
            num_scan,
            num_actions,
            scan_encoder_dims,
            actor_hidden_dims,
            priv_encoder_dims,
            num_priv_latent,
            num_priv_explicit,
            num_hist,
            activation,
            scan_encoder_type=scan_encoder_type,
            scan_cnn_channels=scan_cnn_channels,
            scan_cnn_kernel_sizes=scan_cnn_kernel_sizes,
            scan_cnn_strides=scan_cnn_strides,
            scan_cnn_output_dim=scan_cnn_output_dim,
            scan_encoder_debug=scan_encoder_debug,
            tanh_encoder_output=tanh_encoder_output,
        )
        

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

class ActorCriticRMADoubleReward(nn.Module):
    """
    BEAMDOJO双Critic网络架构 - 分离密集和稀疏奖励学习
    """
    is_recurrent = False
    
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        use_double_critic=False,
                        **kwargs):
        supported_kwargs = {
            'priv_encoder_dims',
            'tanh_encoder_output',
            'scan_encoder_type',
            'scan_cnn_channels',
            'scan_cnn_kernel_sizes',
            'scan_cnn_strides',
            'scan_cnn_output_dim',
            'scan_encoder_debug',
        }
        unexpected = [key for key in kwargs.keys() if key not in supported_kwargs]
        if unexpected:
            print("ActorCriticRMADoubleReward.__init__ got unexpected arguments, which will be ignored: " + str(unexpected))
        super(ActorCriticRMADoubleReward, self).__init__()

        self.kwargs = kwargs
        self.use_double_critic = use_double_critic
        priv_encoder_dims= kwargs.get('priv_encoder_dims', [])
        terrain_onehot_encoder_dims = kwargs.get('terrain_onehot_encoder_dims', [])
        tanh_encoder_output = kwargs.get('tanh_encoder_output', False)
        scan_encoder_type = kwargs.get('scan_encoder_type', 'mlp')
        scan_cnn_channels = kwargs.get('scan_cnn_channels')
        scan_cnn_kernel_sizes = kwargs.get('scan_cnn_kernel_sizes')
        scan_cnn_strides = kwargs.get('scan_cnn_strides')
        scan_cnn_output_dim = kwargs.get('scan_cnn_output_dim')
        scan_encoder_debug = kwargs.get('scan_encoder_debug', False)
        n_terrain_onehot = kwargs.get('n_terrain_onehot', 0)
        activation = get_activation(activation)
        
        # Actor网络（与原版保持一致）
        self.actor = Actor(
            num_prop,
            num_scan,
            num_actions,
            scan_encoder_dims,
            actor_hidden_dims,
            priv_encoder_dims,
            terrain_onehot_encoder_dims,
            num_priv_latent,
            num_priv_explicit,
            num_hist,
            activation,
            scan_encoder_type=scan_encoder_type,
            scan_cnn_channels=scan_cnn_channels,
            scan_cnn_kernel_sizes=scan_cnn_kernel_sizes,
            scan_cnn_strides=scan_cnn_strides,
            scan_cnn_output_dim=scan_cnn_output_dim,
            scan_encoder_debug=scan_encoder_debug,
            tanh_encoder_output=tanh_encoder_output,
            n_terrain_onehot=n_terrain_onehot,
        )
        
        # Critic网络
        if use_double_critic:
            # 双Critic - 一个处理密集奖励，一个处理稀疏奖励
            critic1_layers = []  # 密集奖励Critic
            critic1_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic1_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic1_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic1_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic1_layers.append(activation)
            self.critic1 = nn.Sequential(*critic1_layers)
            
            critic2_layers = []  # 稀疏奖励Critic
            critic2_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic2_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic2_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic2_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic2_layers.append(activation)
            self.critic2 = nn.Sequential(*critic2_layers)
            
            print("Initialized DoubleCritic network")
            print(f"Critic1 (Dense Rewards): {self.critic1}")
            print(f"Critic2 (Sparse Rewards): {self.critic2}")
        else:
            # 单Critic（原版）
            critic_layers = []
            critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        """评估状态价值"""
        if self.use_double_critic:
            # 返回两个价值
            value1 = self.critic1(critic_observations)
            value2 = self.critic2(critic_observations)
            return value1, value2
        else:
            # 单一价值（兼容原版）
            value = self.critic(critic_observations)
            return value
    
    def evaluate_critic1(self, critic_observations):
        """评估密集奖励价值"""
        if self.use_double_critic:
            return self.critic1(critic_observations)
        else:
            raise RuntimeError("evaluate_critic1 called but use_double_critic=False")
    
    def evaluate_critic2(self, critic_observations):
        """评估稀疏奖励价值"""
        if self.use_double_critic:
            return self.critic2(critic_observations)
        else:
            raise RuntimeError("evaluate_critic2 called but use_double_critic=False")
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
