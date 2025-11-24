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

import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union, List

import torch
from tensordict import TensorDict

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")

def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolve the optimizer from the name.

    Args:
        optimizer_name: Name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


def resolve_obs_groups(obs: Union[torch.Tensor, TensorDict],
                       obs_groups_cfg: Dict,
                       default_sets: Union[Iterable[str], None] = None
                       ) -> Tuple[TensorDict, Dict]:
    """
    解析观察组配置，将扁平的观察tensor转换为分组的TensorDict
    
    功能：
    1. 解析配置中的"groups"定义，创建slice对象用于切分tensor
    2. 设置默认的观察组集合（如"policy"、"teacher"）
    3. 调用tensor_to_obs_groups将tensor转换为TensorDict
    
    参数：
        obs: 原始观察数据，可以是：
            - torch.Tensor: 形状为 [num_envs, total_obs_dim] 的扁平tensor
            - TensorDict: 已经是分组格式，直接返回
        obs_groups_cfg: 观察组配置字典，格式如下：
            {
                "groups": {
                    "proprio": {"start": 0, "length": 48},      # 本体感觉：索引0-47
                    "scan": {"start": 48, "length": 1200},      # 扫描数据：索引48-1247
                    "priv_explicit": {"start": 1248, "length": 12},  # 显式特权信息
                    # ...
                },
                "policy": ["proprio", "scan"],      # 学生模型使用的组
                "teacher": ["proprio", "scan", "priv_explicit"],  # 教师模型使用的组
            }
        default_sets: 默认集合名称列表（如["teacher"]），如果配置中没有这些键，
                      则使用所有groups的键作为默认值
    
    返回：
        Tuple[TensorDict, Dict]:
            - TensorDict: 分组后的观察数据，每个组是一个键
            - Dict: 解析后的完整配置（包含group_slices）
    
    示例：
        输入obs: tensor([[0.1, 0.2, ..., 0.9]])  # shape: [1, 1300]
        配置: {
            "groups": {
                "proprio": {"start": 0, "length": 48},
                "scan": {"start": 48, "length": 1200},
            }
        }
        输出TensorDict: {
            "proprio": tensor([[0.1, ..., 0.48]]),  # shape: [1, 48]
            "scan": tensor([[0.49, ..., 0.1248]]),  # shape: [1, 1200]
        }
    """
    if obs_groups_cfg is None:
        raise ValueError("obs_groups configuration is required for distillation training.")
    
    # 获取组定义
    group_defs = obs_groups_cfg.get("groups")
    if group_defs is None:
        raise ValueError("obs_groups configuration must contain a 'groups' mapping with slice definitions.")

    # 复制配置并创建slice对象字典
    resolved_cfg = dict(obs_groups_cfg)
    group_slices: Dict[str, slice] = {}
    
    # 遍历每个组定义，创建slice对象
    for name, spec in group_defs.items():
        if "start" not in spec:
            raise ValueError(f"    group '{name}' is missing 'start' index.")
        start = int(spec["start"])  # 起始索引
        
        # 计算结束索引：可以使用"length"或"end"
        if "length" in spec:
            end = start + int(spec["length"])  # 起始 + 长度 = 结束
        elif "end" in spec:
            end = spec["end"]  # 直接指定结束索引
        else:
            raise ValueError(f"Observation group '{name}' must define either 'length' or 'end'.")
        
        # 创建slice对象，用于后续切分tensor
        # slice(start, end) 等价于 obs[:, start:end]
        group_slices[name] = slice(start, end)
    
    # 将slice对象添加到配置中
    resolved_cfg["group_slices"] = group_slices

    # 设置默认集合（如果配置中没有指定）
    # 例如：如果default_sets=["teacher"]，且配置中没有"teacher"键，
    # 则设置"teacher"为所有组的列表
    if default_sets:
        for set_name in default_sets:
            resolved_cfg.setdefault(set_name, list(group_slices.keys()))

    # 调用tensor_to_obs_groups进行实际转换
    obs_td = tensor_to_obs_groups(obs, resolved_cfg)
    return obs_td, resolved_cfg


def tensor_to_obs_groups(obs: Union[torch.Tensor, TensorDict], obs_groups_cfg: Dict) -> TensorDict:
    """
    将扁平的观察tensor转换为分组的TensorDict
    
    功能：
    使用配置中的group_slices（slice对象）将扁平的2D tensor切分成多个组，
    每个组对应TensorDict中的一个键。
    
    参数：
        obs: 观察数据
            - torch.Tensor: 形状为 [num_envs, total_obs_dim] 的扁平tensor
            - TensorDict: 如果已经是TensorDict，直接返回（无需转换）
        obs_groups_cfg: 观察组配置，必须包含"group_slices"键
            group_slices是一个字典：{组名: slice对象}
            例如：{"proprio": slice(0, 48), "scan": slice(48, 1248)}
    
    返回：
        TensorDict: 分组后的观察数据
            - 每个键对应一个观察组
            - 每个值是一个tensor，形状为 [num_envs, group_dim]
    
    工作原理：
        假设原始obs形状为 [num_envs, total_dim]
        对于每个组，使用slice对象切分：
            obs[:, slice(start, end)] → 得到 [num_envs, end-start] 的tensor
        然后将所有组组合成TensorDict
    
    示例：
        输入：
            obs = tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])  # shape: [1, 6]
            group_slices = {
                "group1": slice(0, 3),   # 索引0-2
                "group2": slice(3, 6),   # 索引3-5
            }
        
        处理过程：
            group1_tensor = obs[:, slice(0, 3)] = obs[:, 0:3] = [[0.1, 0.2, 0.3]]
            group2_tensor = obs[:, slice(3, 6)] = obs[:, 3:6] = [[0.4, 0.5, 0.6]]
        
        输出TensorDict：
            {
                "group1": tensor([[0.1, 0.2, 0.3]]),  # shape: [1, 3]
                "group2": tensor([[0.4, 0.5, 0.6]]),  # shape: [1, 3]
            }
    """
    # 如果已经是TensorDict，直接返回（无需转换）
    if isinstance(obs, TensorDict):
        return obs
    
    # 检查tensor维度：必须是2D [num_envs, obs_dim]
    if obs.ndim != 2:
        raise ValueError("Observations must be a 2D tensor of shape [num_envs, obs_dim].")
    
    # 获取组切分配置（slice对象字典）
    group_slices = obs_groups_cfg.get("group_slices")
    if not group_slices:
        raise ValueError("obs_groups configuration missing 'group_slices'. Call resolve_obs_groups first.")
    
    # 使用slice对象切分tensor，为每个组创建一个tensor
    # obs[:, slc] 等价于 obs[:, start:end]
    # 例如：obs[:, slice(0, 48)] → 获取第0到47列
    group_tensors = {name: obs[:, slc] for name, slc in group_slices.items()}
    
    # 将切分后的tensor组合成TensorDict
    # batch_size=[obs.shape[0]] 指定batch维度大小
    return TensorDict(group_tensors, batch_size=[obs.shape[0]])


def store_code_state(log_dir: Union[str, None], repo_paths: Iterable[str]) -> List[str]:
    """Dump git diff files for the provided repositories into the log directory."""
    if not log_dir:
        return []
    saved_files: list[str] = []
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    for module_path in repo_paths:
        module_path = Path(module_path).resolve()
        repo_root = _find_git_root(module_path)
        if repo_root is None:
            continue
        try:
            diff = subprocess.check_output(["git", "-C", str(repo_root), "diff"], stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if not diff.strip():
            continue
        patch_name = f"{repo_root.name}_code_state.patch"
        patch_path = log_path / patch_name
        patch_path.write_bytes(diff)
        saved_files.append(str(patch_path))
    return saved_files


def _find_git_root(start_path: Path) -> Union[Path, None]:
    for path in [start_path, *start_path.parents]:
        if (path / ".git").exists():
            return path
    return None