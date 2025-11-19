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
from typing import Dict, Iterable, Tuple, Union

import torch
from tensordict import TensorDict

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
                       default_sets: Iterable[str] | None = None
                       ) -> Tuple[TensorDict, Dict]:
    """Resolve observation groups into TensorDict slices and metadata."""
    if obs_groups_cfg is None:
        raise ValueError("obs_groups configuration is required for distillation training.")
    group_defs = obs_groups_cfg.get("groups")
    if group_defs is None:
        raise ValueError("obs_groups configuration must contain a 'groups' mapping with slice definitions.")

    resolved_cfg = dict(obs_groups_cfg)
    group_slices: Dict[str, slice] = {}
    for name, spec in group_defs.items():
        if "start" not in spec:
            raise ValueError(f"Observation group '{name}' is missing 'start' index.")
        start = int(spec["start"])
        if "length" in spec:
            end = start + int(spec["length"])
        elif "end" in spec:
            end = spec["end"]
        else:
            raise ValueError(f"Observation group '{name}' must define either 'length' or 'end'.")
        group_slices[name] = slice(start, end)
    resolved_cfg["group_slices"] = group_slices

    if default_sets:
        for set_name in default_sets:
            resolved_cfg.setdefault(set_name, list(group_slices.keys()))

    obs_td = tensor_to_obs_groups(obs, resolved_cfg)
    return obs_td, resolved_cfg


def tensor_to_obs_groups(obs: Union[torch.Tensor, TensorDict], obs_groups_cfg: Dict) -> TensorDict:
    """Convert a flat observation tensor to a TensorDict according to group slices."""
    if isinstance(obs, TensorDict):
        return obs
    if obs.ndim != 2:
        raise ValueError("Observations must be a 2D tensor of shape [num_envs, obs_dim].")
    group_slices = obs_groups_cfg.get("group_slices")
    if not group_slices:
        raise ValueError("obs_groups configuration missing 'group_slices'. Call resolve_obs_groups first.")
    group_tensors = {name: obs[:, slc] for name, slc in group_slices.items()}
    return TensorDict(group_tensors, batch_size=[obs.shape[0]])


def store_code_state(log_dir: str | None, repo_paths: Iterable[str]) -> list[str]:
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


def _find_git_root(start_path: Path) -> Path | None:
    for path in [start_path, *start_path.parents]:
        if (path / ".git").exists():
            return path
    return None