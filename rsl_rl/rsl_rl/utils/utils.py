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

import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable

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


def resolve_obs_groups(obs: TensorDict, cfg_groups: Dict[str, Iterable[str]] | None, default_sets: Iterable[str] | None = None) -> Dict[str, list[str]]:
    """Resolve observation group names for MultiStudentTeacher.

    Ensures each required set has a non-empty list of tensor keys. When a
    configuration is missing, it falls back to the keys present in the provided
    :class:`TensorDict` sample.
    """

    resolved: Dict[str, list[str]] = {}
    cfg_groups = cfg_groups or {}
    for name, group_list in cfg_groups.items():
        if group_list is None:
            resolved[name] = []
        elif isinstance(group_list, (list, tuple)):
            resolved[name] = list(group_list)
        else:
            resolved[name] = [str(group_list)]

    available_keys = list(obs.keys()) if isinstance(obs, TensorDict) else []
    if default_sets:
        for set_name in default_sets:
            if set_name not in resolved or len(resolved[set_name]) == 0:
                resolved[set_name] = available_keys.copy()

    return resolved


def store_code_state(log_dir: str | None, module_file_paths: list[str]) -> list[str]:
    """Persist git status/diff files for the provided repositories.

    Args:
        log_dir: Output directory where the files should be written.
        module_file_paths: List of file paths that belong to git repositories.

    Returns:
        List of generated file paths.
    """

    if log_dir is None:
        return []
    os.makedirs(log_dir, exist_ok=True)
    saved_paths: list[str] = []
    for module_path in module_file_paths:
        if module_path is None:
            continue
        module_path = Path(module_path).resolve()
        repo_dir = None
        try:
            repo_dir = subprocess.check_output(
                ["git", "-C", str(module_path.parent), "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue

        repo_name = Path(repo_dir).name
        status_file = Path(log_dir) / f"{repo_name}_git_status.txt"
        diff_file = Path(log_dir) / f"{repo_name}_git_diff.patch"

        with status_file.open("w", encoding="utf-8") as fh:
            subprocess.run(["git", "-C", repo_dir, "status", "-sb"], stdout=fh, stderr=subprocess.STDOUT, check=False)
        with diff_file.open("w", encoding="utf-8") as fh:
            subprocess.run(["git", "-C", repo_dir, "diff"], stdout=fh, stderr=subprocess.STDOUT, check=False)

        saved_paths.extend([str(status_file), str(diff_file)])

    return saved_paths