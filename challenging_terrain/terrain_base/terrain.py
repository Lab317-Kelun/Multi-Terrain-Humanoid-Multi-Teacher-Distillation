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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import trimesh
import os
from .config import terrain_config


def load_ply_mesh(ply_path, scale=1.0, offset=None):
    """
    加载PLY格式的mesh文件并转换为vertices和triangles
    
    参数:
        ply_path: PLY文件路径
        scale: 缩放因子（默认1.0，不缩放）
        offset: 偏移量 [x, y, z]（可选，用于调整mesh位置）
    
    返回:
        vertices: numpy数组，形状为 (num_vertices, 3)，单位：米
        triangles: numpy数组，形状为 (num_triangles, 3)，顶点索引
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    # 使用trimesh库加载
    mesh = trimesh.load(ply_path)
    if isinstance(mesh, trimesh.Scene):
        # 如果是场景，获取第一个mesh
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    
    # 应用缩放
    if scale != 1.0:
        vertices = vertices * scale
    
    # 应用偏移
    if offset is not None:
        vertices = vertices + np.array(offset, dtype=np.float32)
    
    return vertices, faces


class Terrain:
    def __init__(self, num_robots) -> None:
        cfg = terrain_config
        self.cfg = terrain_config
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        
        if self.type in ["none", 'plane']:
            return
        
        # 如果mesh_type是"ply"，直接从PLY文件加载
        if self.type == "ply":
            ply_path = getattr(cfg, 'ply_path', '/home/cft/kelun/Humanoid-Terrain-Bench/mesh/small_scane.ply')
            ply_scale = getattr(cfg, 'ply_scale', 1.0)
            ply_offset = getattr(cfg, 'ply_offset', None)
            
            print(f"Loading PLY mesh from: {ply_path}")
            self.vertices, self.triangles = load_ply_mesh(ply_path, scale=ply_scale, offset=ply_offset)
            
            # 设置必要的属性以满足后续代码需求（使用默认值）
            self.tot_rows = 100  # 默认值，用于heightsamples和x_edge_mask
            self.tot_cols = 100
            self.x_edge_mask = np.zeros((self.tot_rows, self.tot_cols), dtype=bool)
            self.heightsamples = np.zeros((self.tot_rows, self.tot_cols), dtype=np.float32)
            
            # 设置基本的环境配置
            # terrain_type: 用于标识地形类型，在_get_env_origins中用于设置env_class
            self.terrain_type = np.zeros((1, 1), dtype=np.float32)  # PLY mesh使用单一类型
            self.env_origins = np.array([[[0.0, 0.0, 0.0]]])  # 默认原点
            self.goals = np.zeros((1, 1, cfg.num_goals, 3))
            
            print("Loaded {} vertices".format(self.vertices.shape[0]))
            print("Loaded {} triangles".format(self.triangles.shape[0]))
            return
        
        # 如果不是PLY类型，抛出错误
        raise ValueError(f"Unsupported mesh_type: {self.type}. Only 'ply', 'none', and 'plane' are supported.")
