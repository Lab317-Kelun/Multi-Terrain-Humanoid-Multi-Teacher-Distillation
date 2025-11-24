# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
多教师蒸馏算法实现

多教师蒸馏的核心思想：
1. 教师模型（Teacher）：使用特权信息（privileged information）训练的高性能模型
2. 学生模型（Student）：只能使用受限观察信息，需要学习模仿教师模型的行为
3. 行为克隆（Behavior Cloning）：通过最小化学生动作与教师动作之间的差异来训练学生模型

训练流程：
1. 数据收集：使用学生模型在环境中交互，同时记录教师模型的动作
2. 存储转换：将观察、教师动作、完成标志等存储到DistillationStorage
3. 模型更新：使用存储的数据计算损失，通过反向传播更新学生模型参数
"""

from dataclasses import dataclass
from typing import Union,  Dict, List

import torch
import torch.nn as nn
from tensordict import TensorDict


from rsl_rl.modules.teacher_student import MultiStudentTeacher
from rsl_rl.utils import resolve_optimizer


@dataclass
class DistillationTransition:
    """蒸馏转换数据结构，用于存储单步环境交互的数据
    
    属性：
        observations: 观察数据（TensorDict格式，包含学生和教师的观察）
        privileged_actions: 教师模型基于特权信息生成的动作（目标动作）
        actions: 学生模型生成的动作（用于环境交互）
        rewards: 环境返回的奖励
        dones: 回合结束标志
    """
    observations: Union[TensorDict, None] = None
    privileged_actions: Union[torch.Tensor, None] = None
    actions: Union[torch.Tensor, None] = None
    rewards: Union[torch.Tensor, None] = None
    dones: Union[torch.Tensor, None] = None

    def clear(self) -> None:
        """清空转换数据，释放内存"""
        self.observations = None
        self.privileged_actions = None
        self.actions = None
        self.rewards = None
        self.dones = None


class DistillationStorage:
    """蒸馏数据存储类，用于存储收集到的环境交互数据
    
    存储的数据包括：
    - observations: 观察序列
    - teacher_actions: 教师模型动作序列（作为训练目标）
    - dones: 回合结束标志序列
    """
    
    def __init__(self, num_envs: int, num_transitions_per_env: int, device: str) -> None:
        """
        初始化存储
        
        参数：
            num_envs: 并行环境数量
            num_transitions_per_env: 每个环境收集的转换数量
            device: 存储设备（CPU/GPU）
        """
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        self.clear()

    def add_transition(self, transition: DistillationTransition) -> None:
        """
        添加一个转换到存储中
        
        参数：
            transition: 包含观察、教师动作、完成标志的转换数据
        """
        if transition.observations is None or transition.privileged_actions is None or transition.dones is None:
            raise RuntimeError("Incomplete transition passed to DistillationStorage.")
        # 使用clone()确保数据独立性，避免引用问题
        self.observations.append(transition.observations.clone())
        self.teacher_actions.append(transition.privileged_actions.clone())
        self.dones.append(transition.dones.clone())

    def generator(self):
        """
        生成器函数，用于迭代访问存储的转换数据
        
        返回：
            生成器，每次yield一个(observations, None, teacher_actions, dones)元组
        """
        for i in range(self.num_transitions_per_env):
            yield self.observations[i], None, self.teacher_actions[i], self.dones[i]

    def clear(self) -> None:
        """清空所有存储的数据，准备下一轮数据收集"""
        self.observations: list[TensorDict] = []
        self.teacher_actions: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []


class Distillation:
    """
    蒸馏算法类：用于训练学生模型模仿教师模型的行为
    
    核心流程：
    1. act(): 使用学生模型生成动作，同时记录教师模型的动作
    2. process_env_step(): 处理环境步骤，更新归一化器并存储转换数据
    3. update(): 使用存储的数据进行多轮训练，通过行为克隆损失更新学生模型
    
    关键特性：
    - 支持梯度累积（gradient_length）：累积多个步骤的梯度后再更新
    - 支持多GPU训练：通过all_reduce同步梯度
    - 支持多种损失函数：MSE、Huber等
    """

    policy: MultiStudentTeacher 
    """学生-教师模型，包含学生网络和教师网络"""

    def __init__(
        self,
        policy: MultiStudentTeacher ,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: Union[float, None] = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: Union[Dict, None] = None,
    ) -> None:
        """
        初始化蒸馏算法
        
        参数：
            policy: 学生-教师模型（MultiStudentTeacher实例）
            num_learning_epochs: 每次更新时遍历数据的轮数
            gradient_length: 梯度累积长度，累积多少步后再进行反向传播
            learning_rate: 学习率
            max_grad_norm: 梯度裁剪的最大范数（None表示不裁剪）
            loss_type: 损失函数类型（"mse"或"huber"）
            optimizer: 优化器类型（"adam"等）
            device: 计算设备（"cpu"或"cuda"）
            multi_gpu_cfg: 多GPU配置字典，包含global_rank和world_size
        """
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # 多GPU参数设置
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]  # 当前GPU的全局排名
            self.gpu_world_size = multi_gpu_cfg["world_size"]    # 总GPU数量
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # 蒸馏核心组件
        self.policy = policy
        self.policy.to(self.device)
        self.storage: Union[DistillationStorage, None] = None  # 数据存储，稍后初始化

        # 初始化优化器
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # 初始化转换数据结构（用于单步数据收集）
        self.transition = DistillationTransition()
        self.last_hidden_states = (None, None)  # 用于RNN的隐藏状态（当前未使用）

        # 蒸馏训练参数
        self.num_learning_epochs = num_learning_epochs  # 每次更新时的训练轮数
        self.gradient_length = gradient_length          # 梯度累积长度
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm             # 梯度裁剪阈值

        # 初始化损失函数
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,      # 均方误差损失
            "huber": nn.functional.huber_loss,  # Huber损失（对异常值更鲁棒）
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0  # 更新计数器

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape,
    ) -> None:
        """
        初始化数据存储
        
        参数：
            training_type: 训练类型（当前固定为"distillation"）
            num_envs: 并行环境数量
            num_transitions_per_env: 每个环境收集的转换数量
            obs: 观察数据（用于确定数据形状）
            actions_shape: 动作形状（未使用，为兼容性保留）
        """
        # 创建rollout存储
        self.storage = DistillationStorage(num_envs, num_transitions_per_env, self.device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """
        根据观察生成动作（用于环境交互）
        
        流程：
        1. 使用学生模型生成动作（用于环境交互）
        2. 使用教师模型生成特权动作（作为训练目标）
        3. 记录观察数据
        
        参数：
            obs: 观察数据（TensorDict格式，包含学生和教师的观察组）
            
        返回：
            学生模型生成的动作（用于环境交互）
        """
        # 计算学生模型动作（用于环境交互，需要detach避免梯度传播）
        self.transition.actions = self.policy.act(obs).detach()
        # 计算教师模型动作（基于特权信息，作为训练目标，需要detach）
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        # 记录观察数据（用于后续训练）
        self.transition.observations = obs.clone()
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: Dict[str, torch.Tensor]
    ) -> None:
        """
        处理环境步骤，更新归一化器并存储转换数据
        
        流程：
        1. 更新观察归一化器（使用新的观察数据）
        2. 记录奖励和完成标志
        3. 将完整的转换数据添加到存储中
        4. 清空当前转换数据
        5. 重置模型状态（处理回合结束）
        
        参数：
            obs: 新的观察数据
            rewards: 环境返回的奖励
            dones: 回合结束标志
            extras: 额外的环境信息（未使用）
        """
        # 更新观察归一化器（使用新的观察数据更新统计信息）
        self.policy.update_normalization(obs)

        # 记录奖励和完成标志
        self.transition.rewards = rewards
        self.transition.dones = dones
        
        # 将完整的转换数据添加到存储中
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage before collecting data.")
        self.storage.add_transition(self.transition)
        
        # 清空当前转换数据，准备下一次收集
        self.transition.clear()
        # 重置模型状态（处理RNN隐藏状态等，当dones=True时重置）
        self.policy.reset(dones)

    def update(self) -> Dict[str, float]:
        """
        更新学生模型参数（核心训练函数）
        
        训练流程：
        1. 遍历存储的数据多轮（num_learning_epochs）
        2. 对每个转换数据：
           - 使用学生模型推理得到动作
           - 计算学生动作与教师动作的损失（行为克隆损失）
           - 累积损失和梯度
        3. 当累积步数达到gradient_length时：
           - 进行反向传播
           - 同步多GPU梯度（如果启用）
           - 梯度裁剪（如果启用）
           - 更新参数
        4. 清空存储，准备下一轮数据收集
        
        返回：
            包含平均行为克隆损失的字典
        """
        self.num_updates += 1
        mean_behavior_loss = 0.0  # 用于统计平均损失
        cnt = 0  # 损失计数

        # 多轮训练：对同一批数据训练多轮可以提高数据利用率
        for epoch in range(self.num_learning_epochs):
            # 重置模型状态（用于RNN，当前未使用）
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            if self.storage is None:
                raise RuntimeError("Storage not initialized. Call init_storage before calling update().")

            # 梯度累积的累加器
            accum_loss = 0      # 累积的损失
            accum_steps = 0     # 累积的步数

            # 遍历存储的所有转换数据
            for obs, _, privileged_actions, dones in self.storage.generator():
                # 克隆数据以避免修改原始数据
                obs = obs.clone()
                privileged_actions = privileged_actions.clone()
                
                # 学生模型推理（用于梯度计算，不采样噪声）
                # 这里使用act_inference而不是act，因为训练时不需要动作噪声
                actions = self.policy.act_inference(obs)

                # 行为克隆损失：计算学生动作与教师动作的差异
                # 这是蒸馏的核心：让学生模型学习模仿教师模型的行为
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # 累积损失（用于梯度累积）
                accum_loss = accum_loss + behavior_loss
                accum_steps += 1

                # 统计信息（用于日志记录）
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # 梯度更新步骤：当累积步数达到gradient_length时进行反向传播
                if accum_steps == self.gradient_length:
                    self.optimizer.zero_grad()
                    # 归一化损失：除以步数，避免有效学习率随累积步数变化
                    (accum_loss / accum_steps).backward()
                    
                    # 多GPU训练：同步所有GPU的梯度
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    
                    # 梯度裁剪：防止梯度爆炸
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    
                    # 更新参数
                    self.optimizer.step()
                    # 分离隐藏状态（用于RNN，避免梯度传播到之前的步骤）
                    self.policy.detach_hidden_states()
                    
                    # 重置累加器，准备下一批梯度累积
                    accum_loss = 0
                    accum_steps = 0

                # 重置完成标志（处理RNN状态，当回合结束时重置隐藏状态）
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))
            
            # 处理剩余的累积步骤（如果最后一批不足gradient_length步）
            if accum_steps > 0:
                self.optimizer.zero_grad()
                (accum_loss / accum_steps).backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.policy.detach_hidden_states()
                accum_loss = 0
                accum_steps = 0
        
        # 计算平均损失
        mean_behavior_loss = mean_behavior_loss / max(cnt, 1)
        
        # 清空存储，准备下一轮数据收集
        if self.storage is not None:
            self.storage.clear()
        
        # 保存隐藏状态（用于RNN，当前未使用）
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # 构建损失字典（用于日志记录）
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    def broadcast_parameters(self) -> None:
        """
        将模型参数从主GPU（rank 0）广播到所有GPU
        
        用于多GPU训练的初始化：确保所有GPU上的模型参数一致
        通常在训练开始时调用，或者在加载检查点后调用
        """
        # 获取当前GPU上的模型参数
        model_params = [self.policy.state_dict()]
        # 从rank 0广播模型参数到所有GPU
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 在所有GPU上加载从rank 0广播的参数
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        """
        收集所有GPU的梯度并求平均（梯度同步）
        
        多GPU训练的核心函数：在反向传播后调用，将所有GPU上的梯度进行all_reduce操作
        实现数据并行的梯度同步，确保所有GPU使用相同的梯度更新参数
        
        流程：
        1. 收集所有参数的梯度并展平为一维张量
        2. 使用all_reduce对所有GPU的梯度求和
        3. 除以GPU数量得到平均梯度
        4. 将平均梯度写回各参数的grad属性
        """
        # 收集所有参数的梯度并展平为一维张量
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)  # 拼接所有梯度
        
        # 对所有GPU的梯度求和（all_reduce操作）
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        # 除以GPU数量得到平均梯度
        all_grads /= self.gpu_world_size
        
        # 将平均梯度写回各参数的grad属性
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()  # 参数的元素数量
                # 从共享缓冲区复制数据回参数的梯度
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新偏移量，指向下一个参数
                offset += numel