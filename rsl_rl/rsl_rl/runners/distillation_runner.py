# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Dict, Union

import torch
import torch.distributed as dist
from tensordict import TensorDict

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None

import rsl_rl
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules.teacher_student import MultiStudentTeacher
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state, tensor_to_obs_groups


class DistillationRunner(OnPolicyRunner):
    """
    蒸馏训练运行器：管理多教师蒸馏的完整训练流程
    
    主要职责：
    1. 初始化环境和算法
    2. 管理训练循环（数据收集 + 模型更新）
    3. 处理日志记录和模型保存
    4. 支持多GPU分布式训练
    
    训练循环流程：
    1. 数据收集阶段：使用学生模型在环境中交互，收集观察和教师动作
    2. 模型更新阶段：使用收集的数据训练学生模型，使其模仿教师模型
    3. 日志记录：记录损失、奖励等训练指标
    4. 模型保存：定期保存检查点
    """

    def __init__(self, env: VecEnv, train_cfg: Dict, log_dir: Union[str, None] = None, device: str = "cpu") -> None:
        """
        初始化蒸馏训练运行器
        
        参数：
            env: 向量化环境（支持并行环境）
            train_cfg: 训练配置字典，包含算法、策略等配置
            log_dir: 日志保存目录（None表示不保存日志）
            device: 计算设备（"cpu"或"cuda"）
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]  # 算法配置（Distillation相关）
        self.policy_cfg = train_cfg["policy"]   # 策略配置（MultiStudentTeacher相关）
        self.device = device
        self.env = env

        # 检查并配置多GPU训练
        self._configure_multi_gpu()

        # 存储训练配置
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 每个环境收集的步数
        self.save_interval = self.cfg["save_interval"]          # 模型保存间隔（迭代次数）

        # 日志配置
        self.log_dir = log_dir
        self.writer = None  # 日志写入器（TensorBoard/WandB/Neptune）
        self.logger_type = self.cfg.get("logger_type", "none")  # 日志类型

        # 从环境获取观察数据，用于算法构建
        # 解析观察组：将原始观察分为"policy"（学生）和"teacher"（教师）两组
        raw_obs = self.env.get_observations()
        obs, self.cfg["obs_groups"] = resolve_obs_groups(raw_obs, self.cfg["obs_groups"], default_sets=["teacher"])
        print(f"INFO:Resolved observation groups: {self.cfg['obs_groups']}")
        obs = obs.to(self.device)

        # 创建蒸馏算法实例
        self.alg = self._construct_algorithm(obs)

        # 决定是否禁用日志（多GPU训练时，只有rank 0记录日志）
        # 注意：我们只从rank 0进程（主进程）记录日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # 训练统计信息
        self.tot_timesteps = 0      # 总时间步数
        self.tot_time = 0           # 总训练时间
        self.current_learning_iteration = 0  # 当前学习迭代次数
        self.git_status_repos = [rsl_rl.__file__]  # Git仓库列表（用于代码状态保存）

    def _configure_multi_gpu(self) -> None:
        """
        配置多GPU训练参数
        
        检查是否启用了分布式训练（torch.distributed），并设置相应的参数
        如果启用，获取当前进程的rank和总进程数
        """
        if dist.is_available() and dist.is_initialized():
            # 分布式训练已启用
            self.is_distributed = True
            self.gpu_global_rank = dist.get_rank()      # 当前进程的全局排名（0, 1, 2, ...）
            self.gpu_world_size = dist.get_world_size() # 总进程数（GPU数量）
            self.multi_gpu_cfg = {"global_rank": self.gpu_global_rank, "world_size": self.gpu_world_size}
        else:
            # 单GPU训练
            self.is_distributed = False
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
            self.multi_gpu_cfg = None

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """
        构建蒸馏算法实例
        
        流程：
        1. 创建MultiStudentTeacher策略（包含学生和教师网络）
        2. 创建Distillation算法（包含优化器、损失函数等）
        3. 初始化数据存储
        
        参数：
            obs: 观察数据（用于确定网络输入维度）
            
        返回：
            配置好的Distillation算法实例
        """
        # 初始化策略：创建MultiStudentTeacher实例
        # 从配置中获取类名并动态实例化
        student_teacher_class = eval(self.policy_cfg.pop("class_name"))
        student_teacher: MultiStudentTeacher = student_teacher_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 初始化算法：创建Distillation实例
        # 从配置中获取类名并动态实例化
        # 同样的原理：eval() 将字符串类名转换为类对象
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student_teacher, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 初始化数据存储：为算法分配存储空间
        alg.init_storage(
            "distillation",              # 训练类型
            self.env.num_envs,           # 并行环境数量
            self.num_steps_per_env,      # 每个环境收集的步数
            obs,                         # 观察数据（用于确定形状）
            [self.env.num_actions],      # 动作形状
        )

        return alg
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """
        执行蒸馏训练的主循环
        
        训练流程：
        1. 初始化日志写入器
        2. 检查教师模型是否已加载
        3. 数据收集阶段（Rollout）：
           - 使用学生模型生成动作
           - 在环境中执行动作
           - 记录教师模型的动作（作为训练目标）
           - 存储转换数据
        4. 模型更新阶段：
           - 使用存储的数据训练学生模型
           - 计算行为克隆损失
           - 更新学生模型参数
        5. 日志记录和模型保存
        
        参数：
            num_learning_iterations: 训练迭代次数
            init_at_random_ep_len: 是否随机初始化回合长度（用于探索）
        """
        # 初始化日志写入器
        self._prepare_logging_writer()
        
        # 检查教师模型是否已加载（必须加载教师模型才能进行蒸馏）
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # 随机化初始回合长度（用于探索，增加训练多样性）
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 开始学习：获取初始观察
        obs = tensor_to_obs_groups(self.env.get_observations(), self.cfg["obs_groups"]).to(self.device)
        
        # 获取初始地形ID
        terrain_ids = None
        if hasattr(self.env, "extras") and "terrain_ids" in self.env.extras:
            terrain_ids = self.env.extras["terrain_ids"].to(self.device)
            
        self.train_mode()  # 切换到训练模式（启用dropout等）
        
        # 记录统计信息
        ep_infos = []  # 回合信息列表
        rewbuffer = deque(maxlen=100)  # 奖励缓冲区（最近100个回合）
        lenbuffer = deque(maxlen=100)  # 回合长度缓冲区（最近100个回合）
        cur_reward_sum = torch.zeros(self.env.num_envs, 1, dtype=torch.float, device=self.device)  # 当前回合累计奖励
        cur_episode_length = torch.zeros(self.env.num_envs, 1, dtype=torch.float, device=self.device)  # 当前回合长度
        
        # 确保所有GPU上的参数同步（多GPU训练）
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        
        # 开始训练循环
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # ========== 数据收集阶段（Rollout） ==========
            with torch.inference_mode():  # 禁用梯度计算，加速数据收集
                for _ in range(self.num_steps_per_env):
                    # 采样动作：学生模型生成动作，同时记录教师模型动作
                    actions = self.alg.act(obs, terrain_ids)
                    
                    # 在环境中执行动作
                    obs_tensor, _, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    
                    # 更新地形ID
                    if "terrain_ids" in extras:
                        terrain_ids = extras["terrain_ids"].to(self.device)
                    
                    # 将观察转换为观察组格式并移动到设备
                    obs = tensor_to_obs_groups(obs_tensor, self.cfg["obs_groups"]).to(self.device)
                    
                    # 处理奖励和完成标志
                    #print("INFO:rewards:", rewards)
                    rewards = rewards.to(self.device).unsqueeze(-1)
                    dones = dones.to(self.device).unsqueeze(-1)
                    
                    # 处理环境步骤：更新归一化器并存储转换数据
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    
                    # 记录统计信息（用于日志）
                    if self.log_dir is not None:
                        # 收集回合信息
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        
                        # 更新当前回合的累计奖励和长度
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        
                        # 处理完成的回合：将数据添加到缓冲区并重置
                        done_envs = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[done_envs][:, 0].flatten().cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_envs][:, 0].flatten().cpu().numpy().tolist())
                        cur_reward_sum[done_envs] = 0
                        cur_episode_length[done_envs] = 0

                stop = time.time()
                collection_time = stop - start  # 数据收集时间
                start = stop

            # ========== 模型更新阶段 ==========
            # 使用收集的数据更新学生模型
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start  # 学习时间
            self.current_learning_iteration = it

            # 日志记录和模型保存
            if self.log_dir is not None and not self.disable_logs:
                # 记录训练信息
                self.log(locals())
                # 定期保存模型
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清空回合信息
            ep_infos.clear()
            
            # 保存代码状态（仅在第一次迭代时）
            if it == start_iter and not self.disable_logs:
                # 获取所有diff文件（Git状态）
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果可能，将文件保存到wandb或neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束后保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # 关闭日志写入器
        if self.writer is not None:
            if hasattr(self.writer, "stop"):
                self.writer.stop()
            if hasattr(self.writer, "close"):
                self.writer.close()
            self.writer = None


    def _prepare_logging_writer(self) -> None:
        if self.disable_logs or self.writer is not None:
            return
        if self.logger_type == "tensorboard":
            if self.log_dir and SummaryWriter is not None:
                self.writer = SummaryWriter(self.log_dir, flush_secs=10)
            else:
                print("TensorBoard logging requested, but SummaryWriter is unavailable or log_dir is missing. Disabling logging.")
                self.logger_type = "none"
        elif self.logger_type == "wandb":
            if not self.log_dir:
                print("WandB logging requested, but no log_dir provided. Disabling logging.")
                self.logger_type = "none"
                return
            try:
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
            except ModuleNotFoundError:
                print("WandB logging requested, but wandb is not installed. Disabling logging.")
                self.logger_type = "none"
                return
            entity_override = self.cfg.get("wandb_entity")
            if entity_override:
                os.environ["WANDB_USERNAME"] = str(entity_override)
            name_override = self.cfg.get("wandb_name")
            if name_override:
                os.environ["WANDB_NAME"] = str(name_override)
            group_override = self.cfg.get("wandb_group")
            if group_override:
                os.environ["WANDB_RUN_GROUP"] = str(group_override)
            wandb_cfg = dict(self.cfg)
            wandb_cfg.setdefault("wandb_project", self.cfg.get("wandb_project", "rsl_rl"))
            print(f"Initializing WandB logging to project: {wandb_cfg['wandb_project']}")
            self.writer = WandbSummaryWriter(self.log_dir, flush_secs=10, cfg=wandb_cfg)
            if hasattr(self.writer, "log_config"):
                try:
                    self.writer.log_config(
                        env_cfg=self.cfg.get("env_cfg", {}),
                        runner_cfg=self.cfg.get("runner", {}),
                        alg_cfg=self.cfg.get("algorithm", {}),
                        policy_cfg=self.cfg.get("policy", {}),
                    )
                except Exception as exc:  # pragma: no cover - logging should not block training
                    print(f"Unable to push configuration to WandB: {exc}")
        elif self.logger_type == "neptune":
            if not self.log_dir:
                print("Neptune logging requested, but no log_dir provided. Disabling logging.")
                self.logger_type = "none"
                return
            try:
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
            except ModuleNotFoundError:
                print("Neptune logging requested, but neptune-client is not installed. Disabling logging.")
                self.logger_type = "none"
                return
            self.writer = NeptuneSummaryWriter(self.log_dir, flush_secs=10, cfg=self.cfg)

    def train_mode(self) -> None:
        """切换到训练模式（启用dropout等训练特性）"""
        self.alg.policy.train()

    def eval_mode(self) -> None:
        """切换到评估模式（禁用dropout等训练特性）"""
        self.alg.policy.eval()

    def save(self, path: str, infos=None) -> None:  # noqa: D401 (compat signature)
        """
        保存模型检查点
        
        保存的内容包括：
        - policy_state_dict: 包含学生网络、教师网络、归一化器、动作噪声等所有参数
        - optimizer_state_dict: 优化器状态
        - iter: 当前迭代次数
        - infos: 额外信息
        
        参数：
            path: 保存路径
            infos: 额外的信息字典（可选）
        """
        policy_state = self.alg.policy.state_dict()
        
        # 验证关键参数是否在state_dict中（用于调试）
        expected_keys = ["student", "teacher", "std"]  # 关键参数
        missing_in_state = [key for key in expected_keys if not any(key in k for k in policy_state.keys())]
        if missing_in_state:
            print(f"WARNING: Some expected keys not found in policy.state_dict(): {missing_in_state}")
        
        state = {
            "policy_state_dict": policy_state,                        # 策略网络参数（包含所有子模块）
            "optimizer_state_dict": self.alg.optimizer.state_dict(),  # 优化器状态
            "iter": self.current_learning_iteration,                  # 当前迭代次数
            "infos": infos,                                           # 额外信息
        }
        # 创建目录（如果不存在）
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"Model saved to {path} (keys: {list(state.keys())})")

    def log(self, locs, width: int = 80, pad: int = 35) -> None:
        """
        记录训练日志
        
        参数：
            locs: 包含训练信息的局部变量字典（从learn()函数传递）
            width: 日志输出宽度（未使用）
            pad: 日志对齐填充（未使用）
        """
        # 更新总时间步数和总时间
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # 计算回合统计信息
        ep_string = ""
        if locs["rewbuffer"]:
            mean_reward = statistics.mean(locs["rewbuffer"])              # 平均奖励
            mean_trajectory_length = statistics.mean(locs["lenbuffer"])   # 平均回合长度
            ep_string = f"{'Mean reward':>{pad}} {mean_reward:.2f}\n"
            ep_string += f"{'Mean episode length':>{pad}} {mean_trajectory_length:.2f}\n"
        else:
            mean_reward = 0.0
            mean_trajectory_length = 0.0

        # 计算动作噪声标准差和FPS
        mean_std = self.alg.policy.action_std.mean()  # 平均动作噪声标准差
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # 计算迭代进度信息
        current_iter = locs.get("it", self.current_learning_iteration)
        start_iter = locs.get("start_iter", 0)
        planned_iterations = locs.get("num_learning_iterations")
        if planned_iterations is None:
            total_iterations = locs.get("total_iterations")
            if total_iterations is not None:
                planned_iterations = max(total_iterations - start_iter, 0)
        if not planned_iterations:
            planned_iterations = current_iter - start_iter + 1
        relative_iter = current_iter - start_iter + 1
        scalar_step = relative_iter  # 用于日志记录的步数

        # 构建日志字典
        log_dict = {
            "Loss/learning_rate": self.alg.learning_rate,              # 学习率
            "Policy/mean_noise_std": mean_std.item(),                  # 平均动作噪声标准差
            "Perf/total_fps": fps,                                     # 总FPS（每秒处理的步数）
            "Perf/collection time": locs["collection_time"],           # 数据收集时间
            "Perf/learning_time": locs["learn_time"],                  # 学习时间
            "Train/mean_reward": mean_reward,                          # 平均奖励
            "Train/mean_episode_length": mean_trajectory_length,       # 平均回合长度
        }
        # 添加算法返回的损失（如行为克隆损失）
        log_dict.update({f"Loss/{k}": v for k, v in locs["loss_dict"].items()})

        # 添加回合信息（如果存在）
        if len(locs["ep_infos"]) > 0:
            for key in locs["ep_infos"][0]:
                log_dict[f"Episode/{key}"] = statistics.mean([float(ep_info[key]) for ep_info in locs["ep_infos"]])

        # 写入日志到TensorBoard/WandB/Neptune
        if self.writer is not None and self.logger_type in {"wandb", "tensorboard", "neptune"}:
            for k, v in log_dict.items():
                self.writer.add_scalar(k, v, scalar_step)
        print(f" \033[1m Learning iteration {relative_iter}/{planned_iterations} \033[0m ")
