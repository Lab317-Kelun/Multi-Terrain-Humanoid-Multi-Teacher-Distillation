# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torchact
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn, Union , Dict, List

from rsl_rl.networks import EmpiricalNormalization, HiddenState
from rsl_rl.modules.actor_critic import Actor, get_activation


class MultiStudentTeacher(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: Dict[str, list[str]],
        num_actions: int,
        # Normalization
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        # Actor architecture (defaults aligned with actor_critic.Actor)
        activation: str = "elu",
        scan_encoder_dims: Union[tuple[int], list[int]] = (256, 256, 256),
        priv_encoder_dims: Union[tuple[int], list[int]] = (),
        tanh_encoder_output: bool = False,
        # Student observation layout
        student_num_prop: Union[int, None] = None,
        student_num_scan: Union[int, None] = None,
        student_num_priv_latent: int = 0,
        student_num_priv_explicit: int = 0,
        student_num_hist: int = 0,
        student_actor_hidden_dims: Union[tuple[int], list[int], None] = None,
        # Teacher observation layout
        teacher_num_prop: Union[int, None] = None,
        teacher_num_scan: Union[int, None] = None,
        teacher_num_priv_latent: int = 0,
        teacher_num_priv_explicit: int = 0,
        teacher_num_hist: int = 0,
        teacher_actor_hidden_dims: Union[tuple[int], list[int], None] = None,
        # Backward-compat shims (will be used only if the new args are not provided)
        student_hidden_dims: Union[tuple[int], list[int], None] = None,
        teacher_hidden_dims: Union[tuple[int], list[int], None] = None,
        # Action noise
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        # Encoding behavior
        student_hist_encoding: bool = False,
        teacher_hist_encoding: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "MultiStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # Indicates if teacher has been loaded

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The MultiStudentTeacher module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "The MultiStudentTeacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        # Resolve hidden dims with backward compatibility
        if student_actor_hidden_dims is None:
            student_actor_hidden_dims = student_hidden_dims if student_hidden_dims is not None else [256, 256, 256]
        if teacher_actor_hidden_dims is None:
            teacher_actor_hidden_dims = teacher_hidden_dims if teacher_hidden_dims is not None else [256, 256, 256]

        # Resolve activation module
        activation_mod = get_activation(activation)

        # Store encoding flags
        self.student_hist_encoding = student_hist_encoding
        self.teacher_hist_encoding = teacher_hist_encoding

        # Student Actor
        if None in (student_num_prop, student_num_scan):
            raise ValueError(
                "student_num_prop and student_num_scan must be provided to use Actor architecture for student."
            )
        print(f"Building MultiStudentTeacher with student_num_obs={num_student_obs}, teacher_num_obs={num_teacher_obs}")
        print(f"Student obs groups: {obs_groups['policy']}")
        print(f"Teacher obs groups: {obs_groups['teacher']}")
        print("INFO:Student_num_prop:", student_num_prop)
        print("INFO:Student_num_scan:", student_num_scan)
        print("INFO:Student_num_priv_latent:", student_num_priv_latent)
        print("INFO:Student_num_priv_explicit:", student_num_priv_explicit)
        print("INFO:Student_num_hist:", student_num_hist)
        print("INFO::priv_encoder_dims", priv_encoder_dims)
        self.student = Actor(
            num_prop=student_num_prop,
            num_scan=student_num_scan,
            num_actions=num_actions,
            scan_encoder_dims=list(scan_encoder_dims) if isinstance(scan_encoder_dims, tuple) else scan_encoder_dims,
            actor_hidden_dims=list(student_actor_hidden_dims)
            if isinstance(student_actor_hidden_dims, tuple)
            else student_actor_hidden_dims,
            priv_encoder_dims=list(priv_encoder_dims) if isinstance(priv_encoder_dims, tuple) else list(priv_encoder_dims),
            num_priv_latent=student_num_priv_latent,
            num_priv_explicit=student_num_priv_explicit,
            num_hist=student_num_hist,
            activation=activation_mod,
            tanh_encoder_output=tanh_encoder_output,
        )
        print(f"Student Actor: {self.student}")

        # Student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = nn.Identity()

        # Teacher Actor
        if None in (teacher_num_prop, teacher_num_scan):
            raise ValueError(
                "teacher_num_prop and teacher_num_scan must be provided to use Actor architecture for teacher."
            )
        self.teacher = Actor(
            num_prop=teacher_num_prop,
            num_scan=teacher_num_scan,
            num_actions=num_actions,
            scan_encoder_dims=list(scan_encoder_dims) if isinstance(scan_encoder_dims, tuple) else scan_encoder_dims,
            actor_hidden_dims=list(teacher_actor_hidden_dims)
            if isinstance(teacher_actor_hidden_dims, tuple)
            else teacher_actor_hidden_dims,
            priv_encoder_dims=list(priv_encoder_dims) if isinstance(priv_encoder_dims, tuple) else list(priv_encoder_dims),
            num_priv_latent=teacher_num_priv_latent,
            num_priv_explicit=teacher_num_priv_explicit,
            num_hist=teacher_num_hist,
            activation=activation_mod,
            tanh_encoder_output=tanh_encoder_output,
        )
        self.teacher.eval()
        print(f"Teacher Actor: {self.teacher}")

        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: Union[torch.Tensor, None] = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        """
        更新动作分布（用于采样动作）
        
        注意：这里使用的是self.student网络，因为这是学生模型的动作分布
        """
        # 计算动作均值（使用eval模式，与推理时一致）
        mean = self.student(obs, hist_encoding=self.student_hist_encoding, eval=True)
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """
        学生模型推理（用于训练和推理，返回确定性动作）
        
        注意：确保与训练时的行为一致
        - 使用eval模式（与_update_distribution一致）
        - 返回确定性动作（网络输出均值，无噪声）
        """
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        # 确保使用eval模式进行推理（与训练时的_update_distribution一致）
        return self.student(obs, hist_encoding=self.student_hist_encoding, eval=True)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs, hist_encoding=self.teacher_hist_encoding)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        # print("INFO:obs_groups in get_student_obs:", self.obs_groups["policy"])
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        # print("INFO:obs_list lengths:", [o.shape for o in obs_list])
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: Union[torch.Tensor, None] = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # Make sure teacher is in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters.
        """
        # Check if state_dict contains teacher and student or just teacher parameters
        # 注意：必须先检查"student"键，因为student.actor_backbone.xxx也包含"actor"字符串
        # 如果先检查"actor"，会误判蒸馏checkpoint为PPO checkpoint
        if any("student" in key for key in state_dict):  
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # 情况1：从蒸馏训练的checkpoint加载（包含"student."和"teacher."前缀的键）
            # 同时加载学生网络和教师网络！
            # super().load_state_dict() 会加载整个MultiStudentTeacher模块，包括：
            #   - student网络（student.xxx）
            #   - teacher网络（teacher.xxx）
            #   - student_obs_normalizer（如果有）
            #   - teacher_obs_normalizer（如果有）
            #   - std（动作噪声参数）
            # 这样加载后，学生和教师网络都被加载到不同的网络结构中，
            # 后续可以通过 policy.act_inference() 使用学生网络，
            # 或者通过 policy.evaluate() 使用教师网络
            super().load_state_dict(state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        elif any("actor." in key for key in state_dict):  
            # 情况2：从PPO训练的checkpoint加载（包含"actor."前缀的键，注意是"actor."不是"actor"）
            # 只加载教师网络，因为PPO checkpoint中只有actor网络（作为教师使用）
            # Rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            # Load teacher actor
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # Load normalizer only if available in checkpoint
            if len(teacher_obs_normalizer_state_dict) > 0 and hasattr(self, "teacher_obs_normalizer"):
                # Be lenient here to support checkpoints without normalizer
                self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=False)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Training does not resume
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")