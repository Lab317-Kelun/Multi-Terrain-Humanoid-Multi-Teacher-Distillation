"""
G1 机器人策略部署主程序
功能：加载训练好的 ONNX 策略模型，通过 LCM 与 C++ 控制程序通信，实现实时控制

LCM 消息发布说明：
- 发布到 "pd_plustau_targets" 通道：包含 29 个关节的目标位置和力矩
  * 前 12 个关节（索引 0-11）：来自策略网络输出（左右腿各 6 个）
  * 索引 12-14（腰部 3 个）：当前为 0（未使用）
  * 索引 15-28（手臂 14 个）：当前为 0（未使用）
- C++ 程序（g1_control.cpp）接收后：
  * 使用前 15 个关节（索引 0-14）作为腿部和腰部命令
  * 使用后 14 个关节（索引 15-28）作为手臂命令（来自 arm_action 消息）
"""

import glob
import pickle as pkl
import lcm
import sys

from utils.deployment_runner import DeploymentRunner
from envs.lcm_agent import LCMAgent
from utils.cheetah_state_estimator import StateEstimator
from utils.command_profile import *
import onnxruntime as ort

import pathlib
import os

# os.environ["LCM_DEFAULT_URL"] = "eth0"

# 初始化 LCM 通信（UDP 多播，用于与 C++ 程序通信）
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy():
    """
    加载并运行训练好的策略模型
    
    流程：
    1. 加载 ONNX 格式的策略模型
    2. 初始化状态估计器和命令配置文件
    3. 创建 LCM 代理（负责与 C++ 程序通信）
    4. 启动部署运行器，执行实时控制循环
    """
    # 加载训练好的策略模型路径
    ckpt_path = "/home/unitree/deploy/deploy.onnx"

    # 创建状态估计器（从 LCM 接收机器人状态，估计身体姿态等）
    se = StateEstimator(lc)

    # 控制周期：50Hz（20ms）
    control_dt = 1/50
    # 创建遥控器命令配置文件（处理速度指令等）
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se)

    # 创建 LCM 代理（负责发布关节命令，接收机器人状态）
    hardware_agent = LCMAgent(se, command_profile)
    se.spin()  # 启动状态估计器的 LCM 订阅线程

    # 包装历史信息（用于策略网络的输入）
    from envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    # 加载 ONNX 策略模型
    policy = load_onnx_policy(ckpt_path)

    # 创建部署运行器（协调策略、代理和命令配置）
    deployment_runner = DeploymentRunner(se=None)
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)  # 添加策略模型
    deployment_runner.add_command_profile(command_profile)  # 添加命令配置

    # 最大运行步数（约 200000 秒 = 55.6 小时）
    max_steps = 10000000
    print(f'max steps {max_steps}')

    # 启动部署运行器（开始实时控制循环）
    deployment_runner.run(max_steps=max_steps)


def load_onnx_policy(path):
    """
    加载 ONNX 格式的策略模型
    
    @param path ONNX 模型文件路径
    @return 推理函数，输入观测张量，输出动作张量
    
    注意：策略网络输出 12 个动作（对应左右腿各 6 个关节）
    """
    model = ort.InferenceSession(path)
    def run_inference(input_tensor):
        """
        执行策略推理
        
        @param input_tensor 输入观测张量（形状：[1, obs_dim]）
        @return 输出动作张量（形状：[1, 12]），12 个动作对应左右腿各 6 个关节
        """
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference


if __name__ == '__main__':
    load_and_run_policy()
