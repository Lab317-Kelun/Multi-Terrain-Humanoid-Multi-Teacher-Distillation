"""
部署运行器模块
功能：协调策略网络、环境代理和命令配置，执行实时控制循环

主要功能：
1. 管理多个环境代理（控制代理、开环代理等）
2. 执行机器人校准（移动到标称姿态）
3. 运行实时控制循环（50Hz）
4. 处理紧急停止和安全检查
"""

import copy
import time
import os

import numpy as np
import torch


class DeploymentRunner:
    """
    部署运行器类
    功能：协调策略网络、环境代理和命令配置，执行实时控制循环
    
    工作流程：
    1. 初始化：注册代理、策略、命令配置
    2. 校准：等待用户按 R2，平滑移动到零位
    3. 控制循环：策略推理 → 执行动作 → 获取观测 → 循环
    4. 安全检查：监控姿态，处理紧急停止
    """
    def __init__(self, experiment_name="unnamed", se=None):
        """
        初始化部署运行器
        
        @param experiment_name 实验名称（未使用）
        @param se 状态估计器（未使用）
        """
        self.agents = {}                    # 环境代理字典 {name: agent}
        self.policy = None                  # 策略网络（ONNX 推理函数）
        self.command_profile = None          # 命令配置文件（速度命令）
        self.se = se                        # 状态估计器（未使用）

        self.control_agent_name = None      # 控制代理名称（用于实时控制）
        self.command_agent_name = None      # 命令代理名称（未使用）



    def add_open_loop_agent(self, agent, name):
        """
        添加开环代理（当前未使用）
        
        @param agent 环境代理
        @param name 代理名称
        """
        self.agents[name] = agent
        # self.logger.add_robot(name, agent.env.cfg)  # 注释掉，logger 未定义

    def add_control_agent(self, agent, name):
        """
        添加控制代理（用于实时控制）
        
        @param agent 环境代理（如 LCMAgent + HistoryWrapper）
        @param name 代理名称（如 "hardware_closed_loop"）
        """
        self.control_agent_name = name
        self.agents[name] = agent

    def set_command_agents(self, name):
        """
        设置命令代理（当前未使用）
        
        @param name 代理名称
        """
        self.command_agent = name

    def add_policy(self, policy):
        """
        添加策略网络
        
        @param policy 策略推理函数（输入观测，输出动作）
        """
        self.policy = policy

    def add_command_profile(self, command_profile):
        """
        添加命令配置文件
        
        @param command_profile 命令配置（如 RCControllerProfile）
        """
        self.command_profile = command_profile


    def calibrate(self, wait=True):
        """
        校准机器人到标称姿态（零位）
        
        功能：
        1. 等待用户按 R2 按钮开始校准
        2. 平滑地将机器人从当前位置移动到标称姿态
        3. 等待用户再次按 R2 开始控制
        
        @param wait 是否等待用户按 R2（True=等待，False=直接执行）
        @return 校准后的观测（包含历史观测）
        
        标称姿态（12 个关节）：
        - 左腿：[-0.1, 0, 0, 0.3, -0.2, 0]
        - 右腿：[-0.1, 0, 0, 0.3, -0.2, 0]
        """
        # 第一步：如果机器人不在标称姿态，平滑移动到标称姿态
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()  # 获取当前观测
                joint_pos = agent.dof_pos[:12]  # 获取当前关节位置（前 12 个：左右腿）
                
                # 标称姿态目标（左右腿各 6 个关节）
                final_goal = np.array([-0.1000,  0.0000,  0.0000,  0.3000, -0.2000,  0.0000, -0.1000,  0.0000,
                    0.0000,  0.3000, -0.2000,  0.0000], dtype=np.float)

                # 等待用户按 R2 开始校准
                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break
                
                # 生成平滑的目标序列（从当前位置到标称姿态）
                target = joint_pos  # 从当前位置开始
                cal_action = np.zeros((agent.num_envs, agent.num_lower_dofs))  # 校准动作
                target_sequence = []  # 目标序列
                
                # 逐步接近目标（每次最多移动 0.05 rad）
                while np.max(np.abs(target - final_goal)) > 0.01:  # 误差阈值：0.01 rad
                    target -= np.clip((target - final_goal), -0.05, 0.05)  # 限制步长
                    target_sequence += [copy.deepcopy(target)]  # 保存中间目标
                
                # 执行平滑移动序列
                for target in target_sequence:
                    next_target = target
                    action_scale = 0.25  # 动作缩放因子（与策略网络一致）

                    # 将目标位置转换为动作（除以缩放因子）
                    next_target = next_target / action_scale
                    cal_action[:, 0:12] = next_target  # 填充前 12 个关节
                    
                    # 执行动作
                    agent.step(torch.from_numpy(cal_action))
                    agent.get_obs()
                    time.sleep(0.05)  # 等待 50ms（20Hz）

                # 等待用户按 R2 开始控制
                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                # 重置所有代理
                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs


    def run(self, num_log_steps=1000000000, max_steps=100000000):
        """
        运行部署控制循环
        
        流程：
        1. 检查必要组件（控制代理、命令配置）
        2. 重置所有代理
        3. 校准机器人到标称姿态
        4. 执行实时控制循环（50Hz）
        5. 处理紧急停止和安全检查
        
        @param num_log_steps 日志记录步数（未使用）
        @param max_steps 最大运行步数（默认 100000000，约 55.6 小时）
        """
        # 检查必要组件
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        # assert self.policy is not None, "cannot deploy, runner has no policy!"  # 注释掉，允许无策略运行
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # TODO: 添加基本通信测试

        # 重置所有代理
        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs
        
        # 校准机器人到标称姿态（等待用户按 R2）
        control_obs = self.calibrate(wait=True)['obs_history']

        # 开始实时控制循环
        try:
            for i in range(max_steps):
                # 1. 策略推理：根据历史观测计算动作
                action = self.policy(control_obs)  # 输入：历史观测 [1, 546]，输出：动作 [1, 12]
                
                # 2. 执行动作：所有代理执行相同的动作
                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].step(action)  # 执行动作，获取新观测

                    if agent_name == self.control_agent_name:
                        control_obs = obs['obs_history']  # 更新历史观测（用于下一轮推理）

                # 3. 安全检查：检测异常姿态（紧急停止）
                rpy = self.agents[self.control_agent_name].se.get_rpy()  # 获取欧拉角
                # 如果横滚或俯仰角超过 1.6 弧度（约 91.7 度），紧急停止
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)  # 立即返回零位（low 参数未使用）

                # 4. 处理 R2 按钮：用户手动触发校准
                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    # 返回零位
                    control_obs = self.calibrate(wait=False)['obs_history']
                    time.sleep(1)  # 等待 1 秒
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    # 等待用户再次按 R2 继续
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

            # 循环结束，返回标称姿态
            control_obs = self.calibrate(wait=False)

        except KeyboardInterrupt:
            # 捕获键盘中断（Ctrl+C），优雅退出
            pass
