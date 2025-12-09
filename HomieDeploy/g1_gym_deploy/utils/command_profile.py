"""
命令配置文件模块
功能：提供速度命令给策略网络，支持预定义命令序列和实时遥控器命令

命令格式：[vx, vy, vyaw, height]
  - vx: 前进速度（m/s）
  - vy: 侧向速度（m/s）
  - vyaw: 偏航角速度（rad/s）
  - height: 目标高度（m）
"""

import torch


class CommandProfile:
    """
    基础命令配置类
    功能：管理预定义的命令序列（时间序列命令）
    
    用于场景：
    - 测试预定义的运动轨迹
    - 回放记录的命令序列
    """
    def __init__(self, dt, max_time_s=10.):
        """
        初始化命令配置
        
        @param dt 控制周期（秒）
        @param max_time_s 最大时间长度（秒），默认 10 秒
        """
        self.dt = dt                                    # 控制周期
        self.max_timestep = int(max_time_s / self.dt)  # 最大时间步数
        self.commands = torch.zeros((self.max_timestep, 9))  # 命令序列（时间步数 × 9维命令）
        self.start_time = 0                             # 开始时间（用于计算相对时间）

    def get_command(self, t):
        """
        根据时间获取命令
        
        @param t 当前时间（秒，绝对时间）
        @return 命令数组（9 维）
        """
        # 计算当前时间步（相对于开始时间）
        timestep = int((t - self.start_time) / self.dt)
        # 限制在有效范围内
        timestep = min(timestep, self.max_timestep - 1)
        return self.commands[timestep, :]

    def get_buttons(self):
        """
        获取按钮状态（基础类返回 0，表示无按钮）
        
        @return 按钮状态（0）
        """
        return 0

    def reset(self, reset_time):
        """
        重置命令配置（设置新的开始时间）
        
        @param reset_time 重置时间（秒，绝对时间）
        """
        self.start_time = reset_time


class RCControllerProfile(CommandProfile):
    """
    遥控器命令配置类
    功能：从状态估计器获取实时遥控器命令，提供给策略网络
    
    继承自 CommandProfile，但重写了 get_command() 方法，
    从 StateEstimator 实时获取命令，而不是使用预定义序列
    
    速度命令来源（优先级从高到低）：
    1. pedal_command 通道：如果外部程序发布了 pedal_command，优先使用
    2. 遥控器摇杆：从 rc_command 通道的摇杆位置计算速度命令
       - 左摇杆 Y 轴（向上推）→ 前进速度 vx（最大 0.6 m/s）
       - 左摇杆 X 轴（向右推）→ 侧向速度 vy（最大 0.5 m/s）
       - 右摇杆 X 轴（向右推）→ 偏航角速度 vyaw（最大 0.8 rad/s）
       - 右摇杆 Y 轴（向上推）→ 目标高度 height（范围：0.2 ~ 1.28 m）
    """
    def __init__(self, dt, state_estimator, x_scale=1.0, y_scale=1.0, yaw_scale=1.0, probe_vel_multiplier=1.0):
        """
        初始化遥控器命令配置
        
        @param dt 控制周期（秒）
        @param state_estimator 状态估计器实例（用于获取实时命令）
        @param x_scale X 方向速度缩放因子（默认 1.0，未使用）
        @param y_scale Y 方向速度缩放因子（默认 1.0，未使用）
        @param yaw_scale 偏航角速度缩放因子（默认 1.0，未使用）
        @param probe_vel_multiplier 探测速度倍数（默认 1.0，未使用）
        """
        super().__init__(dt)
        self.state_estimator = state_estimator  # 状态估计器（从 LCM 接收命令）
        
        # 速度缩放因子（当前未使用，注释掉的代码中会用到）
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.probe_vel_multiplier = probe_vel_multiplier

        # 按钮触发命令（用于每个按钮的预定义命令，当前未使用）
        self.triggered_commands = {i: None for i in range(4)}  # 4 个按钮的命令配置
        self.currently_triggered = [0, 0, 0, 0]  # 当前触发的按钮状态
        self.button_states = [0, 0, 0, 0]         # 按钮状态

    def get_command(self, t):
        """
        获取实时速度命令（从状态估计器）
        
        @param t 当前时间（秒，未使用，因为命令是实时的）
        @return 速度命令 [vx, vy, vyaw, height]
        
        调用链：
        - 本方法调用 state_estimator.get_command()
        - state_estimator.get_command() 会：
          1. 优先检查是否收到过 pedal_command，如果有则使用
          2. 否则从遥控器摇杆（rc_command）计算速度命令
        
        注意：当前直接从状态估计器获取命令，不进行额外缩放
        注释掉的代码显示了如何使用缩放因子（x_scale, y_scale, yaw_scale）
        """
        # 从状态估计器获取实时命令
        # 内部逻辑：优先使用 pedal_command，否则从遥控器摇杆计算
        command = self.state_estimator.get_command()
        
        # 注释掉的代码：速度缩放（当前未使用，可用于调整遥控器灵敏度）
        # command[0] = command[0] * self.x_scale   # X 方向速度缩放
        # command[1] = command[1] * self.y_scale   # Y 方向速度缩放
        # command[2] = command[2] * self.yaw_scale # 偏航角速度缩放
        
        return command

    def get_buttons(self):
        """
        获取按钮状态（从状态估计器）
        
        @return 按钮状态（R2 按钮状态：0 或 1）
        """
        return self.state_estimator.get_buttons()

