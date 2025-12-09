"""
历史观测包装器模块
功能：维护观测历史，为策略网络提供时序信息

策略网络需要历史观测来理解机器人的运动状态和趋势。
例如：当前速度、加速度方向等需要从历史观测中推断。

历史长度：10 步（与 MuJoCo 版本一致）
- 单步观测维度：305 维（48 proprio + 225 heights + 3 priv_explicit + 29 priv_latent）
- 历史观测：只维护 proprio 历史（48 维 × 10 步 = 480 维）
- 完整观测：305 + 480 = 785 维
"""

# import isaacgym

# assert isaacgym, "import isaacgym before pytorch"
import torch


class HistoryWrapper:
    """
    历史观测包装器类
    功能：包装环境代理，维护并返回历史观测序列（与 MuJoCo 版本一致）
    
    工作原理：
    1. 维护一个滑动窗口，只存储最近 N 步的 proprio 观测（48 维）
    2. 每次获取新观测时，提取 proprio 部分（前 48 维），更新历史
    3. 返回完整观测：当前观测（305维）+ proprio历史（480维）= 785维
    
    示例：
    - 历史长度 = 10
    - 当前观测 = [305 维]（48 proprio + 225 heights + 3 priv_explicit + 29 priv_latent）
    - proprio 历史 = [48×10 = 480 维]
    - 完整观测 = [305 + 480 = 785 维]
    """
    def __init__(self, env):
        """
        初始化历史包装器
        
        @param env 被包装的环境代理（如 LCMAgent）
        """
        self.env = env

        # 历史配置（只维护 proprio 历史，与 MuJoCo 版本一致）
        self.obs_history_length = self.env.num_history_length  # 历史长度（10 步）
        self.proprio_dim = 48  # proprio 观测维度
        self.num_obs_history = self.obs_history_length * self.proprio_dim  # 历史观测总维度（48 × 10 = 480）
        
        # 初始化历史观测缓冲区（只存储 proprio）
        # 形状：[num_envs, num_obs_history]
        # 例如：[1, 480] - 1 个环境，480 维 proprio 历史
        self.obs_history = torch.zeros(
            self.env.num_envs, 
            self.num_obs_history, 
            dtype=torch.float,
            device=self.env.device, 
            requires_grad=False  # 不需要梯度（仅用于推理）
        )

    def step(self, action):
        """
        执行一步动作并更新历史观测
        
        @param action 动作张量（形状：[1, 12]）
        @return 字典，包含：
          - 'obs': 当前观测（形状：[1, 305]）
          - 'obs_history': 完整观测（形状：[1, 785]）
        
        历史更新逻辑：
        - 从当前观测提取 proprio 部分（前 48 维）
        - 更新 proprio 历史：移除最旧的（前 48 维），添加新的（后 48 维）
        - 拼接完整观测：当前观测（305维）+ proprio历史（480维）= 785维
        """
        # 执行动作，获取新观测（305 维）
        obs = self.env.step(action)
        
        # 提取 proprio 部分（前 48 维）
        proprio_obs = obs[:, :self.proprio_dim]
        
        # 更新历史观测：滑动窗口（只维护 proprio 历史）
        # 移除最旧的 proprio 观测（前 48 维），添加新的（后 48 维）
        self.obs_history = torch.cat((self.obs_history[:, self.proprio_dim:], proprio_obs), dim=-1)
        
        # 拼接完整观测：当前观测（305维）+ proprio历史（480维）= 785维
        full_obs = torch.cat((obs, self.obs_history), dim=-1)
        
        return {'obs': obs, 'obs_history': full_obs}

    def get_observations(self):
        """
        获取观测（不执行动作）
        
        @return 字典，包含当前观测和完整观测
        
        注意：此方法会更新历史观测，但不执行动作
        """
        obs = self.env.get_observations()
        # 提取 proprio 部分（前 48 维）
        proprio_obs = obs[:, :self.proprio_dim]
        # 更新历史观测（滑动窗口）
        self.obs_history = torch.cat((self.obs_history[:, self.proprio_dim:], proprio_obs), dim=-1)
        # 拼接完整观测
        full_obs = torch.cat((obs, self.obs_history), dim=-1)
        return {'obs': obs, 'obs_history': full_obs}

    def get_obs(self):
        """
        获取当前观测（不执行动作）
        
        @return 字典，包含当前观测和完整观测
        
        这是最常用的方法，在控制循环中调用
        """
        obs = self.env.get_obs()
        # 提取 proprio 部分（前 48 维）
        proprio_obs = obs[:, :self.proprio_dim]
        # 更新历史观测（滑动窗口）
        self.obs_history = torch.cat((self.obs_history[:, self.proprio_dim:], proprio_obs), dim=-1)
        # 拼接完整观测
        full_obs = torch.cat((obs, self.obs_history), dim=-1)
        return {'obs': obs, 'obs_history': full_obs}

    def reset(self):
        """
        重置环境并清空历史观测
        
        @return 字典，包含重置后的观测和历史观测（全为 0）
        
        重置时：
        1. 调用环境的 reset() 方法
        2. 将历史观测缓冲区清零
        3. 返回初始观测和零历史
        """
        ret = self.env.reset()
        # 清空历史观测（全部设为 0）
        self.obs_history[:, :] = 0
        return {"obs": ret, "obs_history": self.obs_history}

    def __getattr__(self, name):
        """
        属性访问代理
        
        功能：如果 HistoryWrapper 没有某个属性，则从被包装的环境代理中获取
        这使得 HistoryWrapper 可以透明地访问环境的所有属性和方法
        
        @param name 属性名称
        @return 环境代理的属性或方法
        
        示例：
        - wrapper.num_obs → env.num_obs
        - wrapper.device → env.device
        - wrapper.se → env.se
        """
        return getattr(self.env, name)
