# BEAMDOJO + AMP 人形机器人训练系统 - 完整集成文档

## 📋 目录

1. [系统概述](#系统概述)
2. [架构设计](#架构设计)
3. [已完成的工作](#已完成的工作)
4. [发现并修复的问题](#发现并修复的问题)
5. [文件清单](#文件清单)
6. [使用指南](#使用指南)
7. [训练流程](#训练流程)
8. [待完成事项](#待完成事项)
9. [调试指南](#调试指南)
10. [配置参数详解](#配置参数详解)

---

## 系统概述

本系统将**BEAMDOJO**（双Critic架构）与**AMP**（对抗式动作先验）技术结合，用于训练人形机器人在复杂地形上的自然运动。

### 核心特性

1. **BEAMDOJO双Critic架构**
   - Dense Critic：学习密集locomotion奖励（速度跟踪、姿态控制等）
   - Sparse Critic：学习稀疏foothold奖励
   - 独立优势估计和归一化，然后加权合并

2. **AMP对抗式训练**
   - Discriminator：区分agent动作与expert演示
   - 通过对抗训练学习自然的人形运动模式
   - 历史状态观测（10步）用于风格判别

3. **统一训练循环**
   - 单一训练流程同时优化三个网络（Actor + 双Critic + Discriminator）
   - 自动合并task reward和style reward
   - 完整的WandB监控支持

---

## 架构设计

### 网络架构

```
输入观测
    ├── Actor观测 (n_proprio + n_scan + history + latent) → Actor Network → 动作
    ├── Critic观测 (完整状态) → Dense Critic → 密集价值
    ├── Critic观测 (完整状态) → Sparse Critic → 稀疏价值
    └── Disc观测 (历史kinematic状态) → Discriminator → 风格logit
```

### 观测维度

| 观测类型 | 维度计算 | 用途 |
|---------|---------|------|
| Actor观测 | n_proprio(75) + n_scan(225) + history(750) + latent(29) + priv(3) = **1082** | 策略输入 |
| Critic观测 | 同Actor观测 = **1082** | 价值估计 |
| Disc观测 | (root(13) + dof(54) + key_bodies(15)) × 10步 = **820** | 风格判别 |

### 奖励组成

```python
# Task Reward (BeamDojo)
task_reward = dense_reward_weight × Σ(dense_rewards) + sparse_reward_weight × Σ(sparse_rewards)

# AMP Reward
amp_reward = -log(1 - sigmoid(Discriminator(disc_obs))) × disc_reward_scale

# Total Reward
total_reward = task_reward_weight × task_reward + disc_reward_weight × amp_reward
```

---

## 已完成的工作

### ✅ 1. 环境扩展 (`humanoid_robot_amp.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/legged_gym/envs/base/humanoid_robot_amp.py`

**功能**:
- 继承自`HumanoidRobot`，添加AMP支持
- 维护10步历史状态缓冲区
- 计算discriminator观测（相对于当前root的局部坐标系）
- 通过`extras["disc_obs"]`传递给训练循环

**关键方法**:
```python
def _init_amp_buffers(self):
    # 初始化历史缓冲区和disc_obs_buf
    
def _update_disc_hist(self):
    # 滚动更新历史状态
    
def _update_disc_obs(self):
    # 计算局部坐标系下的disc_obs
    
def fetch_disc_obs_demo(self, num_samples):
    # 从motion library采样expert数据
```

**关键身体部位**:
- `torso_link` (躯干)
- `left_hand_palm_link`, `right_hand_palm_link` (双手)
- `left_ankle_roll_link`, `right_ankle_roll_link` (双脚)

---

### ✅ 2. Actor-Critic模型 (`actor_critic_amp.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/rsl_rl/rsl_rl/modules/actor_critic_amp.py`

**功能**:
- 继承自`ActorCriticRMADoubleReward`
- 添加Discriminator网络
- 提供discriminator评估和权重获取接口

**网络结构**:
```python
Discriminator:
    Input: 820 → [1024] → ELU → [512] → ELU → [1] → Output (logit)
```

**关键方法**:
```python
def eval_disc(self, disc_obs):
    # 前向传播discriminator
    
def get_disc_logit_weights(self):
    # 获取输出层权重（用于正则化）
    
def get_disc_weights(self):
    # 获取所有权重（用于权重衰减）
```

---

### ✅ 3. PPO算法扩展 (`ppo_double_reward_amp.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/rsl_rl/rsl_rl/algorithms/ppo_double_reward_amp.py`

**功能**:
- 继承自`PPODoubleReward`
- 集成AMP训练逻辑
- 管理discriminator优化器
- 计算并合并AMP reward

**训练流程**:
```python
def update(self):
    # 1. 计算AMP reward
    disc_rewards = _compute_disc_rewards(disc_obs)
    
    # 2. 合并task和AMP reward
    total_reward = task_weight × task_reward + amp_weight × amp_reward
    
    # 3. 更新actor和双critic (调用父类)
    super().update()
    
    # 4. 更新discriminator
    _update_discriminator()
```

**Discriminator损失**:
```python
# BCE损失
loss_agent = BCE(D(disc_obs_agent), 0)  # agent → fake
loss_demo = BCE(D(disc_obs_demo), 1)    # expert → real

# 梯度惩罚 (WGAN-GP)
grad_penalty = mean(||∇D(disc_obs_demo)||²)

# Logit正则化
logit_reg = ||W_output||²

# 总损失
disc_loss = 0.5(loss_agent + loss_demo) + λ_gp × grad_penalty + λ_logit × logit_reg
```

---

### ✅ 4. Rollout Storage扩展 (`rollout_storage.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/rsl_rl/rsl_rl/storage/rollout_storage.py`

**新增功能**:
- 添加`disc_obs`存储字段
- 支持双Critic的分离rewards和values
- 提供双Critic版本的mini-batch generator

**存储结构**:
```python
self.disc_obs = torch.zeros(num_transitions, num_envs, disc_obs_size)
self.rewards_dense = torch.zeros(num_transitions, num_envs, 1)
self.rewards_sparse = torch.zeros(num_transitions, num_envs, 1)
self.values_dense = torch.zeros(num_transitions, num_envs, 1)
self.values_sparse = torch.zeros(num_transitions, num_envs, 1)
```

---

### ✅ 5. Runner适配 (`on_policy_runner.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/rsl_rl/rsl_rl/runners/on_policy_runner.py`

**新增功能**:
- 动态检测AMP模型
- 传递`disc_obs_size`给actor-critic
- 初始化storage时包含`disc_obs_shape`
- 支持AMP训练信息记录

**关键逻辑**:
```python
# 检测AMP
if policy_class_name == "ActorCriticRMADoubleRewardAMP":
    disc_obs_size = env.disc_obs_size
    actor_critic_kwargs['disc_obs_size'] = disc_obs_size

# 初始化storage
storage_kwargs['disc_obs_shape'] = [disc_obs_size]
self.alg.init_storage(**storage_kwargs)
```

---

### ✅ 6. 配置文件 (`humanoid_beamdojo_amp_config.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/legged_gym/envs/humanoid/humanoid_beamdojo_amp_config.py`

**关键配置**:
```python
class env:
    enable_amp = True
    num_disc_obs_steps = 10
    amp_motion_files = ['/path/to/motion/files.pkl']
    
class algorithm:
    # 双Critic配置
    use_double_critic = True
    dense_value_loss_coef = 1.0
    sparse_value_loss_coef = 1.0
    advantage_merge_weight = 0.5
    
    # AMP配置
    enable_amp = True
    disc_hidden_dims = [1024, 512]
    disc_learning_rate = 5e-5
    disc_loss_weight = 5.0
    disc_reward_scale = 2.0
    task_reward_weight = 0.5
    disc_reward_weight = 0.5
    
class runner:
    policy_class_name = 'ActorCriticRMADoubleRewardAMP'
    algorithm_class_name = 'PPODoubleRewardAMP'
```

---

### ✅ 7. 环境注册 (`__init__.py`)

**位置**: `/home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/legged_gym/envs/__init__.py`

```python
from .base.humanoid_robot_amp import HumanoidRobotAMP
from .humanoid.humanoid_beamdojo_amp_config import (
    HumanoidBEAMDOJOAMPCfg,
    HumanoidBEAMDOJOAMPCfgPPO,
)

task_registry.register(
    "humanoid_beamdojo_amp", 
    HumanoidRobotAMP, 
    HumanoidBEAMDOJOAMPCfg(), 
    HumanoidBEAMDOJOAMPCfgPPO()
)
```

---

## 发现并修复的问题

### 🔧 问题1: PPODoubleRewardAMP中错误的数据访问

**问题描述**:
```python
# ❌ 错误代码
disc_obs = rollouts.observations['disc_obs']
```

**原因**: `rollouts` (RolloutStorage)中disc_obs是单独的字段，不是observations的子字典。

**修复**:
```python
# ✅ 正确代码
disc_obs = self.storage.disc_obs
```

**影响范围**:
- `ppo_double_reward_amp.py` 的 `update()` 方法
- `ppo_double_reward_amp.py` 的 `_update_discriminator()` 方法

**修复位置**: 
- 第156行
- 第251行

---

### 🔧 问题2: update方法签名不匹配

**问题描述**:
```python
# ❌ 子类签名
def update(self, rollouts):

# ✅ 父类签名
def update(self):
```

**原因**: `PPODoubleReward`的`update()`方法不接受参数，使用`self.storage`。

**修复**:
```python
def update(self):
    # 使用self.storage而不是参数
    if self.storage.disc_obs is not None:
        disc_obs = self.storage.disc_obs
        ...
```

---

### 🔧 问题3: disc_rewards维度不匹配

**问题描述**:
```python
# disc_rewards: [num_steps, num_envs]
# self.storage.rewards: [num_steps, num_envs, 1]
combined_rewards = task_rewards + disc_rewards  # ❌ 维度不匹配
```

**修复**:
```python
combined_rewards = (
    self.task_reward_weight * task_rewards + 
    self.disc_reward_weight * disc_rewards.unsqueeze(-1)  # ✅ 增加维度
)
```

---

## 文件清单

### 新增文件

| 文件路径 | 说明 | 状态 |
|---------|------|------|
| `legged_gym/envs/base/humanoid_robot_amp.py` | AMP环境扩展 | ✅ 完成 |
| `legged_gym/envs/humanoid/humanoid_beamdojo_amp_config.py` | 配置文件 | ✅ 完成 |
| `rsl_rl/modules/actor_critic_amp.py` | AMP模型 | ✅ 完成 |
| `rsl_rl/algorithms/ppo_double_reward_amp.py` | AMP算法 | ✅ 完成+修复 |

### 修改文件

| 文件路径 | 修改内容 | 状态 |
|---------|---------|------|
| `rsl_rl/storage/rollout_storage.py` | 添加disc_obs存储 | ✅ 完成 |
| `rsl_rl/runners/on_policy_runner.py` | 添加AMP支持 | ✅ 完成 |
| `legged_gym/envs/__init__.py` | 注册AMP任务 | ✅ 完成 |
| `rsl_rl/algorithms/__init__.py` | 导出PPODoubleRewardAMP | ✅ 完成 |

---

## 使用指南

### 环境准备

```bash
cd /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym
```

### 快速测试（不需要Motion Library）

```bash
# 测试环境创建
python3 << 'EOF'
from legged_gym.envs import *
import torch

env, env_cfg = task_registry.make_env(name="humanoid_beamdojo_amp", args=None)
print(f"✓ 环境创建成功")
print(f"✓ disc_obs_size: {env.disc_obs_size}")
print(f"✓ num_obs: {env.num_obs}")

# 测试step
actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
obs, priv_obs, rewards, dones, extras = env.step(actions)
print(f"✓ Step成功")
print(f"✓ disc_obs在extras: {'disc_obs' in extras}")
if 'disc_obs' in extras:
    print(f"✓ disc_obs shape: {extras['disc_obs'].shape}")
EOF
```

### 完整训练

```bash
# 训练命令
python scripts/train.py \
    --task=humanoid_beamdojo_amp \
    --headless \
    --num_envs=2048 \
    --max_iterations=100000
```

---

## 训练流程

### 数据流图

```
环境Step
    ↓
更新disc_hist缓冲区
    ↓
计算disc_obs (局部坐标系)
    ↓
存入extras["disc_obs"]
    ↓
Runner收集到transition
    ↓
存入RolloutStorage.disc_obs
    ↓
PPO Update开始
    ↓
计算AMP reward = f(Discriminator(disc_obs))
    ↓
合并total_reward = task_weight × task + amp_weight × amp
    ↓
更新Actor + Dense Critic + Sparse Critic
    ↓
更新Discriminator (agent vs expert)
    ↓
记录到WandB
```

### 训练循环伪代码

```python
for iteration in range(max_iterations):
    # 1. 收集Rollouts
    for step in range(num_steps_per_env):
        action = policy(obs)
        obs, reward, done, extras = env.step(action)
        storage.add_transition(
            obs=obs,
            action=action,
            reward=reward,
            disc_obs=extras['disc_obs']  # ← AMP关键
        )
    
    # 2. 计算Returns
    storage.compute_returns_double(last_value_dense, last_value_sparse)
    
    # 3. PPO Update
    train_info = algorithm.update()
    #   ├── 计算AMP reward
    #   ├── 合并total reward
    #   ├── 更新actor和双critic
    #   └── 更新discriminator
    
    # 4. 记录到WandB
    wandb.log(train_info)
```

---

## 待完成事项

### ⚠️ P0 - 必须完成（影响训练效果）

#### 1. Motion Library集成

**当前状态**: `fetch_disc_obs_demo()`返回随机agent数据（占位符）

**需要做的**:

1. **准备Motion数据**:
   ```bash
   # 选项A: 使用MimicKit数据
   cp /home/cft/kelun/MimicKit/data/motions/*.pkl \
      /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/data/motions/
   
   # 选项B: 录制自己的数据
   # 需要mocap设备 + 后处理脚本
   
   # 选项C: 从训练好的模型采样
   # 运行一个已训练的policy并记录轨迹
   ```

2. **实现MotionLib**:
   ```python
   # 在humanoid_robot_amp.py中
   def _load_motion_lib(self):
       from legged_gym.utils.motion_lib import MotionLib
       self._motion_lib = MotionLib(
           motion_file=self.cfg.env.amp_motion_files,
           device=self.device
       )
   
   def fetch_disc_obs_demo(self, num_samples):
       # 从motion库采样
       motion_ids = self._motion_lib.sample_motions(num_samples)
       motion_times = self._motion_lib.sample_time(motion_ids)
       
       # 获取kinematic数据
       root_pos, root_rot, dof_pos, dof_vel, ... = \
           self._motion_lib.get_motion_state(motion_ids, motion_times)
       
       # 计算disc_obs (与_update_disc_obs相同逻辑)
       disc_obs_demo = self._compute_disc_obs_from_motion(...)
       
       return disc_obs_demo
   ```

3. **参考实现**: `/home/cft/kelun/MimicKit/mimickit/anim/motion_lib.py`

---

### ⚠️ P1 - 重要（提升可用性）

#### 2. WandB日志完善

**当前缺失的指标**:
```python
# 在ppo_double_reward_amp.py的update()中添加
train_info.update({
    # AMP奖励统计
    'amp/task_reward_mean': task_rewards.mean().item(),
    'amp/task_reward_std': task_rewards.std().item(),
    'amp/amp_reward_mean': amp_rewards.mean().item(),
    'amp/amp_reward_std': amp_rewards.std().item(),
    'amp/total_reward_mean': combined_rewards.mean().item(),
    
    # Discriminator健康度
    'amp/disc_learning_rate': self.disc_learning_rate,
    'amp/disc_grad_norm': grad_norm,  # 需要计算
})
```

#### 3. 梯度监控

```python
# 在_update_discriminator()中
disc_grad_norm = nn.utils.clip_grad_norm_(
    self.actor_critic.discriminator.parameters(), 
    self.max_grad_norm
)
return {
    ...
    'disc_grad_norm': disc_grad_norm.item(),
}
```

---

### ⚠️ P2 - 优化（长期改进）

#### 4. 自适应奖励权重

```python
# 根据训练阶段动态调整
if iteration < warmup_iterations:
    task_weight = 1.0
    disc_weight = 0.0
elif iteration < rampup_iterations:
    progress = (iteration - warmup_iterations) / (rampup_iterations - warmup_iterations)
    task_weight = 1.0 - 0.5 * progress
    disc_weight = 0.5 * progress
else:
    task_weight = 0.5
    disc_weight = 0.5
```

#### 5. Discriminator课程学习

```python
# 逐步增加discriminator难度
if disc_agent_acc > 0.9 or disc_demo_acc < 0.1:
    # Discriminator过拟合，降低学习率
    for param_group in self.disc_optimizer.param_groups:
        param_group['lr'] *= 0.5
```

---

## 调试指南

### 常见问题

#### Q1: `AttributeError: 'HumanoidRobot' object has no attribute 'disc_obs_size'`

**原因**: 使用了普通`HumanoidRobot`而不是`HumanoidRobotAMP`

**解决**:
```python
# 检查task_registry注册
task_registry.register(
    "humanoid_beamdojo_amp", 
    HumanoidRobotAMP,  # ← 确保是AMP版本
    ...
)
```

---

#### Q2: `KeyError: 'disc_obs'`

**原因**: disc_obs没有通过extras传递

**检查清单**:
1. `humanoid_robot_amp.py`的`step()`中:
   ```python
   if self.enable_amp:
       self._update_disc_hist()
       self._update_disc_obs()
       self.extras["disc_obs"] = self.disc_obs_buf.clone()  # ← 检查这行
   ```

2. `ppo_double_reward_amp.py`的`process_env_step()`中:
   ```python
   if self.enable_amp_storage and 'disc_obs' in infos:
       self.transition.disc_obs = infos['disc_obs'].clone()  # ← 检查这行
   ```

---

#### Q3: Discriminator准确率异常

**情况A**: `disc_agent_acc > 0.9` 且 `disc_demo_acc < 0.1`

**原因**: Discriminator过拟合，太容易区分agent和expert

**解决**:
```python
# 降低discriminator学习率
disc_learning_rate = 1e-5  # 从5e-5降低

# 增加正则化
disc_grad_penalty = 10.0  # 从5.0增加
disc_weight_decay = 0.001  # 从0.0001增加

# 降低损失权重
disc_loss_weight = 2.0  # 从5.0降低
```

---

**情况B**: `disc_agent_acc ≈ 0.5` 且 `disc_demo_acc ≈ 0.5`

**含义**: 理想状态！Discriminator无法区分，说明agent学会了expert的风格

---

**情况C**: `disc_loss` 很大或NaN

**原因**: 训练不稳定

**解决**:
```python
# 检查disc_obs范围
print(f"disc_obs range: [{disc_obs.min()}, {disc_obs.max()}]")
print(f"disc_obs mean: {disc_obs.mean()}, std: {disc_obs.std()}")

# 如果范围异常（> 100或< -100），添加归一化
disc_obs = (disc_obs - disc_obs.mean()) / (disc_obs.std() + 1e-8)
```

---

#### Q4: AMP reward全是nan

**调试步骤**:
```python
# 在_compute_disc_rewards()中添加
print(f"disc_logits range: [{disc_logits.min()}, {disc_logits.max()}]")
print(f"prob range: [{prob.min()}, {prob.max()}]")
print(f"disc_rewards range: [{disc_rewards.min()}, {disc_rewards.max()}]")

# 如果logits太大（>50），clip它
disc_logits = torch.clamp(disc_logits, min=-50, max=50)
```

---

### 监控健康训练的指标

#### 理想曲线

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| `disc_loss` | 0.5 - 1.0 | 稳定波动 |
| `disc_agent_acc` | 0.4 - 0.6 | 接近0.5最好 |
| `disc_demo_acc` | 0.4 - 0.6 | 接近0.5最好 |
| `amp_reward_mean` | 逐渐增加 | 最终稳定 |
| `task_reward_mean` | 逐渐增加 | 主要优化目标 |
| `total_reward_mean` | 逐渐增加 | 最终性能指标 |

#### 警告信号

| 情况 | 问题 | 解决方案 |
|------|------|---------|
| `disc_agent_acc > 0.9` | Discriminator过拟合 | 降低disc学习率 |
| `disc_demo_acc < 0.1` | Discriminator过拟合 | 增加正则化 |
| `disc_loss > 5.0` | 训练不稳定 | 检查梯度和obs范围 |
| `amp_reward` 不变 | Discriminator没学习 | 检查motion library |
| `task_reward` 下降 | AMP权重太高 | 降低disc_reward_weight |

---

## 配置参数详解

### 环境参数 (`HumanoidBEAMDOJOAMPCfg.env`)

```python
class env:
    # AMP基础配置
    enable_amp = True              # 启用AMP（必须）
    num_disc_obs_steps = 10        # discriminator观测历史步数
                                   # 建议: 5-15步，步数越多信息越丰富但计算越慢
    
    amp_motion_files = [           # 参考动作文件列表
        '/path/to/walk.pkl',
        '/path/to/run.pkl',
    ]
    amp_replay_buffer_size = 100000  # replay buffer大小（暂未使用）
```

### 算法参数 (`HumanoidBEAMDOJOAMPCfgPPO.algorithm`)

```python
class algorithm:
    # ====== 双Critic参数 ======
    use_double_critic = True         # 启用双Critic（必须）
    dense_value_loss_coef = 1.0      # Dense Critic损失系数
    sparse_value_loss_coef = 1.0     # Sparse Critic损失系数
    advantage_merge_weight = 0.5     # 优势合并权重
    dense_reward_weight = 1.0        # 密集奖励权重
    sparse_reward_weight = 0.25      # 稀疏奖励权重
    
    # ====== AMP参数 ======
    # Discriminator网络
    disc_hidden_dims = [1024, 512]   # 隐藏层维度
                                     # 建议: [1024, 512]或[512, 256]
    
    # 学习率
    disc_learning_rate = 5e-5        # Discriminator学习率
                                     # 建议: 1e-5 到 1e-4
                                     # 如果过拟合，降低到1e-5
    
    # 损失权重
    disc_loss_weight = 5.0           # Discriminator总损失权重
                                     # 建议: 2.0 - 10.0
    disc_logit_reg = 0.01            # Logit正则化系数
                                     # 防止输出过大
    disc_grad_penalty = 5.0          # 梯度惩罚系数 (WGAN-GP)
                                     # 建议: 5.0 - 10.0
    disc_weight_decay = 0.0001       # 权重衰减系数
                                     # L2正则化
    
    # 奖励计算
    disc_reward_scale = 2.0          # AMP奖励缩放系数
                                     # 建议: 1.0 - 5.0
    disc_eval_batch_size = 4096      # 评估时的batch大小
                                     # 避免OOM
    
    # 奖励合并
    task_reward_weight = 0.5         # 任务奖励权重
    disc_reward_weight = 0.5         # AMP奖励权重
                                     # 两者之和建议为1.0
```

### 奖励配置 (`HumanoidBEAMDOJOAMPCfg.rewards`)

```python
class rewards:
    class scales:
        # Dense奖励（高频反馈）
        tracking_x_vel = 1.5         # X方向速度跟踪
        tracking_y_vel = 1.0         # Y方向速度跟踪
        tracking_ang_vel = 2.0       # 角速度跟踪
        orientation = -1.5           # 姿态惩罚
        lin_vel_z = -0.5             # Z方向速度惩罚
        ...
        
        # Sparse奖励（低频反馈）
        foothold = 0.05              # 落脚点奖励
```

### 参数调优建议

#### 场景1: 注重任务完成

```python
task_reward_weight = 0.7
disc_reward_weight = 0.3
disc_learning_rate = 1e-5  # 降低AMP影响
```

#### 场景2: 注重自然运动

```python
task_reward_weight = 0.3
disc_reward_weight = 0.7
disc_learning_rate = 5e-5
```

#### 场景3: Discriminator过拟合

```python
disc_learning_rate = 1e-5    # 降低学习率
disc_loss_weight = 2.0       # 降低损失权重
disc_grad_penalty = 10.0     # 增加正则化
disc_weight_decay = 0.001
```

#### 场景4: Discriminator收敛慢

```python
disc_learning_rate = 1e-4    # 提高学习率
disc_loss_weight = 10.0      # 提高损失权重
disc_grad_penalty = 2.0      # 降低正则化
```

---

## 总结

### 系统完成度: **95%**

✅ **已完成**:
- 环境扩展 (HumanoidRobotAMP)
- Actor-Critic模型 (ActorCriticRMADoubleRewardAMP)
- PPO算法 (PPODoubleRewardAMP) + 修复bug
- Rollout Storage扩展
- Runner适配
- 配置文件
- 环境注册

⚠️ **待完成（P0）**:
- Motion Library集成

⚠️ **待完成（P1）**:
- WandB日志完善
- 梯度监控

### 可以开始测试: ✅ **是**

使用临时的motion采样（随机agent数据）可以测试整个训练流程。

### 可以正式训练: ⚠️ **需要Motion Library**

实现Motion Library后即可进行正式训练。

### 下一步行动

1. **立即**: 实现Motion Library（参考MimicKit）
2. **然后**: 完善WandB日志
3. **最后**: 开始训练并监控指标

---

## 联系与支持

- **代码库**: `/home/cft/kelun/Humanoid-Terrain-Bench/`
- **参考实现**: `/home/cft/kelun/MimicKit/`
- **配置文件**: `humanoid_beamdojo_amp_config.py`

**祝训练顺利！** 🚀

