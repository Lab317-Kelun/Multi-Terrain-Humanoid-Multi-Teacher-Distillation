## 🤖 Assistant

<div style="border: 2px solid #dddddd; border-radius: 10px;">
  <details style="padding: 5px;">
    <summary>已深度思考</summary>
    我们将在现有项目基础上添加两个主要功能：1) 两阶段训练（Stage1和Stage2） 2) 双critic网络（一个处理密集奖励，一个处理稀疏奖励）。<br><br> 现有项目可能已经有一个critic，现在需要将其扩展为两个，并修改训练流程以支持两阶段训练。<br><br> 步骤概述：<br> 1. 修改网络结构，将现有的单一critic改为两个critic（DoubleCritic）。<br> 2. 修改PPO算法，使其能够处理两个critic，分别计算两个奖励组的优势，然后合并。<br> 3. 实现两阶段训练：Stage1在平坦地形上训练，使用目标地形的感知信息，但不因踩空而终止；Stage2在真实地形上微调，踩空会终止。<br> 4. 实现Foothold奖励计算，用于稀疏奖励。<br> 5. 修改训练循环，支持两阶段切换。<br><br> 下面我们逐步实现这些功能。<br><br> 首先，我们修改网络结构，定义DoubleCritic网络。
  </details>
</div>

基于BEAMDOJO论文，我为您提供一个清晰的开发需求文档，用于在现有Humanoid Terrain Bench项目基础上添加Stage1训练和双Critic网络功能。

# BEAMDOJO功能集成开发需求

## 1. 项目概述

### 1.1 目标
在现有Humanoid Terrain Bench项目基础上，集成BEAMDOJO论文中的核心创新：
- **双Critic网络架构**：分离密集locomotion奖励和稀疏foothold奖励的学习
- **两阶段训练流程**：Stage1软约束训练 + Stage2硬约束微调
- **多边形脚foothold奖励**：基于采样的精确落脚点评估

### 1.2 现有基础
- 基于IsaacGym和RSL RL的人形机器人训练框架
- 现有单一Critic网络架构
- 地形生成和课程学习系统
- 基本的locomotion奖励函数

## 2. 核心功能需求

### 2.1 双Critic网络系统

#### 2.1.1 网络架构重构
```python
# 现有：单一Critic
class Critic(nn.Module):
    def __init__(self, obs_dim):
        self.net = MLP(obs_dim, 1)

# 新需求：双Critic
class DoubleCritic(nn.Module):
    def __init__(self, obs_dim):
        self.critic1 = MLP(obs_dim, 1)  # 密集奖励Critic
        self.critic2 = MLP(obs_dim, 1)  # 稀疏奖励Critic
```

#### 2.1.2 价值函数更新逻辑
- **Critic1**：专门学习密集locomotion奖励（速度跟踪、姿态、关节控制等）
- **Critic2**：专门学习稀疏foothold奖励
- 独立的TD-loss计算和参数更新

#### 2.1.3 优势计算与合并
```python
# 分别计算两个Critic的优势
advantage1 = compute_gae(rewards_group1, values1, gamma, lam)
advantage2 = compute_gae(rewards_group2, values2, gamma, lam)

# 归一化后加权合并
total_advantage = (w1 * normalize(advantage1) + 
                   w2 * normalize(advantage2))
```

### 2.2 两阶段训练流程

#### 2.2.1 Stage1 - 软地形动力学约束训练
**配置参数**：
```yaml
stage1:
  terrain_type: "flat_with_target_perception"
  termination_on_misstep: false
  training_terrain: "stones_everywhere"
  commands: 
    vx_range: [-1.0, 1.0]
    vy_range: [-1.0, 1.0] 
    yaw_range: [-1.0, 1.0]
  perception_source: "target_terrain"  # 提供目标地形的高度图
```

**核心特性**：
- 在平坦地形上物理行走
- 接收目标稀疏地形的感知信息
- 踩空只给foothold惩罚，不终止episode
- 使用双Critic分别学习

#### 2.2.2 Stage2 - 硬地形动力学约束训练
**配置参数**：
```yaml
stage2:
  terrain_type: ["stepping_stones", "balancing_beams"]
  termination_on_misstep: true
  commands:
    vx_range: [-1.0, 1.0]
    vy_range: [0.0, 0.0]
    yaw_range: [0.0, 0.0]
  fine_tune_from: "stage1_checkpoint"
```

**核心特性**：
- 在真实稀疏地形上微调
- 踩空立即终止episode
- 只提供x方向命令，无yaw修正
- 继续使用双Critic架构

### 2.3 Foothold奖励系统

#### 2.3.1 多边形脚采样奖励
```python
def compute_foothold_reward(robot, terrain, config):
    """
    基于论文公式(2)的foothold奖励计算
    """
    total_penalty = 0
    for foot_id in [0, 1]:  # 两只脚
        if is_foot_in_contact(foot_id):
            sample_points = sample_foot_points(foot_id, config.n_samples)
            for point in sample_points:
                terrain_height = get_terrain_height(terrain, point)
                if terrain_height < config.epsilon:  # 踩空
                    total_penalty += 1
    return -total_penalty
```

#### 2.3.2 采样参数配置
```yaml
foothold_reward:
  n_samples: 16        # 每只脚采样点数
  epsilon: -0.1        # 深度容忍阈值(m)
  foot_geometry:       # 脚部几何参数
    length: 0.2
    width: 0.1
```

### 2.4 奖励函数重组

#### 2.4.1 Group 1 - 密集Locomotion奖励
```yaml
rewards_group1:
  - xy_velocity_tracking
  - yaw_velocity_tracking  
  - base_height
  - orientation
  - z_velocity
  - roll_pitch_velocity
  - action_rate
  - smoothness
  - joint_velocities
  - joint_accelerations
  - joint_limits
  - joint_power
  - feet_ground_parallel
  - feet_distance
  - feet_air_time
  - feet_clearance
```

#### 2.4.2 Group 2 - 稀疏Foothold奖励
```yaml
rewards_group2:
  - foothold_reward
```

## 3. 技术实现细节

### 3.1 网络架构修改

#### 3.1.1 DoubleCritic类实现
```python
class DoubleCritic(nn.Module):
    def __init__(self, observation_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        # Critic for dense locomotion rewards
        self.critic1 = MLP(observation_dim, 1, hidden_dims)
        # Critic for sparse foothold reward  
        self.critic2 = MLP(observation_dim, 1, hidden_dims)
        
    def forward(self, observations):
        value1 = self.critic1(observations)
        value2 = self.critic2(observations)
        return value1, value2
        
    def get_value1(self, observations):
        return self.critic1(observations)
        
    def get_value2(self, observations):
        return self.critic2(observations)
```

#### 3.1.2 PPO算法适配
修改现有的PPO实现以支持双Critic：
- 分别计算两个Critic的value loss
- 独立的优势估计和归一化
- 合并后的优势用于policy更新

### 3.2 地形系统扩展

#### 3.2.1 两阶段地形生成
```python
class TwoStageTerrainManager:
    def __init__(self, config):
        self.stage = config.initial_stage
        self.curriculum_level = 0
        
    def get_stage1_terrain(self):
        """平坦地形 + 目标地形感知"""
        flat_terrain = generate_flat_terrain()
        target_perception = get_target_terrain_heightmap()
        return flat_terrain, target_perception
        
    def get_stage2_terrain(self):
        """真实稀疏地形"""
        return generate_sparse_terrain(self.curriculum_level)
```

#### 3.2.2 课程学习集成
沿用现有课程学习框架，为两阶段分别配置：
- Stage1: Stones Everywhere地形，逐步增加稀疏度
- Stage2: Stepping Stones + Balancing Beams，逐步增加难度

### 3.3 训练流程控制

#### 3.3.1 阶段切换逻辑
```python
def should_switch_to_stage2(current_policy, metrics):
    """基于性能指标决定是否切换到Stage2"""
    success_rate = metrics['success_rate']
    foothold_accuracy = metrics['foothold_accuracy']
    
    if (success_rate > config.stage1_success_threshold and 
        foothold_accuracy > config.stage1_accuracy_threshold):
        return True
    return False
```

#### 3.3.2 检查点管理
- Stage1训练完成后自动保存检查点
- Stage2从Stage1检查点初始化
- 支持从任意阶段继续训练

## 4. 配置系统

### 4.1 新增配置参数

#### 4.1.1 双Critic配置
```yaml
double_critic:
  enabled: true
  weights:
    w1: 1.0      # locomotion奖励权重
    w2: 0.25     # foothold奖励权重
  network:
    hidden_dims: [512, 256, 128]
    activation: "elu"
```

#### 4.1.2 两阶段训练配置
```yaml
training_stages:
  stage1:
    enabled: true
    max_steps: 5000000
    terrain: "stones_everywhere"
    termination_on_misstep: false
    commands:
      vx_range: [-1.0, 1.0]
      vy_range: [-1.0, 1.0]
      yaw_range: [-1.0, 1.0]
      
  stage2:
    enabled: true  
    max_steps: 5000000
    terrains: ["stepping_stones", "balancing_beams"]
    termination_on_misstep: true
    commands:
      vx_range: [-1.0, 1.0]
      vy_range: [0.0, 0.0]
      yaw_range: [0.0, 0.0]
```

#### 4.1.3 Foothold奖励配置
```yaml
foothold_reward:
  enabled: true
  n_samples: 16
  epsilon: -0.1
  weight: 1.0
```

## 5. 集成到现有项目

### 5.1 文件结构修改
```
humanoid_terrain_bench/
├── networks/
│   ├── double_critic.py          # 新增
│   └── existing_networks.py
├── algorithms/
│   ├── ppo_double_critic.py      # 修改
│   └── existing_algorithms.py
├── envs/
│   ├── terrain_stage_manager.py   # 新增
│   └── existing_envs.py
├── rewards/
│   ├── foothold_reward.py        # 新增
│   └── existing_rewards.py
└── configs/
    ├── beamdojo.yaml             # 新增
    └── existing_configs.yaml
```

### 5.2 向后兼容性
- 保持现有单一Critic模式的兼容性
- 通过配置文件开关启用/禁用BEAMDOJO功能
- 现有训练脚本无需修改即可继续工作

## 6. 验证和测试计划

### 6.1 功能验证
- [ ] 双Critic网络前向/反向传播正确
- [ ] 两阶段地形生成和切换正常
- [ ] Foothold奖励计算准确
- [ ] 优势计算和合并逻辑正确

### 6.2 性能验证
- [ ] Stage1训练收敛性测试
- [ ] Stage2微调效果验证
- [ ] 与单一Critic的对比实验
- [ ] 稀疏地形上的成功率测试

### 6.3 消融实验
- [ ] 有/无Stage1训练对比
- [ ] 有/无双Critic对比
- [ ] 不同奖励权重的影响

## 7. 预期成果

### 7.1 性能指标
- **成功率提升**：在稀疏地形上达到90%+成功率
- **学习效率**：相比单一Critic减少30-50%训练时间
- **foot放置精度**：foothold错误率低于10%

### 7.2 交付物
1. 完整的双Critic网络实现
2. 两阶段训练流程
3. Foothold奖励计算模块
4. 配置文件和示例脚本
5. 验证测试结果

这个需求文档应该足够清晰，让Copilot Claude能够在您现有代码基础上进行系统性的开发。需要我进一步详细说明某个具体模块吗？
