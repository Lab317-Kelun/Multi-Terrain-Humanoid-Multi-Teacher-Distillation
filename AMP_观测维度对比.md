# AMP观测维度计算对比

## 当前实现（我们的代码）

### 单步观测维度计算

根据 `humanoid_robot_amp.py` 的 `_init_amp_buffers()` 方法：

```python
root_pos_dim = 3  # root_height_obs=True
root_rot_dim = 6  # tan_norm格式
joint_rot_dim = 27 * 6 = 162  # 27个DOF，每个6维tan_norm
key_pos_dim = 7 * 3 = 21  # 7个关键身体部位，每个3维位置
root_vel_dim = 3
root_ang_vel_dim = 3
dof_vel_dim = 27

pos_obs_dim = 3 + 6 + 162 + 21 = 192
vel_obs_dim = 3 + 3 + 27 = 33
single_step_disc_obs_size = 192 + 33 = 225
```

### 关键身体部位（7个）

```python
key_bodies = [
    "torso_link",                    # 躯干中心
    "left_hip_yaw_link",             # 左髋
    "right_hip_yaw_link",            # 右髋
    "left_knee_link",                # 左膝
    "right_knee_link",               # 右膝
    "left_ankle_roll_link",          # 左脚踝
    "right_ankle_roll_link"          # 右脚踝
]
```

### 总维度（10步历史）

```
disc_obs_size = 225 * 10 = 2250维
```

## MimicKit实现

### 单步观测维度计算

根据 `MimicKit/mimickit/envs/deepmimic_env.py` 的 `compute_tar_obs()` 和 `compute_disc_vel_obs()`：

**位置观测（pos_obs）**：
- `root_pos_obs`: 2或3维（根据`root_height_obs`）
- `root_rot_obs`: 6维（tan_norm）
- `joint_rot_obs`: `num_joints * 6`维（每个关节6维tan_norm）
- `key_pos_obs`: `num_key_bodies * 3`维（如果有关键部位）

**速度观测（vel_obs）**：
- `root_vel_obs`: 3维
- `root_ang_vel_obs`: 3维
- `dof_vel`: `num_dof`维

### 关键身体部位（5个）

根据 `MimicKit/data/envs/amp_g1_env.yaml`：

```yaml
key_bodies: [
    "left_ankle_roll_link",   # 左脚踝
    "right_ankle_roll_link",  # 右脚踝
    "head_link",              # 头部
    "left_wrist_yaw_link",    # 左手腕
    "right_wrist_yaw_link"    # 右手腕
]
```

### G1机器人的DOF数量

- **MimicKit**: 29个DOF（包含waist_roll和waist_pitch）
- **我们的URDF**: 27个DOF（waist_roll和waist_pitch是fixed）

### MimicKit的观测维度（假设）

如果MimicKit使用G1机器人（29个DOF，5个key_bodies，root_height_obs=True）：

```
root_pos_dim = 3
root_rot_dim = 6
joint_rot_dim = 29 * 6 = 174
key_pos_dim = 5 * 3 = 15
root_vel_dim = 3
root_ang_vel_dim = 3
dof_vel_dim = 29

pos_obs_dim = 3 + 6 + 174 + 15 = 198
vel_obs_dim = 3 + 3 + 29 = 35
single_step_disc_obs_size = 198 + 35 = 233
```

## 差异分析

### 1. 关键身体部位数量不同

- **MimicKit**: 5个（脚踝、头部、手腕）
- **我们的**: 7个（躯干、髋、膝、脚踝）

**影响**: 
- 我们的key_pos维度: 7 * 3 = 21
- MimicKit的key_pos维度: 5 * 3 = 15
- 差异: 6维

### 2. DOF数量不同

- **MimicKit**: 29个DOF
- **我们的**: 27个DOF

**影响**:
- 我们的joint_rot维度: 27 * 6 = 162
- MimicKit的joint_rot维度: 29 * 6 = 174
- 差异: 12维

### 3. 总维度差异

**我们的单步观测**:
```
225维 = 3 + 6 + 162 + 21 + 3 + 3 + 27
```

**MimicKit的单步观测（估算）**:
```
233维 = 3 + 6 + 174 + 15 + 3 + 3 + 29
```

**差异**: 8维（主要是DOF和key_bodies的差异）

## 对齐建议

### 选项1：完全对齐MimicKit（推荐用于对比实验）

修改 `humanoid_beamdojo_amp_config.py`:

```python
key_bodies = [
    "left_ankle_roll_link",
    "right_ankle_roll_link", 
    "head_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link"
]
```

这样key_pos维度会变成 5 * 3 = 15维，单步观测变成 225 - 6 = 219维。

### 选项2：保持当前配置（推荐用于实际训练）

当前配置更适合locomotion任务：
- 包含更多下半身关键点（髋、膝），有助于学习步态
- 不包含上半身关键点（头部、手腕），减少不必要的信息

## 验证

当前实现的观测维度计算**逻辑正确**，与MimicKit的**计算方式完全一致**，只是：
1. 关键身体部位的选择不同（这是合理的，因为任务不同）
2. DOF数量不同（这是URDF差异导致的，已通过motion_lib的DOF映射处理）

**结论**: 观测维度的计算方式已完全对齐MimicKit，差异仅在于配置选择（key_bodies和DOF数量），这是合理的。

