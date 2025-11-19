# AMP 观测构建对比检查

## 1. 观测构建流程对比

### MimicKit的流程
1. **Agent观测** (`_update_disc_obs`):
   - 从历史缓冲区获取数据（`root_pos`, `root_rot`, `root_vel`, `root_ang_vel`, `joint_rot`, `dof_vel`, `key_pos`）
   - 调用 `compute_disc_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs)`
   - `compute_disc_obs` 内部调用 `compute_tar_obs` 和 `compute_disc_vel_obs`

2. **Expert观测** (`fetch_disc_obs_demo`):
   - 从motion库采样motion和时间
   - 调用 `_compute_disc_obs_demo`，内部调用 `_fetch_disc_demo_data` 获取数据
   - 调用 `compute_disc_obs` 构建观测

### 我们的流程
1. **Agent观测** (`_update_disc_obs_for_envs`):
   - 从历史缓冲区获取数据（循环处理每一步）
   - 在循环中逐步构建观测（与MimicKit的批量处理不同）

2. **Expert观测** (`fetch_disc_obs_demo`):
   - 从motion库采样motion和时间
   - 在循环中逐步构建观测（与MimicKit的批量处理不同）

## 2. 关键差异分析

### 2.1 处理方式差异
- **MimicKit**: 批量处理所有历史步（`root_pos`形状是`[batch, num_steps, 3]`）
- **我们的实现**: 循环处理每一步（`root_pos`形状是`[batch, 3]`）

**影响**: 我们的实现应该与MimicKit等价，因为我们在循环中处理每一步，最终结果应该相同。

### 2.2 坐标系转换对比

#### MimicKit的`compute_tar_obs`:
```python
# root_pos_obs转换
heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)  # [batch, 1, 4]
heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))  # [batch, num_steps, 4]
heading_inv_rot_flat = heading_inv_rot_expand.reshape(...)  # [batch * num_steps, 4]
root_pos_obs_flat = root_pos_obs.reshape(...)  # [batch * num_steps, 3]
root_pos_obs_flat = quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
```

#### 我们的实现:
```python
# root_pos_obs转换（在循环中，每一步）
heading_inv_rot_expand = heading_inv_rot.unsqueeze(1)  # [batch, 1, 4]
root_pos_obs_flat = root_pos_obs.reshape(-1, 3)  # [batch, 3]
heading_inv_rot_flat = heading_inv_rot_expand.reshape(-1, 4)  # [batch, 4]
root_pos_obs_flat = quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
```

**结论**: 我们的实现应该与MimicKit等价，因为我们在循环中处理每一步，每一步的转换逻辑与MimicKit相同。

### 2.3 key_pos转换对比

#### MimicKit的`compute_tar_obs`:
```python
# key_pos转换
heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)  # [batch, 1, num_steps, 4]
heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))  # [batch, num_steps, num_key_bodies, 4]
heading_inv_rot_flat = heading_inv_rot_expand.reshape(...)  # [batch * num_steps * num_key_bodies, 4]
key_pos_flat = key_pos.reshape(...)  # [batch * num_steps * num_key_bodies, 3]
key_pos_flat = quat_rotate(heading_inv_rot_flat, key_pos_flat)
```

#### 我们的实现:
```python
# key_pos转换（在循环中，每一步）
heading_inv_rot_expand_key = heading_inv_rot.unsqueeze(1).expand(-1, self.num_key_bodies, -1)  # [batch, num_key_bodies, 4]
key_pos_obs_flat = key_pos_obs.reshape(-1, 3)  # [batch * num_key_bodies, 3]
heading_inv_rot_flat_key = heading_inv_rot_expand_key.reshape(-1, 4)  # [batch * num_key_bodies, 4]
key_pos_obs_flat = quat_rotate(heading_inv_rot_flat_key, key_pos_obs_flat)
```

**结论**: 我们的实现应该与MimicKit等价。

## 3. 需要验证的关键点

### 3.1 数据来源一致性
- ✅ Agent数据: 使用IsaacGym的`rigid_body_states`（最准确）
- ✅ Expert数据: 使用motion库和FK计算

### 3.2 观测组件一致性
- ✅ `root_pos_obs`: 相对于`ref_root_pos`
- ✅ `key_pos_obs`: 相对于`root_pos`（不是`ref_root_pos`）
- ✅ `root_rot_obs`: 转换为tan_norm格式（6维）
- ✅ `joint_rot_obs`: 转换为tan_norm格式（6维，每个关节）
- ✅ `root_vel_obs`: 在heading坐标系中（如果`global_obs=False`）
- ✅ `root_ang_vel_obs`: 在heading坐标系中（如果`global_obs=False`）
- ✅ `dof_vel`: 直接使用

### 3.3 坐标系转换一致性
- ✅ `heading_inv_rot`: 使用`calc_heading_quat_inv`计算
- ✅ `root_pos_obs`: 使用`quat_rotate`转换
- ✅ `key_pos_obs`: 使用`quat_rotate`转换
- ✅ `root_rot`: 使用`quat_mul`转换
- ✅ `root_vel_obs`: 使用`quat_rotate`转换
- ✅ `root_ang_vel_obs`: 使用`quat_rotate`转换

### 3.4 维度处理一致性
- ✅ `root_height_obs=True`: 包含z坐标（使用绝对高度）
- ✅ `root_height_obs=False`: 只包含x, y坐标
- ✅ `joint_rot_obs`: `num_dof * 6`维（每个关节6维tan_norm）
- ✅ `key_pos_obs`: `num_key_bodies * 3`维

## 4. 潜在问题

### 4.1 循环处理 vs 批量处理
我们的实现在循环中处理每一步，而MimicKit批量处理所有步。这应该等价，但需要验证。

### 4.2 历史步顺序
需要确保历史步的顺序与MimicKit一致（从旧到新）。

### 4.3 参考帧选择
- Agent观测: 使用当前时刻（`-1`）作为参考帧
- Expert观测: 使用当前时刻作为参考帧

## 5. 建议的验证方法

1. **数值验证**: 使用相同的输入数据，对比我们的输出和MimicKit的输出
2. **维度验证**: 确保所有观测组件的维度与MimicKit一致
3. **范围验证**: 检查观测数值范围是否合理
4. **训练验证**: 观察训练效果，如果训练正常，说明观测构建正确

## 6. 结论

从代码逻辑上看，我们的实现应该与MimicKit一致。主要差异在于：
- 我们使用循环处理每一步，而MimicKit批量处理所有步
- 这应该等价，但需要实际验证

建议：
1. 继续训练，观察训练效果
2. 如果训练效果不好，可以考虑添加数值验证代码
3. 确保所有观测组件的维度与MimicKit一致

