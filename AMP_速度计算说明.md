# AMP速度计算说明

## 问题

MimicKit提供的motion数据格式是35维：`[root_pos(3), root_rot_exp_map(3), dof_pos(29)]`，**没有速度信息**。

那么速度是如何得到的呢？

## MimicKit的实现

MimicKit在**加载motion数据时**通过**有限差分**预计算所有帧的速度：

### 1. Root线速度 (root_vel)

```python
# MimicKit/mimickit/anim/motion_lib.py 第191-193行
root_vel = torch.zeros_like(root_pos)
root_vel[..., :-1, :] = fps * (root_pos[..., 1:, :] - root_pos[..., :-1, :])
root_vel[..., -1, :] = root_vel[..., -2, :]  # 最后一帧使用倒数第二帧的速度
```

**公式**: `v = fps * (pos[t+1] - pos[t])`

### 2. Root角速度 (root_ang_vel)

```python
# MimicKit/mimickit/anim/motion_lib.py 第195-198行
root_ang_vel = torch.zeros_like(root_pos)
root_drot = torch_util.quat_diff(root_rot[..., :-1, :], root_rot[..., 1:, :])
root_ang_vel[..., :-1, :] = fps * torch_util.quat_to_exp_map(root_drot)
root_ang_vel[..., -1, :] = root_ang_vel[..., -2, :]
```

**方法**: 
1. 计算相邻帧的四元数差分: `dq = quat_diff(q[t+1], q[t])`
2. 将四元数差分转换为指数映射（角速度的表示）
3. 乘以fps得到角速度

### 3. DOF速度 (dof_vel)

```python
# MimicKit/mimickit/anim/motion_lib.py 第200行
dof_vel = self._kin_char_model.compute_frame_dof_vel(joint_rot, dt)
```

**方法**: 使用kinematic character model的方法，从joint_rot计算dof_vel

## 我们的实现

我们的实现在`get_motion_state`中**按需计算**速度（使用有限差分）：

### 1. DOF速度 (dof_vel)

```python
# legged_gym/utils/motion_lib.py 第210-214行
dof0 = apply_dof_mapping(frame0[:, self.dof_pos_start:self.dof_pos_end])
dof1 = apply_dof_mapping(frame1[:, self.dof_pos_start:self.dof_pos_end])
dof_vel = (dof1 - dof0)[:, :self.num_dofs] / self.dt
```

**公式**: `dof_vel = (dof_pos[t+1] - dof_pos[t]) / dt`

### 2. Root速度 (root_vel)

```python
# legged_gym/utils/motion_lib.py 第218-270行 (get_root_vel方法)
# 线速度
root_lin_vel = (root_pos1 - root_pos0) / dt_small

# 角速度（使用四元数差分）
q0_inv = quat_conjugate(root_rot0)
dq = quat_mul(root_rot1, q0_inv)
# ... 转换为角速度
root_ang_vel = axis * theta / dt_small
```

## 对比

| 方面 | MimicKit | 我们的实现 |
|------|----------|-----------|
| **计算时机** | 加载时预计算所有帧 | 按需计算（每次调用时） |
| **root_vel** | `fps * (pos[t+1] - pos[t])` | `(pos[t+1] - pos[t]) / dt` |
| **root_ang_vel** | 四元数差分 → 指数映射 | 四元数差分 → 角速度 |
| **dof_vel** | `kin_char_model.compute_frame_dof_vel` | `(dof[t+1] - dof[t]) / dt` |
| **效率** | 高（预计算） | 中等（按需计算） |
| **正确性** | ✅ 正确 | ✅ 正确（逻辑一致） |

## 结论

1. **速度是通过有限差分从位置/旋转数据计算出来的**，不是从motion文件中直接读取的
2. **我们的实现逻辑正确**，与MimicKit的方法一致
3. **差异**：MimicKit预计算所有帧的速度（更高效），我们按需计算（更灵活但稍慢）
4. **对于训练影响**：两种方法的结果应该是一致的，因为都使用相同的有限差分公式

## 优化建议（可选）

如果需要提高效率，可以像MimicKit一样在`__init__`中预计算所有帧的速度：

```python
# 在__init__中预计算速度
self._frame_root_vel = fps * (self._frame_root_pos[1:] - self._frame_root_pos[:-1])
self._frame_dof_vel = (self._frame_dof_pos[1:] - self._frame_dof_pos[:-1]) / self.dt
# ... 然后在get_motion_state中直接使用预计算的速度
```

但当前的按需计算方式已经足够，且更简单易维护。

