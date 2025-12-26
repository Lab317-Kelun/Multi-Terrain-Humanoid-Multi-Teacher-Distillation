# 域随机化功能对照表

## 说明
本文档列出了所有域随机化功能的详细信息，包括作用、配置参数和代码实现位置。

---

## 1. 物理属性随机化（Rigid Body Properties）

### 1.1 基础质量随机化 (randomize_base_mass)
- **作用**: 随机化机器人基座（base）的质量，模拟负载变化
- **配置参数**: 
  - `randomize_base_mass`: 是否启用
  - `added_mass_range`: 质量添加范围 [min, max] (kg)
- **实现位置**: `_process_rigid_body_props()` (line 813-818)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_base_mass:
      rng_mass = self.cfg.domain_rand.added_mass_range
      rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
      props[0].mass += rand_mass
  ```
- **注意**: 只在环境创建时应用

---

### 1.2 质心位移随机化 (randomize_com_displacement)
- **作用**: 随机化基座的质心位置（相对于默认值），比randomize_base_com更精确
- **配置参数**:
  - `randomize_com_displacement`: 是否启用
  - `com_displacement_range`: 质心位移范围 [min, max] (m)
- **实现位置**: 
  - 初始化: `_init_buffers()` (line 1305-1307)
  - 应用: `_process_rigid_body_props()` (line 821-823)
- **代码**:
  ```python
  # 初始化
  if self.cfg.domain_rand.randomize_com_displacement:
      self.com_displacement = torch_rand_float(
          self.cfg.domain_rand.com_displacement_range[0], 
          self.cfg.domain_rand.com_displacement_range[1], 
          (self.num_envs, 3), device=self.device)
  
  # 应用
  if self.cfg.domain_rand.randomize_com_displacement:
      rand_com = self.com_displacement[env_id].cpu().numpy()
      props[0].com = self.default_com + gymapi.Vec3(*rand_com)
  ```
- **注意**: 优先于 `randomize_base_com`，使用绝对位置而非累积偏移

---

### 1.3 连杆质量随机化 (randomize_link_mass)
- **作用**: 随机化所有连杆（link）的质量，不包括base
- **配置参数**:
  - `randomize_link_mass`: 是否启用
  - `link_mass_range`: 质量缩放范围 [min, max] (比例因子)
- **实现位置**: `_process_rigid_body_props()` (line 831-835)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_link_mass:
      rng = self.cfg.domain_rand.link_mass_range
      for i in range(1, len(props)):
          scale = np.random.uniform(rng[0], rng[1])
          props[i].mass = scale * self.default_rigid_body_mass[i].item()
  ```
- **注意**: 只在环境创建时应用，从index 1开始（跳过base）

---

### 1.4 摩擦系数随机化 (randomize_friction)
- **作用**: 随机化地面/接触面的摩擦系数
- **配置参数**:
  - `randomize_friction`: 是否启用
  - `friction_range`: 摩擦系数范围 [min, max]
- **实现位置**: `_process_rigid_shape_props()` (line 758-767)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_friction:
      if env_id==0:
          friction_range = self.cfg.domain_rand.friction_range
          num_buckets = 64
          bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
          friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
          self.friction_coeffs = friction_buckets[bucket_ids]
      for s in range(len(props)):
          props[s].friction = self.friction_coeffs[env_id]
  ```
- **注意**: 使用bucket方法，所有环境共享64个随机值

---

### 1.5 弹性系数随机化 (randomize_restitution)
- **作用**: 随机化碰撞恢复系数（弹性）
- **配置参数**:
  - `randomize_restitution`: 是否启用
  - `restitution_range`: 弹性系数范围 [min, max] (0-1)
- **实现位置**: `_process_rigid_shape_props()` (line 769-776)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_restitution:
      if env_id==0:
          restitution_range = self.cfg.domain_rand.restitution_range
          self.restitution_coeffs = torch_rand_float(
              restitution_range[0], restitution_range[1], 
              (self.num_envs,1), device=self.device)
      for s in range(len(props)):
          props[s].restitution = self.restitution_coeffs[env_id]
  ```
- **注意**: 只在环境创建时应用

---

## 2. 执行器/控制器随机化 (Actuator/Controller)

### 2.1 PD增益随机化 (randomize_kp, randomize_kd)
- **作用**: 随机化PD控制器的比例增益(Kp)和微分增益(Kd)
- **配置参数**:
  - `randomize_kp`: 是否启用Kp随机化
  - `kp_range`: Kp缩放范围 [min, max] (比例因子)
  - `randomize_kd`: 是否启用Kd随机化
  - `kd_range`: Kd缩放范围 [min, max] (比例因子)
- **实现位置**: 
  - 初始化: `_init_buffers()` (line 1296-1299)
  - 应用: `_compute_torques()` (line 1031, 1037)
  - 重置: `reset_idx()` (line 491-494)
- **代码**:
  ```python
  # 初始化
  if self.cfg.domain_rand.randomize_kp:
      self.Kp_factors = torch_rand_float(
          self.cfg.domain_rand.kp_range[0], 
          self.cfg.domain_rand.kp_range[1], 
          (self.num_envs, self.num_actions), device=self.device)
  
  # 应用（P控制模式）
  torques = self.p_gains * self.Kp_factors * (...) - self.d_gains * self.Kd_factors * self.dof_vel
  
  # 重置时更新
  if self.cfg.domain_rand.randomize_kp:
      self.Kp_factors[env_ids] = torch_rand_float(...)
  ```
- **注意**: 每次环境重置时会重新采样

---

### 2.2 电机强度随机化 (randomize_motor)
- **作用**: 随机化电机的强度和阻尼特性
- **配置参数**:
  - `randomize_motor`: 是否启用
  - `motor_strength_range`: 电机强度范围 [min, max] (比例因子)
- **实现位置**: 
  - 初始化: `_init_buffers()` (line 1256-1257)
  - 应用: `_compute_torques()` (line 1030-1033)
- **代码**:
  ```python
  # 初始化
  str_rng = self.cfg.domain_rand.motor_strength_range
  self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(
      2, self.num_envs, self.num_dof, dtype=torch.float, device=self.device) + str_rng[0]
  
  # 应用
  if not self.cfg.domain_rand.randomize_motor:
      torques = self.p_gains * self.Kp_factors * (...) - self.d_gains * self.Kd_factors * (...)
  else:
      torques = self.motor_strength[0] * self.p_gains * self.Kp_factors * (...) - \
                self.motor_strength[1] * self.d_gains * self.Kd_factors * (...)
  ```
- **注意**: 生成2个张量（分别用于Kp和Kd），每个环境每个关节独立

---

### 2.3 关节注入随机化 (randomize_joint_injection)
- **作用**: 在每个时间步向关节注入随机扭矩，模拟外部干扰
- **配置参数**:
  - `randomize_joint_injection`: 是否启用
  - `joint_injection_range`: 注入扭矩范围 [min, max] (相对于torque_limits的比例)
- **实现位置**: 
  - 初始化: `_init_buffers()` (line 1300-1301)
  - 应用: `_compute_torques()` (line 1034, 1038)
  - 重置: `reset_idx()` (line 495-496)
- **代码**:
  ```python
  # 初始化
  if self.cfg.domain_rand.randomize_joint_injection:
      self.joint_injection = torch_rand_float(
          self.cfg.domain_rand.joint_injection_range[0], 
          self.cfg.domain_rand.joint_injection_range[1], 
          (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
  
  # 应用
  torques = torques + self.actuation_offset + self.joint_injection
  
  # 重置时更新
  if self.cfg.domain_rand.randomize_joint_injection:
      self.joint_injection[env_ids] = torch_rand_float(...) * self.torque_limits.unsqueeze(0)
  ```
- **注意**: 在每个控制步骤都应用，每次重置时重新采样

---

### 2.4 执行器偏移随机化 (randomize_actuation_offset)
- **作用**: 随机化执行器的零位偏移，模拟校准误差
- **配置参数**:
  - `randomize_actuation_offset`: 是否启用
  - `actuation_offset_range`: 偏移范围 [min, max] (相对于torque_limits的比例)
- **实现位置**: 
  - 初始化: `_init_buffers()` (line 1302-1303)
  - 应用: `_compute_torques()` (line 1034, 1038)
  - 重置: `reset_idx()` (line 497-498)
- **代码**:
  ```python
  # 初始化
  if self.cfg.domain_rand.randomize_actuation_offset:
      self.actuation_offset = torch_rand_float(
          self.cfg.domain_rand.actuation_offset_range[0], 
          self.cfg.domain_rand.actuation_offset_range[1], 
          (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
  
  # 应用
  torques = torques + self.actuation_offset + self.joint_injection
  
  # 重置时更新
  if self.cfg.domain_rand.randomize_actuation_offset:
      self.actuation_offset[env_ids] = torch_rand_float(...) * self.torque_limits.unsqueeze(0)
  ```
- **注意**: 类似于joint_injection，但在概念上表示系统性的校准误差

---

## 3. 初始状态随机化 (Initial State)

### 3.1 初始关节位置随机化 (randomize_initial_joint_pos)
- **作用**: 随机化重置时的关节初始位置
- **配置参数**:
  - `randomize_initial_joint_pos`: 是否启用
  - `initial_joint_pos_scale`: 位置缩放范围 [min, max] (相对于default_dof_pos)
  - `initial_joint_pos_offset`: 位置偏移范围 [min, max] (绝对偏移，rad)
- **实现位置**: `_reset_dofs()` (line 1055-1058)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_initial_joint_pos:
      init_dos_pos = self.default_dof_pos * torch_rand_float(
          self.cfg.domain_rand.initial_joint_pos_scale[0], 
          self.cfg.domain_rand.initial_joint_pos_scale[1], 
          (len(env_ids), self.num_dof), device=self.device)
      init_dos_pos += torch_rand_float(
          self.cfg.domain_rand.initial_joint_pos_offset[0], 
          self.cfg.domain_rand.initial_joint_pos_offset[1], 
          (len(env_ids), self.num_dof), device=self.device)
      self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
  ```
- **注意**: 先缩放后偏移，最后裁剪到关节限位内

---

### 3.2 起始位置随机化 (randomize_start_pos)
- **作用**: 随机化机器人重置时的XY位置
- **配置参数**:
  - `randomize_start_pos`: 是否启用
  - 硬编码范围: [-0.3, 0.3] (m)
- **实现位置**: `_reset_root_states()` (line 1080-1081)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_start_pos:
      self.root_states[env_ids, :2] += torch_rand_float(
          -0.3, 0.3, (len(env_ids), 2), device=self.device)
  ```
- **注意**: 只在custom_origins模式下应用

---

### 3.3 起始Y位置随机化 (randomize_start_y)
- **作用**: 随机化机器人重置时的Y轴位置（可独立于randomize_start_pos）
- **配置参数**:
  - `randomize_start_y`: 是否启用
  - `rand_y_range`: Y轴随机范围 (m)
- **实现位置**: `_reset_root_states()` (line 1083-1084)
- **代码**:
  ```python
  if self.cfg.domain_rand.randomize_start_y:
      self.root_states[env_ids, 1] += self.cfg.domain_rand.rand_y_range * \
          torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
  ```
- **注意**: 可以叠加在randomize_start_pos之上

---

### 3.4 起始偏航角随机化 (randomize_start_yaw)
- **作用**: 随机化机器人重置时的偏航角（绕Z轴旋转）
- **配置参数**:
  - `randomize_start_yaw`: 是否启用
  - `rand_yaw_range`: 偏航角随机范围 (rad)
- **实现位置**: `_reset_root_states()` (line 1086, 1088-1090)
- **代码**:
  ```python
  rand_yaw = self.cfg.domain_rand.rand_yaw_range * torch_rand_float(
      -1, 1, (len(env_ids), 1), device=self.device).squeeze(1) \
      if self.cfg.domain_rand.randomize_start_yaw else torch.zeros(len(env_ids), device=self.device)
  if self.cfg.domain_rand.randomize_start_yaw or self.cfg.domain_rand.randomize_start_pitch:
      quat = quat_from_euler_xyz(0 * rand_yaw, rand_pitch, rand_yaw)
      self.root_states[env_ids, 3:7] = quat
  ```
- **注意**: 与pitch一起应用到四元数

---

### 3.5 起始俯仰角随机化 (randomize_start_pitch)
- **作用**: 随机化机器人重置时的俯仰角（绕Y轴旋转）
- **配置参数**:
  - `randomize_start_pitch`: 是否启用
  - `rand_pitch_range`: 俯仰角随机范围 (rad)
- **实现位置**: `_reset_root_states()` (line 1087, 1088-1090)
- **代码**:
  ```python
  rand_pitch = self.cfg.domain_rand.rand_pitch_range * torch_rand_float(
      -1, 1, (len(env_ids), 1), device=self.device).squeeze(1) \
      if self.cfg.domain_rand.randomize_start_pitch else torch.zeros(len(env_ids), device=self.device)
  if self.cfg.domain_rand.randomize_start_yaw or self.cfg.domain_rand.randomize_start_pitch:
      quat = quat_from_euler_xyz(0 * rand_yaw, rand_pitch, rand_yaw)
      self.root_states[env_ids, 3:7] = quat
  ```
- **注意**: 与yaw一起应用到四元数

---

## 4. 外部干扰随机化 (External Disturbance)

### 4.1 推力随机化 (push_robots)
- **作用**: 定期给机器人施加随机推力，模拟外部干扰
- **配置参数**:
  - `push_robots`: 是否启用
  - `push_interval_s`: 推力间隔时间 (秒)
  - `max_push_vel_xy`: 最大推力速度 (m/s)
- **实现位置**: 
  - 触发: `_post_physics_step_callback()` (line 859)
  - 应用: `_push_robots()` (line 1097-1102)
- **代码**:
  ```python
  # 触发
  if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
      self._push_robots()
  
  # 应用
  def _push_robots(self):
      max_vel = self.cfg.domain_rand.max_push_vel_xy
      self.root_states[:, 7:9] = torch_rand_float(
          -max_vel, max_vel, (self.num_envs, 2), device=self.device)
      self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
  ```
- **注意**: 直接设置线速度，影响XY方向

---

## 5. 未实现的功能 (Not Implemented) ❌

以下配置参数在配置文件中存在，但在代码中**未实现**：

### 5.1 randomize_payload_mass ❌
- **配置位置**: `humanoid_beamdojo_config.py` line 156-158
- **原因**: 这是上肢相关的功能（torso/hand payload），当前实现不考虑上肢

### 5.2 randomize_body_displacement ❌
- **配置位置**: `humanoid_beamdojo_config.py` line 163-164
- **原因**: 这是上肢相关的功能（torso body displacement），当前实现不考虑上肢

### 5.3 randomize_start_vel ❌
- **配置位置**: `humanoid_beamdojo_config.py` line 194
- **原因**: 配置存在但代码中未使用

---

## 总结

### 已实现的功能数量: 16个
1. randomize_base_mass ✅
2. randomize_com_displacement ✅
3. randomize_link_mass ✅
4. randomize_friction ✅
5. randomize_restitution ✅
6. randomize_kp ✅
7. randomize_kd ✅
8. randomize_motor ✅
9. randomize_joint_injection ✅
10. randomize_actuation_offset ✅
11. randomize_initial_joint_pos ✅
12. randomize_start_pos ✅
13. randomize_start_y ✅
14. randomize_start_yaw ✅
15. randomize_start_pitch ✅
16. push_robots ✅

### 未实现的功能数量: 3个
1. randomize_payload_mass ❌
2. randomize_body_displacement ❌
3. randomize_start_vel ❌

### 注意事项
1. ⚠️ `randomize_base_com` 和 `randomize_com_displacement` 不能同时有效（使用elif）
2. ⚠️ `randomize_start_pos` 和 `randomize_start_y` 可以叠加使用
3. ⚠️ `randomize_start_yaw` 和 `randomize_start_pitch` 一起应用到四元数
4. ✅ 所有执行器相关的随机化（kp, kd, joint_injection, actuation_offset）在reset_idx时都会重新采样

