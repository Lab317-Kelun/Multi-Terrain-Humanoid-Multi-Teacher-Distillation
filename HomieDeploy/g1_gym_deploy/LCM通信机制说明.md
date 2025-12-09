# LCM 通信机制说明（代码层面）

## 1. LCM vs ROS 对比

### ROS（你熟悉的）
- **消息格式**：`.msg` 文件定义，编译生成 Python/C++ 代码
- **通信方式**：基于 TCP/UDP，有 ROS Master 作为中央节点
- **序列化**：使用 protobuf 或自定义格式
- **发现机制**：通过 ROS Master 发现节点和话题

### LCM（本项目使用）
- **消息格式**：`.lcm` 文件定义，编译生成 Python/C++ 代码
- **通信方式**：基于 UDP 多播（multicast），**无需中央节点**
- **序列化**：使用二进制打包（struct.pack），更轻量
- **发现机制**：通过 UDP 多播自动发现，无需 Master

## 2. LCM 通信流程（代码层面）

### 2.1 初始化 LCM 实例

```python
# Python 代码（lcm_agent.py 第26行）
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
```

**解释**：
- `udpm://`：UDP 多播协议
- `239.255.76.67`：多播组 IP 地址（所有 LCM 节点使用同一个组）
- `7667`：端口号
- `ttl=255`：生存时间（数据包可以跨 255 个网络跳）

**关键点**：所有进程（Python 和 C++）使用**相同的多播地址和端口**，这样它们就能互相通信。

### 2.2 消息定义（.lcm 文件）

```lcm
// pd_tau_targets_lcmt.lcm
struct pd_tau_targets_lcmt
{
    double q_des[29];          // 目标关节位置
    double tau_ff[29];         // 前馈力矩
    int64_t timestamp_us;      // 时间戳
}
```

**编译过程**：
1. `.lcm` 文件 → `lcm-gen` 工具 → 生成 Python/C++ 代码
2. Python：生成 `pd_tau_targets_lcmt.py`（包含 `encode()` 和 `decode()` 方法）
3. C++：生成 `pd_tau_targets_lcmt.hpp` 和 `.cpp`

### 2.3 发布消息（Publisher）

```python
# Python 代码（lcm_agent.py 第220-221行）
command_for_robot = pd_tau_targets_lcmt()
command_for_robot.q_des = self.joint_pos_target      # 填充数据
command_for_robot.tau_ff = self.torques
command_for_robot.timestamp_us = int(time.time() * 10 ** 6)

# 编码并发布
lc.publish("pd_plustau_targets", command_for_robot.encode())
```

**执行步骤**：
1. **创建消息对象**：`pd_tau_targets_lcmt()` 实例
2. **填充数据**：设置各个字段的值
3. **编码**：`encode()` 将 Python 对象转换为二进制字节流
   ```python
   # encode() 内部实现（pd_tau_targets_lcmt.py 第31-34行）
   def _encode_one(self, buf):
       buf.write(struct.pack('>29d', *self.q_des[:29]))    # 29个double，大端序
       buf.write(struct.pack('>29d', *self.tau_ff[:29]))   # 29个double
       buf.write(struct.pack(">q", self.timestamp_us))      # 1个int64
   ```
4. **发布**：`lc.publish(channel, binary_data)`
   - `channel`：字符串通道名（类似 ROS 的 topic）
   - `binary_data`：编码后的二进制数据
   - LCM 通过 UDP 多播发送到网络

### 2.4 订阅消息（Subscriber）

```python
# Python 代码（cheetah_state_estimator.py 第118-121行）
self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
self.bodydate_state_subscription = self.lc.subscribe("body_control_data", self._bodydata_cb)
self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
```

**执行步骤**：
1. **订阅**：`lc.subscribe(channel, callback_function)`
   - `channel`：要订阅的通道名
   - `callback_function`：收到消息时调用的回调函数
2. **回调函数**：收到消息时自动调用
   ```python
   def _imu_cb(self, channel, data):
       # data 是二进制数据
       msg = state_estimator_lcmt.decode(data)  # 解码
       self.euler = np.array(msg.rpy)           # 使用数据
   ```

### 2.5 消息处理循环（Polling）

```python
# Python 代码（cheetah_state_estimator.py 第342-362行）
def poll(self):
    while True:
        timeout = 0.01  # 10ms 超时
        # 使用 select 检查 LCM 文件描述符是否有数据
        rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
        
        if rfds:
            # 有消息到达，处理所有待处理的消息
            self.lc.handle()  # 触发所有已订阅通道的回调函数
```

**执行步骤**：
1. **监听**：`select.select()` 检查 LCM 的 socket 文件描述符
2. **处理**：如果有数据，调用 `lc.handle()`
3. **分发**：`handle()` 内部会：
   - 读取 UDP 数据包
   - 根据通道名找到对应的订阅者
   - 调用注册的回调函数
   - 回调函数中解码消息并使用数据

### 2.6 C++ 端的实现

```cpp
// C++ 代码（g1_control.cpp 第310-312行）
_simpleLCM.subscribe("pd_plustau_targets", &G1Control::handleActionLCM, this);

// 处理线程（在独立线程中运行）
void G1Control::_simpleLCMThread() {
    while (true) {
        int ret = _simpleLCM.handle();  // 阻塞等待消息
        if (ret < 0) break;
    }
}

// 回调函数
void G1Control::handleActionLCM(const lcm::ReceiveBuffer* rbuf,
                                 const std::string& channel,
                                 const pd_tau_targets_lcmt* msg) {
    // msg 已经自动解码为 C++ 对象
    // 直接使用 msg->q_des[i] 等字段
}
```

## 3. 消息编码/解码机制

### 3.1 编码过程（Python → 二进制）

```python
# pd_tau_targets_lcmt.py 第25-34行
def encode(self):
    buf = BytesIO()  # 创建内存缓冲区
    buf.write(pd_tau_targets_lcmt._get_packed_fingerprint())  # 写入消息类型指纹（8字节）
    self._encode_one(buf)  # 编码实际数据
    return buf.getvalue()  # 返回二进制字节流

def _encode_one(self, buf):
    # 使用 struct.pack 将 Python 数据打包为二进制
    buf.write(struct.pack('>29d', *self.q_des[:29]))    # 29个double，大端序，232字节
    buf.write(struct.pack('>29d', *self.tau_ff[:29]))   # 29个double，232字节
    buf.write(struct.pack(">q", self.timestamp_us))     # 1个int64，8字节
    # 总计：8(指纹) + 232 + 232 + 8 = 480字节
```

**关键点**：
- **指纹（Fingerprint）**：消息类型的哈希值，用于验证消息类型
- **大端序（>）**：网络字节序，确保跨平台兼容
- **固定大小**：每个字段大小固定，便于快速解码

### 3.2 解码过程（二进制 → Python）

```python
# pd_tau_targets_lcmt.py 第36-51行
@staticmethod
def decode(data):
    buf = BytesIO(data)  # 将二进制数据转为可读流
    fingerprint = buf.read(8)  # 读取指纹
    if fingerprint != pd_tau_targets_lcmt._get_packed_fingerprint():
        raise ValueError("Decode error")  # 类型不匹配
    return pd_tau_targets_lcmt._decode_one(buf)

@staticmethod
def _decode_one(buf):
    self = pd_tau_targets_lcmt()
    self.q_des = struct.unpack('>29d', buf.read(232))      # 读取232字节，解包为29个double
    self.tau_ff = struct.unpack('>29d', buf.read(232))     # 读取232字节，解包为29个double
    self.timestamp_us = struct.unpack(">q", buf.read(8))[0] # 读取8字节，解包为1个int64
    return self
```

## 4. 网络传输机制

### 4.1 UDP 多播

```
发布者（Python）                   订阅者（C++）
     │                                  │
     │  UDP 多播包                      │
     │  (239.255.76.67:7667)            │
     ├─────────────────────────────────>│
     │                                  │
     │  所有订阅该通道的节点都会收到     │
```

**特点**：
- **无连接**：不需要建立连接（不像 TCP）
- **多播**：一个数据包可以同时发送给多个接收者
- **不可靠**：可能丢包（但机器人控制通常能容忍偶尔丢包）
- **低延迟**：比 TCP 更快，适合实时控制

### 4.2 通道（Channel）机制

```python
# 发布到通道 "pd_plustau_targets"
lc.publish("pd_plustau_targets", data)

# 订阅通道 "pd_plustau_targets"
lc.subscribe("pd_plustau_targets", callback)
```

**类似 ROS 的 topic**：
- 通道名是字符串，用于路由消息
- 发布者和订阅者通过通道名匹配
- 一个通道可以有多个订阅者（一对多）

## 5. 完整通信示例

### Python → C++ 通信流程

```
1. Python 端（lcm_agent.py）
   ├─> 创建消息对象：command_for_robot = pd_tau_targets_lcmt()
   ├─> 填充数据：command_for_robot.q_des = [...]
   ├─> 编码：binary_data = command_for_robot.encode()
   └─> 发布：lc.publish("pd_plustau_targets", binary_data)
       │
       │ UDP 多播 (239.255.76.67:7667)
       │
       ▼
2. 网络传输
   └─> UDP 数据包通过网络发送
       │
       ▼
3. C++ 端（g1_control.cpp）
   ├─> LCM 线程接收 UDP 数据包
   ├─> 根据通道名 "pd_plustau_targets" 找到订阅者
   ├─> 自动解码：msg = pd_tau_targets_lcmt::decode(data)
   └─> 调用回调：handleActionLCM(channel, msg)
       └─> 使用数据：motor_command.q_target = msg->q_des[i]
```

### C++ → Python 通信流程

```
1. C++ 端（g1_control.cpp）
   ├─> 创建消息对象：rc_command_lcmt msg
   ├─> 填充数据：msg.left_stick[0] = gamepad.lx
   ├─> 编码：binary_data = msg.encode()
   └─> 发布：_simpleLCM.publish("rc_command", &msg)
       │
       │ UDP 多播
       │
       ▼
2. Python 端（cheetah_state_estimator.py）
   ├─> poll() 循环检测到数据
   ├─> lc.handle() 处理消息
   ├─> 根据通道名 "rc_command" 找到订阅者
   ├─> 调用回调：_rc_command_cb(channel, data)
   ├─> 解码：msg = rc_command_lcmt.decode(data)
   └─> 使用数据：self.left_stick = msg.left_stick
```

## 6. 与 ROS 的主要区别

| 特性 | ROS | LCM |
|------|-----|-----|
| **中央节点** | 需要 rosmaster | 不需要 |
| **消息格式** | .msg → 生成代码 | .lcm → 生成代码 |
| **通信方式** | TCP/UDP，通过 rosmaster 路由 | UDP 多播，直接通信 |
| **序列化** | protobuf/自定义 | 二进制打包（struct） |
| **发现机制** | 通过 rosmaster | 通过 UDP 多播 |
| **延迟** | 较高（需要 rosmaster） | 较低（直接通信） |
| **适用场景** | 复杂系统，需要服务发现 | 实时控制，低延迟 |

## 7. 总结

**LCM 的核心机制**：
1. **消息定义**：`.lcm` 文件定义数据结构
2. **自动生成**：编译生成 Python/C++ 的 encode/decode 代码
3. **UDP 多播**：所有节点使用相同的多播地址，自动发现
4. **通道路由**：通过通道名匹配发布者和订阅者
5. **二进制序列化**：使用 struct.pack/unpack 高效编码/解码
6. **轮询处理**：订阅者通过 poll() 循环接收消息

**优势**：
- 无需中央节点，更简单
- 低延迟，适合实时控制
- 轻量级，适合嵌入式系统

**劣势**：
- 无消息队列，可能丢包
- 无服务发现，需要手动配置通道名
- 不适合复杂系统（ROS 更适合）

