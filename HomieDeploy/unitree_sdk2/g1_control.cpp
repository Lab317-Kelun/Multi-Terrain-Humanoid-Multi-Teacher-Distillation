/**
 * G1 机器人全身控制部署程序（不含手部）
 * 功能：通过 LCM 接收策略网络的关节命令，通过 DDS 发送给机器人执行
 */

#include <yaml-cpp/yaml.h>  // YAML 配置文件解析

#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <lcm/lcm-cpp.hpp>  // LCM 通信库（轻量级消息传递）
#include <thread>

// DDS 通信（数据分发服务）
#include <unitree/robot/channel/channel_publisher.hpp>    // DDS 发布者
#include <unitree/robot/channel/channel_subscriber.hpp>   // DDS 订阅者

// IDL 接口定义（机器人底层命令和状态）
#include <unitree/idl/hg/LowCmd_.hpp>   // G1 底层命令消息
#include <unitree/idl/hg/LowState_.hpp>  // G1 底层状态消息

// LCM 消息类型定义
#include "pd_tau_targets_lcmt.hpp"        // PD 控制目标（位置+力矩）
#include "state_estimator_lcmt.hpp"       // 状态估计器数据（身体姿态、角速度等）
#include "body_control_data_lcmt.hpp"     // 身体控制数据（关节位置、速度）
#include "rc_command_lcmt.hpp"            // 遥控器命令（摇杆、按钮）
#include "arm_action_lcmt.hpp"            // 手臂动作命令
// 游戏手柄/遥控器
#include "unitree/common/thread/thread.hpp"              // 线程工具
#include "unitree/idl/go2/WirelessController_.hpp"       // 无线控制器消息定义
#include "example/wireless_controller/advanced_gamepad.hpp"  // 高级游戏手柄处理类

// DDS 主题定义
#define TOPIC_JOYSTICK "rt/wirelesscontroller"  // 遥控器主题

static const std::string HG_CMD_TOPIC = "rt/lowcmd";    // G1 底层命令主题
static const std::string HG_STATE_TOPIC = "rt/lowstate"; // G1 底层状态主题

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

const int G1_NUM_MOTOR = 29; // G1 电机数量（不包括双手的 7*2=14 个电机）

/**
 * 线程安全的数据缓冲区模板类
 * 用于在多线程环境下安全地读写数据
 */
template <typename T>
class DataBuffer {
 public:
  // 设置数据（写操作，使用独占锁）
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  // 获取数据（读操作，使用共享锁，允许多个读操作并发）
  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  // 清空数据
  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;      // 存储的数据
  std::shared_mutex mutex;       // 读写锁（支持多读单写）
};

/**
 * IMU 状态结构体
 * 存储机器人身体的姿态和运动信息
 */
struct ImuState {
  std::array<float, 3> rpy = {};    // 欧拉角：roll（横滚）、pitch（俯仰）、yaw（偏航）
  std::array<float, 3> omega = {};  // 角速度（rad/s）
  std::array<float, 4> quat = {};   // 四元数（w, x, y, z）
  std::array<float, 3> abody = {};  // 身体加速度（m/s²）
};

/**
 * 电机命令结构体
 * 存储发送给每个电机的控制指令
 */
struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};   // 目标关节位置（rad）
  std::array<float, G1_NUM_MOTOR> dq_target = {};  // 目标关节速度（rad/s）
  std::array<float, G1_NUM_MOTOR> kp = {};         // 位置增益（刚度）
  std::array<float, G1_NUM_MOTOR> kd = {};         // 速度增益（阻尼）
  std::array<float, G1_NUM_MOTOR> tau_ff = {};    // 前馈力矩（N·m）
};

/**
 * 电机状态结构体
 * 存储从机器人读取的电机当前状态
 */
struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};   // 当前关节位置（rad）
  std::array<float, G1_NUM_MOTOR> dq = {};  // 当前关节速度（rad/s）
};


/**
 * G1 所有关节的刚度（位置增益 Kp）
 * 索引顺序：左腿(6) + 右腿(6) + 腰部(3) + 左臂(7) + 右臂(7) = 29
 */
std::array<float, G1_NUM_MOTOR> Kp{
    150, 150, 150, 300, 40, 40,      // 左腿：髋部pitch/roll/yaw, 膝盖, 踝部pitch/roll
    150, 150, 150, 300, 40, 40,      // 右腿：髋部pitch/roll/yaw, 膝盖, 踝部pitch/roll
    300, 300, 300,                   // 腰部：yaw, roll, pitch
    150, 150, 150, 100,  10, 10, 5,  // 左臂：肩部pitch/roll/yaw, 肘部, 腕部roll/pitch/yaw
    150, 150, 150, 100,  10, 10, 5,  // 右臂：肩部pitch/roll/yaw, 肘部, 腕部roll/pitch/yaw
};

/**
 * G1 所有关节的阻尼（速度增益 Kd）
 * 索引顺序与 Kp 相同
 */
std::array<float, G1_NUM_MOTOR> Kd{
    2, 2, 2, 4, 2, 2,     // 左腿
    2, 2, 2, 4, 2, 2,     // 右腿
    5, 5, 5,              // 腰部
    4, 4, 4, 1, 0.5, 0.5, 0.5,  // 左臂
    4, 4, 4, 1, 0.5, 0.5, 0.5   // 右臂
};

/**
 * 控制模式枚举
 * PR: Position/Resistance（位置/阻力模式）
 * AB: Active Brake（主动制动模式）
 */
enum PRorAB { PR = 0, AB = 1 };

/**
 * G1 关节索引枚举
 * 定义每个关节在数组中的位置索引
 */
enum G1JointIndex {
  // 左腿（索引 0-5）
  LeftHipPitch = 0,    // 左髋关节俯仰
  LeftHipRoll = 1,     // 左髋关节横滚
  LeftHipYaw = 2,       // 左髋关节偏航
  LeftKnee = 3,         // 左膝关节
  LeftAnklePitch = 4,  // 左踝关节俯仰
  LeftAnkleB = 4,      // 左踝关节B（别名）
  LeftAnkleRoll = 5,   // 左踝关节横滚
  LeftAnkleA = 5,      // 左踝关节A（别名）
  
  // 右腿（索引 6-11）
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  
  // 腰部（索引 12-14，注意：G1 23dof/29dof 版本中腰部可能被锁定）
  WaistYaw = 12,        // 腰部偏航
  WaistRoll = 13,       // 腰部横滚（G1 23dof/29dof 中无效）
  WaistA = 13,          // 腰部A（别名，无效）
  WaistPitch = 14,      // 腰部俯仰（G1 23dof/29dof 中无效）
  WaistB = 14,          // 腰部B（别名，无效）
  
  // 左臂（索引 15-21）
  LeftShoulderPitch = 15,  // 左肩俯仰
  LeftShoulderRoll = 16,    // 左肩横滚
  LeftShoulderYaw = 17,     // 左肩偏航
  LeftElbow = 18,           // 左肘关节
  LeftWristRoll = 19,       // 左腕横滚
  LeftWristPitch = 20,      // 左腕俯仰（G1 23dof 中无效）
  LeftWristYaw = 21,        // 左腕偏航（G1 23dof 中无效）
  
  // 右臂（索引 22-28）
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,     // G1 23dof 中无效
  RightWristYaw = 28        // G1 23dof 中无效
};

/**
 * CRC32 校验和计算函数
 * 用于验证 DDS 消息的完整性，防止数据传输错误
 * 
 * @param ptr 数据指针
 * @param len 数据长度（以32位字为单位）
 * @return CRC32 校验值
 */
inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;           // CRC32 初始值
  const uint32_t dwPolynomial = 0x04c11db7;  // CRC32 多项式
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else
        CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

/**
 * G1 控制主类
 * 负责机器人状态的接收、控制命令的计算和发送
 */
class G1Control {
 private:
  // 时间相关
  double time_;              // 当前时间（秒）
  double control_dt_;        // 控制周期（0.005秒 = 5ms，对应 200Hz）
  double duration_;          // 初始化持续时间（3秒，用于平滑过渡到零位）
  
  // 控制模式
  PRorAB mode_;              // 控制模式：PR（位置/阻力）或 AB（主动制动）
  uint8_t mode_machine_;     // 机器人模式机状态
  std::vector<std::vector<double>> frames_data_;  // 帧数据（未使用）

  // 线程安全的数据缓冲区
  DataBuffer<MotorState> motor_state_buffer_;      // 电机状态缓冲区
  DataBuffer<MotorCommand> motor_command_buffer_;   // 电机命令缓冲区
  DataBuffer<ImuState> imu_state_buffer_;          // IMU 状态缓冲区

  // DDS 通信通道
  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;      // 底层命令发布者
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;  // 底层状态订阅者
  
  // 线程指针
  ThreadPtr command_writer_ptr_;    // 命令写入线程
  ThreadPtr control_thread_ptr_;     // 控制计算线程
  ThreadPtr joystick_thread_ptr_;    // 摇杆处理线程
  
  // LCM 通信
  lcm::LCM _simpleLCM;                           // LCM 实例
  std::thread _simple_LCM_thread;                // LCM 处理线程
  bool _firstRun;                                // 首次运行标志
  bool _firstCommandReceived;                    // 是否已收到第一个命令
  
  // LCM 消息数据
  state_estimator_lcmt body_state_simple = {0};      // 身体状态估计数据
  body_control_data_lcmt joint_state_simple = {0};   // 关节状态数据
  pd_tau_targets_lcmt joint_command_simple = {0};     // 关节命令数据（来自策略网络）
  arm_action_lcmt arm_action_simple = {0};            // 手臂动作数据
  rc_command_lcmt rc_command = {0};                  // 遥控器命令数据
  
  // 游戏手柄/遥控器
  Gamepad gamepad;                                                      // 游戏手柄处理对象
  unitree_go::msg::dds_::WirelessController_ joystick_msg;            // 摇杆消息
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;  // 摇杆订阅者
  std::mutex joystick_mutex;                                            // 摇杆数据互斥锁

 public:
  /**
   * 构造函数
   * @param networkInterface 网络接口名称（如 "eth0" 或 "eth1"）
   */
  G1Control(std::string networkInterface)
      : time_(0.0),
        control_dt_(0.005),      // 5ms 控制周期
        duration_(3.0),          // 3秒初始化时间
        mode_(PR),                // 默认位置/阻力模式
        mode_machine_(0) {
    // 初始化 DDS 通道工厂
    ChannelFactory::Instance()->Init(0, networkInterface);

    // 创建底层命令发布者（用于向机器人发送控制命令）
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // 创建底层状态订阅者（用于接收机器人状态反馈）
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
            HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&G1Control::LowStateHandler, this, std::placeholders::_1), 1);

    // 创建游戏手柄/遥控器订阅者
    joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    joystick_subscriber->InitChannel(std::bind(&G1Control::JoystickHandler, this, std::placeholders::_1), 1);

    // 创建周期性线程
    command_writer_ptr_ =
        CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 5000,
                                &G1Control::LowCommandWriter, this);  // 命令写入线程（200Hz）
    control_thread_ptr_ = CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, 5000, &G1Control::Control, this);  // 控制计算线程（200Hz）
    joystick_thread_ptr_ = CreateRecurrentThreadEx(
        "nn_ctrl", UT_CPU_ID_NONE, 4000, &G1Control::JoystickStep, this);  // 摇杆处理线程（250Hz）

    // 订阅 LCM 消息（接收策略网络的命令）
    _simpleLCM.subscribe("pd_plustau_targets", &G1Control::handleActionLCM, this);  // 关节命令
    _simpleLCM.subscribe("arm_action", &G1Control::handleArmLCM, this);             // 手臂动作
    _simple_LCM_thread = std::thread(&G1Control::_simpleLCMThread, this);           // LCM 处理线程
    _firstCommandReceived = false;

    // TODO: 设置标称姿态
  }

  /**
   * 摇杆消息处理回调函数
   * 当收到 DDS 摇杆消息时调用，更新摇杆状态
   */
  void JoystickHandler(const void *message)
    {
        std::lock_guard<std::mutex> lock(joystick_mutex);
        joystick_msg = *(unitree_go::msg::dds_::WirelessController_ *)message;
    }

  /**
   * LCM 消息处理线程
   * 持续处理接收到的 LCM 消息
   */
  void _simpleLCMThread(){
    while(true){
        _simpleLCM.handle();  // 处理所有待处理的 LCM 消息
    }
    }

  /**
   * 处理来自策略网络的关节命令 LCM 消息
   * @param rbuf LCM 接收缓冲区
   * @param chan 通道名称
   * @param msg 关节命令消息（包含目标位置和力矩）
   */
  void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg){
    (void) rbuf;
    (void) chan;
    joint_command_simple = *msg;  // 保存关节命令

    // 标记已收到第一个命令
    if (_firstCommandReceived == false){
      _firstCommandReceived = true;
      std::cout << "First command received" << std::endl;
    }
    
  }

  /**
   * 处理手臂动作 LCM 消息
   * @param rbuf LCM 接收缓冲区
   * @param chan 通道名称
   * @param msg 手臂动作消息
   */
  void handleArmLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const arm_action_lcmt * msg){
    (void) rbuf;
    (void) chan;
    arm_action_simple = *msg;  // 保存手臂动作命令
  }



  /**
   * 底层状态消息处理回调函数
   * 当收到机器人状态反馈时调用，提取电机和 IMU 状态
   * TODO: 未来可能通过 LCM 发送状态数据
   */
  void LowStateHandler(const void *message) {
    unitree_hg::msg::dds_::LowState_ low_state =
        *(const unitree_hg::msg::dds_::LowState_ *)message;

    // 验证 CRC32 校验和，确保数据完整性
    if (low_state.crc() !=
        Crc32Core((uint32_t *)&low_state,
                  (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) {
      std::cout << "low_state CRC Error" << std::endl;
      return;
    }

    // 提取电机状态（位置和速度）
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();    // 关节位置
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();   // 关节速度
    }
    motor_state_buffer_.SetData(ms_tmp);

    // 提取 IMU 状态（姿态和运动信息）
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();        // 角速度
    imu_tmp.rpy = low_state.imu_state().rpy();                // 欧拉角
    imu_tmp.quat = low_state.imu_state().quaternion();        // 四元数
    imu_tmp.abody = low_state.imu_state().accelerometer();    // 加速度
    imu_state_buffer_.SetData(imu_tmp);
    
    // 更新机器人模式机状态
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0)
        std::cout << "G1 type: " << unsigned(low_state.mode_machine())
                  << std::endl;
      mode_machine_ = low_state.mode_machine();
    }
  }

  /**
   * 底层命令写入函数（周期性线程调用）
   * 从命令缓冲区读取控制命令，通过 DDS 发送给机器人
   */
  void LowCommandWriter() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_;              // 设置控制模式
    dds_low_command.mode_machine() = mode_machine_; // 设置模式机状态

    // 从缓冲区获取电机命令
    const std::shared_ptr<const MotorCommand> mc =
        motor_command_buffer_.GetData();
    if (mc) {
      // 填充每个电机的控制命令
      for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:启用, 0:禁用
        dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);      // 前馈力矩
        dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);       // 目标位置
        dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);     // 目标速度
        dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);            // 位置增益
        dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);            // 速度增益
      }

      // 计算并设置 CRC32 校验和
      dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                        (sizeof(dds_low_command) >> 2) - 1);
      // 通过 DDS 发送命令
      lowcmd_publisher_->Write(dds_low_command);
    }
  }

  /**
   * 摇杆处理函数（周期性线程调用）
   * 更新游戏手柄状态，并通过 LCM 发布遥控器命令
   */
  void JoystickStep() {
    {
      // 线程安全地更新游戏手柄状态
      std::lock_guard<std::mutex> lock(joystick_mutex);
      gamepad.Update(joystick_msg);  // 更新手柄状态（解析摇杆和按钮）
    }

    // 填充遥控器命令消息
    rc_command.left_stick[0] = gamepad.lx;   // 左摇杆 X 轴
    rc_command.left_stick[1] = gamepad.ly;    // 左摇杆 Y 轴（+up 是 1, +right 是 0）
    rc_command.right_stick[0] = gamepad.rx; // 右摇杆 X 轴
    rc_command.right_stick[1] = gamepad.ry;  // 右摇杆 Y 轴
    rc_command.right_lower_right_switch = gamepad.R2.pressed;  // R2 按钮状态

    // 通过 LCM 发布遥控器命令（供策略网络使用）
    _simpleLCM.publish("rc_command", &rc_command);
  }

  /**
   * 主控制函数（周期性线程调用）
   * 根据策略网络的命令和机器人当前状态，计算电机控制命令
   */
  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

    // 初始化所有电机命令为 0
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      motor_command_tmp.tau_ff.at(i) = 0.0;
      motor_command_tmp.q_target.at(i) = 0.0;
      motor_command_tmp.dq_target.at(i) = 0.0;
      motor_command_tmp.kp.at(i) = 0.0;
      motor_command_tmp.kd.at(i) = 0.0;
    }

    if (ms) {
      time_ += control_dt_;  // 更新控制时间
      
      if (time_ < duration_) {
        // [阶段 1]：初始化阶段，平滑地将机器人移动到零位姿态
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          double ratio = std::clamp(time_ / duration_, 0.0, 1.0);  // 插值比例 [0, 1]

          double q_des = 0;  // 目标位置为 0（零位）
          motor_command_tmp.tau_ff.at(i) = 0.0;
          // 线性插值：从当前位置平滑过渡到零位
          motor_command_tmp.q_target.at(i) =
              (q_des - ms->q.at(i)) * ratio + ms->q.at(i);
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = Kp[i];  // 使用预设的刚度
          motor_command_tmp.kd.at(i) = Kd[i];   // 使用预设的阻尼
        }
      } else {
          // [阶段 2]：正常运行阶段，执行策略网络的命令
          
          // 发布身体状态估计数据（供策略网络使用）
          const std::shared_ptr<const ImuState> imu_tmp_ptr = imu_state_buffer_.GetData();
          if (imu_tmp_ptr) {
            for (int i = 0; i < 3; i++){
              body_state_simple.rpy[i] = imu_tmp_ptr->rpy.at(i);           // 欧拉角
              body_state_simple.omegaBody[i] = imu_tmp_ptr->omega.at(i);  // 角速度
              body_state_simple.aBody[i] = imu_tmp_ptr->abody.at(i);      // 加速度
            }
          }
          
          // 发布关节状态数据（供策略网络使用）
          const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
          if (ms){
            for (int i = 0; i < G1_NUM_MOTOR; ++i){
              joint_state_simple.q[i] = ms->q.at(i);   // 关节位置
              joint_state_simple.qd[i] = ms->dq.at(i);  // 关节速度
            }
          }
          _simpleLCM.publish("state_estimator_data", &body_state_simple);  // 发布身体状态
          _simpleLCM.publish("body_control_data", &joint_state_simple);    // 发布关节状态

          // 应用策略网络的关节命令（前15个关节：腿部和腰部）
          for (int i = 0; i < 15; ++i){
            motor_command_tmp.q_target.at(i) = joint_command_simple.q_des[i];  // 目标位置
            motor_command_tmp.dq_target.at(i) = 0.0;                            // 目标速度（设为0）
            motor_command_tmp.kp.at(i) = Kp[i];                                 // 使用预设刚度
            motor_command_tmp.kd.at(i) = Kd[i];                                  // 使用预设阻尼
          }
          
          // 应用手臂动作命令（后14个关节：左臂7个 + 右臂7个）
          for (int i = 0; i < 14; ++i){
            motor_command_tmp.q_target.at(i+15) = arm_action_simple.act[i];  // 目标位置
            motor_command_tmp.dq_target.at(i+15) = 0.0;                       // 目标速度
            motor_command_tmp.kp.at(i+15) = Kp[i+15];                          // 使用预设刚度
            motor_command_tmp.kd.at(i+15) = Kd[i+15];                          // 使用预设阻尼
          }
      }
      
      // 将计算好的命令存入缓冲区（供命令写入线程使用）
      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }
};

/**
 * 主函数
 * @param argc 参数个数
 * @param argv 参数数组，argv[1] 应为网络接口名称（如 "eth0" 或 "eth1"）
 */
int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: G1 Whole-body Control Deployment w/o hands" << std::endl;
    exit(0);
  }
  std::string networkInterface = argv[1];  // 获取网络接口名称
  G1Control custom(networkInterface);      // 创建 G1 控制对象（启动所有线程）
  while (true) usleep(20000);               // 主线程休眠（20ms），保持程序运行
  return 0;
}