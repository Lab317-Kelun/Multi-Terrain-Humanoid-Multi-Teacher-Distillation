/**
 * Dex-3 机械手控制程序
 * 功能：通过 LCM 接收手部动作命令，通过 DDS 控制左右两只机械手
 */

#include <chrono>
#include <thread>
#include <lcm/lcm-cpp.hpp>  // LCM 通信库
#include <unitree/idl/hg/HandState_.hpp>  // 手部状态消息
#include <unitree/idl/hg/HandCmd_.hpp>    // 手部命令消息
#include <unitree/robot/channel/channel_publisher.hpp>   // DDS 发布者
#include <unitree/robot/channel/channel_subscriber.hpp>  // DDS 订阅者
#include <iostream>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>
#include "hand_action_lcmt.hpp"

/**
 * 左右手各关节的力矩限制（单位：N·m 或 rad）
 * 索引顺序：拇指0, 拇指1, 拇指2, 中指0, 中指1, 食指0, 食指1
 */
const float maxTorqueLimits_left[7]=  {  1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0   };  // 左手最大限制
const float minTorqueLimits_left[7]=  { -1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75}; // 左手最小限制
const float maxTorqueLimits_right[7]= {  1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75}; // 右手最大限制
const float minTorqueLimits_right[7]= { -1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0    }; // 右手最小限制

#define MOTOR_MAX 7   // 每只手的电机数量
#define SENSOR_MAX 9  // 每只手的压力传感器数量
uint8_t hand_id = 0;  // 手部ID（未使用）

/**
 * RIS 模式结构体（位域定义）
 * 用于编码电机控制模式
 */
typedef struct {
    uint8_t id     : 4;  // 电机ID（4位）
    uint8_t status : 3;  // 状态（3位）
    uint8_t timeout: 1;  // 超时标志（1位）
} RIS_Mode_t;

// 左手 DDS 通道命名空间
std::string ldds_namespace = "rt/dex3/left";           // 左手命令通道
std::string lsub_namespace = "rt/dex3/left/state";     // 左手状态通道
unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> lhandcmd_publisher;      // 左手命令发布者
unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> lhandstate_subscriber; // 左手状态订阅者
unitree_hg::msg::dds_::HandCmd_ msg;    // 手部命令消息（复用）
unitree_hg::msg::dds_::HandState_ lstate;  // 左手状态

// 右手 DDS 通道命名空间
std::string rdds_namespace = "rt/dex3/right";          // 右手命令通道
std::string rsub_namespace = "rt/dex3/right/state";    // 右手状态通道
unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> rhandcmd_publisher;      // 右手命令发布者
unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> rhandstate_subscriber; // 右手状态订阅者
unitree_hg::msg::dds_::HandState_ rstate;  // 右手状态

std::mutex stateMutex;  // 状态互斥锁（未使用）


/**
 * 手部控制类
 * 负责控制左右两只 Dex-3 机械手
 */
class HandControl {
    private:
        lcm::LCM _simpleLCM;                    // LCM 实例
        std::thread _simple_LCM_thread;        // LCM 处理线程
        std::thread _simple_hand_thread;       // 手部控制线程
        hand_action_lcmt hand_action_simple = {0};  // 手部动作命令（来自策略网络）
        float q_left[7] = {0};                 // 左手目标关节位置
        float q_right[7] = {0};                // 右手目标关节位置
        float hand_action[14] = {0};           // 手部动作数组（未使用）
        
    public:
    /**
     * 构造函数
     * @param val 初始化参数（未使用）
     */
    HandControl(int  val){
        // 初始化 DDS 通道工厂
        unitree::robot::ChannelFactory::Instance()->Init(0);
        
        // 初始化左手 DDS 通道
        lhandcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(ldds_namespace + "/cmd"));
        lhandstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(lsub_namespace));
        lhandcmd_publisher->InitChannel();
        lstate.motor_state().resize(MOTOR_MAX);           // 调整电机状态数组大小
        lstate.press_sensor_state().resize(SENSOR_MAX);   // 调整压力传感器数组大小
        msg.motor_cmd().resize(MOTOR_MAX);                // 调整命令数组大小

        // 初始化右手 DDS 通道
        rhandcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(rdds_namespace + "/cmd"));
        rhandstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(rsub_namespace));
        rhandcmd_publisher->InitChannel();
        rstate.motor_state().resize(MOTOR_MAX);
        rstate.press_sensor_state().resize(SENSOR_MAX);
        
        // 订阅 LCM 消息（接收策略网络的手部动作命令）
        _simpleLCM.subscribe("hand_action", &HandControl::handleHandLCM, this);
        _simple_LCM_thread = std::thread(&HandControl::simpleLCMThread, this);      // 启动 LCM 处理线程
        _simple_hand_thread = std::thread(&HandControl::simplehandThread, this);    // 启动手部控制线程
        
        // 初始化手部动作命令（设置默认值：左手最小限制，右手最大限制）
        for (int item = 0; item < 7; item++){
            hand_action_simple.act[item] = minTorqueLimits_left[item];      // 左手：最小限制
            hand_action_simple.act[item+7] = maxTorqueLimits_right[item];  // 右手：最大限制
        }
        // 特殊处理：设置某些关节的初始值
        hand_action_simple.act[0] = 0.0;                      // 左手拇指0关节
        hand_action_simple.act[1] = maxTorqueLimits_left[1];   // 左手拇指1关节
        hand_action_simple.act[2] = maxTorqueLimits_left[2];   // 左手拇指2关节
        hand_action_simple.act[7] = 0.0;                      // 右手拇指0关节
        hand_action_simple.act[8] = minTorqueLimits_right[1]; // 右手拇指1关节
        hand_action_simple.act[9] = minTorqueLimits_right[2];  // 右手拇指2关节
    }
    /**
     * 控制手部电机旋转
     * @param isLeftHand true=左手, false=右手
     */
    void rotateMotors(bool isLeftHand) {
        // 根据左右手选择对应的力矩限制
        const float* maxTorqueLimits = isLeftHand ? maxTorqueLimits_left : maxTorqueLimits_right;
        const float* minTorqueLimits = isLeftHand ? minTorqueLimits_left : minTorqueLimits_right;

        // 根据左右手选择目标位置数组
        float* target = isLeftHand ? q_left : q_right;

        // 为每个电机设置控制命令
        for (int i = 0; i < MOTOR_MAX; i++) {
            // 构建 RIS 模式（电机控制模式编码）
            RIS_Mode_t ris_mode;
            ris_mode.id = i;        // 电机ID
            ris_mode.status = 0x01; // 状态：启用
            ris_mode.timeout = 0x01; // 超时标志
            
            // 将位域编码为单个字节
            uint8_t mode = 0;
            mode |= (ris_mode.id & 0x0F);            // 低4位：电机ID
            mode |= (ris_mode.status & 0x07) << 4;  // 中间3位：状态
            mode |= (ris_mode.timeout & 0x01) << 7;  // 最高位：超时标志
            
            // 设置电机命令
            msg.motor_cmd()[i].mode(mode);           // 控制模式
            msg.motor_cmd()[i].tau(0);               // 力矩（设为0，使用位置控制）
            msg.motor_cmd()[i].kp(0.5);              // 位置增益
            msg.motor_cmd()[i].kd(0.1);              // 速度增益
            msg.motor_cmd()[i].q(target[i]);         // 目标位置
        }

        // 根据左右手选择对应的发布者发送命令
        if (isLeftHand){
            lhandcmd_publisher->Write(msg);  // 发送左手命令
        }
        else{
            rhandcmd_publisher->Write(msg); // 发送右手命令
        }

        usleep(100);  // 休眠 100 微秒
    }

    /**
     * 处理手部动作 LCM 消息回调函数
     * @param rbuf LCM 接收缓冲区
     * @param chan 通道名称
     * @param msg 手部动作消息（包含14个关节的目标位置：左手7个 + 右手7个）
     */
    void handleHandLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const hand_action_lcmt * msg){
        (void) rbuf;
        (void) chan;
        hand_action_simple = *msg;  // 保存手部动作命令
    }

    /**
     * LCM 消息处理线程
     * 持续处理接收到的 LCM 消息
     */
    void simpleLCMThread(){
        while(true){
            _simpleLCM.handle();  // 处理所有待处理的 LCM 消息
        }
    }

    /**
     * 手部控制线程
     * 持续读取手部动作命令并发送给机械手
     */
    void simplehandThread(){
        while (true) {
            // 从 LCM 消息中提取左右手的目标位置
            for (int i = 0; i < 7; i++){
                q_left[i] = hand_action_simple.act[i];      // 左手：前7个关节
                q_right[i] = hand_action_simple.act[i+7];   // 右手：后7个关节
            }
            // 发送控制命令给左右手
            rotateMotors(true);   // 控制左手
            rotateMotors(false);  // 控制右手
        }
    }
};


/**
 * 主函数
 * 创建手部控制对象并保持程序运行
 */
int main() {
    HandControl cus(0);  // 创建手部控制对象（启动所有线程）
    while (true) usleep(20000);  // 主线程休眠（20ms），保持程序运行
    return 0;
}