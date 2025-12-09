#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <chrono> // Include chrono for timing
#include <deque> // For history buffer
#include <vector> // For storing default angles etc.
#include <iomanip> // For std::setprecision, std::setw
#include <sstream> // Added for string formatting
#include <map>

#include "gamepad.hpp"

// ROS
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h" // Include PointCloud2 message header
#include <pcl_conversions/pcl_conversions.h> // For converting ROS PointCloud2 to PCL
#include <pcl/point_cloud.h> // For PCL PointCloud
#include <pcl/point_types.h> // For PCL point types

// TF2
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> // For converting TF2 transforms to geometry_msgs
#include <tf2_eigen/tf2_eigen.h> // For TF2 <-> Eigen conversions

// Eigen
#include <Eigen/Dense>

// PCL specific includes for kNN
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <pcl/common/transforms.h> // For transforming clouds

// TorchScript / LibTorch
#include <torch/script.h>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

// Removed chatterCallback as it's unused

template <typename T>
class DataBuffer {
 public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};

const int G1_NUM_MOTOR = 29;
// Number of actions/DoFs the policy controls (matching Python)
const int POLICY_DOF_COUNT = 23;
// Observation vector size: 3(ang_vel) + 3(grav) + N(q) + N(dq) + N(action)
const int TORSO_OBS_SIZE = 6 + 3 * POLICY_DOF_COUNT; // 6 + 3*23 = 75
const int HISTORY_LENGTH = 5;

struct ImuState {
  // TODO check if this is correct
  std::array<float, 4> quat = {1.0, 0.0, 0.0, 0.0}; // w, x, y, z (initialize to identity)
  std::array<float, 3> omega = {};
};
struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};
struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
};

// Stiffness for all G1 Joints
// std::array<float, G1_NUM_MOTOR> Kp{
//     60, 60, 60, 100, 40, 40,      // legs
//     60, 60, 60, 100, 40, 40,      // legs
//     60, 40, 40,                   // waist
//     40, 40, 40, 40,  40, 40, 40,  // arms
//     40, 40, 40, 40,  40, 40, 40   // arms
// };

std::array<float, G1_NUM_MOTOR> Kp{
  75, 75, 75, 75, 20, 20,
  75, 75, 75, 75, 20, 20,
  75, 75, 75,
  75, 75, 75, 75, 20, 20, 20,
  75, 75, 75, 75, 20, 20, 20,

};

// Damping for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kd{
  2, 2, 2, 2, 0.2, 0.1,
  2, 2, 2, 2, 0.2, 0.1,
  2, 2, 2,
  2, 2, 2, 2, 1, 1, 1,
  2, 2, 2, 2, 1, 1, 1,
    // 1, 1, 1, 2, 1, 1,     // legs
    // 1, 1, 1, 2, 1, 1,     // legs
    // 1, 1, 1,              // waist
    // 1, 1, 1, 1, 1, 1, 1,  // arms
    // 1, 1, 1, 1, 1, 1, 1   // arms
};

// Initial angles for the 23 policy-controlled joints (knees bent)
static const std::vector<float> init_angles = {
    -0.312f,  0.000f,  0.000f,  0.669f, -0.363f,  0.000f, // left leg
    -0.312f,  0.000f,  0.000f,  0.669f, -0.363f,  0.000f, // right leg
     0.000f,  0.000f,  0.073f,                         // waist (y/r/p)
     0.200f,  0.200f,  0.000f,  0.600f,                 // left arm (shoulder p/r/y, elbow)
     0.200f, -0.200f,  0.000f,  0.600f                  // right arm (shoulder p/r/y, elbow)
}; // Total 23 DoF

// Default upper body targets (11 DoF) - derived from the last 11 elements of init_angles
// (Waist: 3 DoF, Left Arm: 4 DoF, Right Arm: 4 DoF)
static const std::vector<float> default_upper_body_targets = {
    init_angles[12], init_angles[13], init_angles[14], // waist (y/r/p)
    init_angles[15], init_angles[16], init_angles[17], init_angles[18], // left arm (shoulder p/r/y, elbow)
    init_angles[19], init_angles[20], init_angles[21], init_angles[22]  // right arm (shoulder p/r/y, elbow)
};

// State machine definition
enum class ControlState {
    INITIAL_HOLD,      // Start state, holding current position
    GOING_TO_DEFAULT,  // Moving to/holding default pose
    RUNNING_POLICY,    // Executing the policy
    SHUTDOWN           // Preparing to exit
};

// Default positions for specific non-policy joints (e.g., wrists)
// Used when the policy is active to command these joints to a default state.
static const std::map<int, float> non_policy_default_dof_pos = {
    {19, 0.0f}, // LeftWristRoll
    {20, 0.0f}, // LeftWristPitch (INVALID for 23dof)
    {21, 0.0f}, // LeftWristYaw (INVALID for 23dof)
    {26, 0.0f}, // RightWristRoll
    {27, 0.0f}, // RightWristPitch (INVALID for 23dof)
    {28, 0.0f}  // RightWristYaw (INVALID for 23dof)
};

// Add Mode enum back here
enum class Mode {
  PR = 0,  // Series Control for Ptich/Roll Joints
  AB = 1   // Parallel Control for A/B Joints
};

// Matches Python order for the first 23 joints
enum PolicyJointIndex {
  LeftHipPitch_P = 0,
  LeftHipRoll_P = 1,
  LeftHipYaw_P = 2,
  LeftKnee_P = 3,
  LeftAnklePitch_P = 4,
  LeftAnkleRoll_P = 5,
  RightHipPitch_P = 6,
  RightHipRoll_P = 7,
  RightHipYaw_P = 8,
  RightKnee_P = 9,
  RightAnklePitch_P = 10,
  RightAnkleRoll_P = 11,
  WaistYaw_P = 12,
  WaistRoll_P = 13,       // May be unused depending on robot variant
  WaistPitch_P = 14,      // May be unused depending on robot variant
  LeftShoulderPitch_P = 15,
  LeftShoulderRoll_P = 16,
  LeftShoulderYaw_P = 17,
  LeftElbow_P = 18,
  RightShoulderPitch_P = 19,
  RightShoulderRoll_P = 20,
  RightShoulderYaw_P = 21,
  RightElbow_P = 22
};

// Map Policy Joint Index to G1 Motor Index (Needs verification for specific robot)
// Assuming a direct mapping for the first 23 for now
const std::vector<int> policy_joint_to_motor_idx = {
    //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25
    // Indices 13 (WaistRoll) and 14 (WaistPitch) might map differently or not exist
    // on some G1 variants (like 23 DoF). Adjust as needed.
};

// Mapping G1 Motor Indices back to policy indices (or -1 if not controlled)
// Useful for setting Kp/Kd/Hold commands correctly
std::vector<int> motor_to_policy_joint_idx(G1_NUM_MOTOR, -1);
void initialize_motor_to_policy_map() {
    for(int policy_idx = 0; policy_idx < POLICY_DOF_COUNT; ++policy_idx) {
        int motor_idx = policy_joint_to_motor_idx[policy_idx];
        if (motor_idx >= 0 && motor_idx < G1_NUM_MOTOR) {
            motor_to_policy_joint_idx[motor_idx] = policy_idx;
        } else {
             ROS_ERROR("Invalid motor index %d in policy_joint_to_motor_idx for policy joint %d during map initialization", motor_idx, policy_idx);
        }
    }
}


enum G1JointIndex { // Keep original for reference/completeness
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,        // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13,           // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14,       // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14,           // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,   // NOTE INVALID for g1 23dof
  LeftWristYaw = 21,     // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,  // NOTE INVALID for g1 23dof
  RightWristYaw = 28     // NOTE INVALID for g1 23dof
};


inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
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

// --- Heightmap Configuration ---
#define GRID_X_SIDE_LENGTH 11 // Example size (match python config)
#define GRID_Y_SIDE_LENGTH 11 // Example size
#define GRID_X_RESOLUTION 0.1 // Example resolution
#define GRID_Y_RESOLUTION 0.1 // Example resolution
#define ROS_BASE_FRAME "odom_corrected" // Frame to transform points into
#define ROS_TORSO_FRAME "torso_link"    // Robot torso frame
#define KNN_K 3                  // Number of neighbors for interpolation
// #define KNN_MAX_DISTANCE 0.15    // Max distance threshold for 1st NN
#define KNN_MAX_DISTANCE 0.15    // Max distance threshold for 1st NN
#define KNN_DEFAULT_HEIGHT_OFFSET 0.85 // Offset from torso Z for default height
// --- End Heightmap Configuration ---

class VideomimicInferenceReal {
 private:
  double control_dt_;  // Control loop period (e.g., 0.02 for 50 Hz)
  uint8_t mode_machine_; // Stores the robot's machine mode from LowState
  int counter_; // General purpose counter, e.g., for logging

  Gamepad gamepad_;
  REMOTE_DATA_RX rx_; // Buffer for gamepad data

  // Data buffers for sharing state between threads
  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  // DDS Communication
  ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber_;

  // Threads
  ThreadPtr command_writer_ptr_, control_thread_ptr_;

  // Optional: Motion Switcher Client (used in example to disable other controllers)
  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;

  // ROS Members
  ros::NodeHandle* n_ptr_;
  ros::Subscriber elevation_cloud_sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // TorchScript Policy Model
  torch::jit::script::Module default_module_;
  torch::jit::script::Module heightmap_module_;

  // --- Heightmap Members ---
  Eigen::Matrix<float, Eigen::Dynamic, 2> local_query_points_; // Local grid coordinates
  Eigen::Matrix<float, GRID_Y_SIDE_LENGTH, GRID_X_SIDE_LENGTH> heights_; // Heightmap grid data
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_; // KD-Tree for point cloud search (used in callback)
  int grid_points_count_; // Total points in the heightmap grid

  // --- Members for Policy Observation Calculation ---
  std::vector<float> default_angles_; // Default joint angles for policy DOFs
  float dof_pos_scale_;
  float dof_vel_scale_;
  float ang_vel_scale_;
  float action_scale_; // Needed if actions map directly to target delta
  Eigen::VectorXf last_action_; // Store the last action output by the policy

  // History buffer for torso observations
  std::deque<Eigen::VectorXf> torso_obs_history_;
  bool history_initialized_; // Flag to indicate if history buffer is filled
  // History buffers for relative torso XY and Yaw
  std::deque<Eigen::Vector2f> torso_xy_history_;
  std::deque<float> torso_yaw_history_;
  bool xy_history_initialized_; // Flag for XY history
  bool yaw_history_initialized_; // Flag for Yaw history

  // --- State Machine ---
  ControlState current_control_state_;
  Eigen::VectorXf transition_start_q_;
  Eigen::VectorXf transition_target_q_;
  ros::Time transition_start_time_;
  ros::Duration transition_duration_;
  bool is_transitioning_;
  bool use_heightmap_policy_; // Flag to select policy
  bool B_was_pressed_; // For manual button toggle debounce

 public:
  VideomimicInferenceReal(std::string networkInterface, ros::NodeHandle* nh)
      : control_dt_(0.02), // Set control loop to 50 Hz
        mode_machine_(0),
        counter_(0),
        n_ptr_(nh),
        grid_points_count_(GRID_X_SIDE_LENGTH * GRID_Y_SIDE_LENGTH),
        kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()),
        // Initialize default angles from config
        default_angles_{ -0.1000f,  0.0000f,  0.0000f,  0.3000f, -0.2000f,  0.0000f, // left leg
                         -0.1000f,  0.0000f,  0.0000f,  0.3000f, -0.2000f,  0.0000f, // right leg
                          0.0000f,  0.0000f,  0.0000f,                           // waist
                          0.0000f,  0.0000f,  0.0000f,  0.0000f,                 // left arm
                          0.0000f,  0.0000f,  0.0000f,  0.0000f },               // right arm
        // TODO: Set correct scaling factors from python config.py
        dof_pos_scale_(1.0f),
        dof_vel_scale_(0.05f),
        ang_vel_scale_(0.25f),
        action_scale_(0.25f),
        last_action_(Eigen::VectorXf::Zero(POLICY_DOF_COUNT)),
        history_initialized_(false),
        current_control_state_(ControlState::INITIAL_HOLD), // Start in initial hold state
        transition_duration_(ros::Duration(2.0)), // Set transition time (e.g., 2 seconds)
        is_transitioning_(false),
        use_heightmap_policy_(false), // Start with default policy
        B_was_pressed_(false) // Initialize debounce flag
         {
    // Initialize transition members
    transition_start_q_ = Eigen::VectorXf::Zero(G1_NUM_MOTOR);
    transition_target_q_ = Eigen::VectorXf::Zero(G1_NUM_MOTOR);

    initialize_motor_to_policy_map(); // Build the reverse map

    ChannelFactory::Instance()->Init(0, networkInterface);

    ROS_WARN("Using PLACEHOLDER scaling factors. Update in constructor!");

    // Initialize TF listener and buffer *before* using them
    if (n_ptr_) {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        ROS_INFO("TF Listener initialized.");
    } else {
        ROS_ERROR("Cannot initialize TF Listener: Invalid NodeHandle.");
    }

    // try to shutdown motion control-related service (Optional, from example)
    msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
    msc_->SetTimeout(5.0f);
    msc_->Init();
    std::string form, name;
    while (msc_->CheckMode(form, name), !name.empty()) {
      ROS_INFO("Waiting for MotionSwitcherClient mode release...");
      if (msc_->ReleaseMode())
        ROS_ERROR("Failed to switch to Release Mode");
      sleep(1); // Use sleep(1) instead of sleep(5)
    }
    ROS_INFO("MotionSwitcherClient mode released.");


    // --- Load the TorchScript model --- (Now loading two models)
    std::string default_model_path = "/home/unitree/Desktop/20250414_170842_g1_deepmimic_dict.pt";
    std::string heightmap_model_path = "/home/unitree/Desktop/20250502_124756_g1_deepmimic_dict.pt";

    try {
        default_module_ = torch::jit::load(default_model_path);
        ROS_INFO("Successfully loaded DEFAULT TorchScript model from %s", default_model_path.c_str());
        default_module_.eval(); // Set to evaluation mode
    }
    catch (const c10::Error& e) {
        ROS_ERROR("Error loading the DEFAULT TorchScript model: %s", e.what());
        throw; // Re-throw exception to signal failure
    }

    try {
        heightmap_module_ = torch::jit::load(heightmap_model_path);
        ROS_INFO("Successfully loaded HEIGHTMAP TorchScript model from %s", heightmap_model_path.c_str());
        heightmap_module_.eval(); // Set to evaluation mode
    }
    catch (const c10::Error& e) {
        ROS_ERROR("Error loading the HEIGHTMAP TorchScript model: %s", e.what());
        throw; // Re-throw exception to signal failure
    }


    // create DDS publisher/subscriber
    lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();
    lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(std::bind(&VideomimicInferenceReal::LowStateHandler, this, std::placeholders::_1), 1);

    // --- Initialize ROS Subscriber ---
    if (n_ptr_) {
        elevation_cloud_sub_ = n_ptr_->subscribe<sensor_msgs::PointCloud2>(
            "/elevation_map_fused_visualization/elevation_cloud", 10, // TODO: Make topic name configurable
            &VideomimicInferenceReal::ElevationCloudCallback, this);
        ROS_INFO("Subscribed to /elevation_map_fused_visualization/elevation_cloud topic.");
    } else {
        ROS_ERROR("Invalid NodeHandle passed to VideomimicInferenceReal constructor.");
    }
    // --- End ROS Subscriber Init ---

    // --- Initialize Heightmap ---
    createElevationGrid();
    heights_.setConstant(KNN_DEFAULT_HEIGHT_OFFSET); // Initialize with offset relative to torso
    ROS_INFO("Heightmap grid created (%d x %d).", GRID_Y_SIDE_LENGTH, GRID_X_SIDE_LENGTH);

    // Create threads for control and command sending (50 Hz = 20000 microseconds)
    unsigned long interval_us_control = static_cast<unsigned long>(20000);
    unsigned long interval_us_command = static_cast<unsigned long>(2000);

    command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, interval_us_command, &VideomimicInferenceReal::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, interval_us_control, &VideomimicInferenceReal::Control, this);
    ROS_INFO("Control thread started at %lu us interval.", interval_us_control);
    ROS_INFO("Command Writer thread started at %lu us interval.", interval_us_command);
  }

  // --- Helper: Set motor command to hold current positions ---
  void SetHoldCommand(MotorCommand& cmd, const MotorState& state) {
      for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) {
          cmd.q_target.at(motor_idx) = state.q.at(motor_idx);
          cmd.dq_target.at(motor_idx) = 0.0f;
          cmd.kp.at(motor_idx) = Kp[motor_idx];
          cmd.kd.at(motor_idx) = Kd[motor_idx];
          cmd.tau_ff.at(motor_idx) = 0.0f;
      }
  }

  // --- Helper: Set motor command to go to the initial pose (init_angles) ---
  void SetInitPoseCommand(MotorCommand& cmd) {
       // Set policy joints to default angles
       for (int policy_idx = 0; policy_idx < POLICY_DOF_COUNT; ++policy_idx) {
           int motor_idx = policy_joint_to_motor_idx[policy_idx];
           if (motor_idx >= 0 && motor_idx < G1_NUM_MOTOR) {
                cmd.q_target.at(motor_idx) = init_angles[policy_idx];
                cmd.dq_target.at(motor_idx) = 0.0f;
                cmd.kp.at(motor_idx) = Kp[motor_idx];
                cmd.kd.at(motor_idx) = Kd[motor_idx];
                cmd.tau_ff.at(motor_idx) = 0.0f;
           }
       }
       // Set non-policy joints to their defaults (or hold if no default specified)
       // NOTE: This part relies on having motor state `ms` available if holding is needed.
       // Consider passing `ms` if some non-policy joints should hold instead of using defaults.
       // For now, assuming all non-policy joints have defaults or we want them at 0.
       for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) {
            if (motor_to_policy_joint_idx[motor_idx] == -1) { // If not controlled by policy
                auto it = non_policy_default_dof_pos.find(motor_idx);
                if (it != non_policy_default_dof_pos.end()) {
                    cmd.q_target.at(motor_idx) = it->second; // Use default
                } else {
                    // Decide what to do if no default exists? Hold current? Go to 0?
                    // Setting to 0 for now, but might need adjustment.
                    // If holding current is desired, this function needs access to `ms`.
                    cmd.q_target.at(motor_idx) = 0.0f; // Default to 0 if not specified
                }
                cmd.dq_target.at(motor_idx) = 0.0f;
                cmd.kp.at(motor_idx) = Kp[motor_idx];
                cmd.kd.at(motor_idx) = Kd[motor_idx];
                cmd.tau_ff.at(motor_idx) = 0.0f;
            }
        }
  }

  // --- Helper: Get Gravity Vector from Quaternion ---
  Eigen::Vector3f getGravityOrientation(const Eigen::Quaternionf& q) {
      // Directly replicate the Python calculation component-wise
      // Python assumes quaternion order [w, x, y, z]
      // Eigen::Quaternionf stores as (x, y, z, w) internally but accessors are correct (q.w(), q.x(), etc.)
      float qw = q.w();
      float qx = q.x();
      float qy = q.y();
      float qz = q.z();

      Eigen::Vector3f gravity_orientation;
      gravity_orientation[0] = 2.0f * (-qz * qx + qw * qy); // Matches Python X
      gravity_orientation[1] = -2.0f * (qz * qy + qw * qx); // Matches Python Y
      gravity_orientation[2] = 1.0f - 2.0f * (qw * qw + qz * qz); // Matches Python Z

      return gravity_orientation;
  }

  // --- DDS LowState Handler ---
  // Receives robot state, updates buffers and gamepad
  void LowStateHandler(const void *message) {
    LowState_ low_state = *(const LowState_ *)message;
    // CRC check (important!)
    if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) {
      ROS_ERROR("CRC Error in LowState");
      return;
    }

    // Update motor state buffer
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    }
    motor_state_buffer_.SetData(ms_tmp);

    // Update IMU state buffer (using Base/Pelvis IMU from LowState)
    ImuState imu_tmp;
    imu_tmp.quat[0] = low_state.imu_state().quaternion()[0]; // w
    imu_tmp.quat[1] = low_state.imu_state().quaternion()[1]; // x
    imu_tmp.quat[2] = low_state.imu_state().quaternion()[2]; // y
    imu_tmp.quat[3] = low_state.imu_state().quaternion()[3]; // z
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_state_buffer_.SetData(imu_tmp);

    // Update gamepad state
    memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
    gamepad_.update(rx_.RF_RX);

    // Update mode machine state
    mode_machine_ = low_state.mode_machine();

    // Optional: Print status periodically
    if (++counter_ % 25 == 0) { // Print more frequently (e.g., every 25 * 2ms = 50ms)
        // ROS_INFO_THROTTLE(1.0,"LowStateHandler updated. A button: %d", static_cast<int>(gamepad_.A.pressed));
    }
  }

   // --- Main Control Loop (Runs at control_dt_ interval) ---
  void Control() {
      // --- Timing Start --- 
      auto control_loop_start_time = std::chrono::high_resolution_clock::now();

      // Get latest state data from buffers
      const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
      const std::shared_ptr<const ImuState> imu_state = imu_state_buffer_.GetData();

      // Wait until valid state is received
      if (!ms || !imu_state) {
          // ROS_WARN_THROTTLE(1.0, "Waiting for initial state data in Control loop...");
          // Send a zero command or hold command while waiting?
          // For now, do nothing until state is valid. LowCommandWriter will send nothing.
          return;
      }

      // --- 1. Calculate Observation Vector ---
      Eigen::VectorXf qj(POLICY_DOF_COUNT);
      Eigen::VectorXf dqj(POLICY_DOF_COUNT);
      for (int i = 0; i < POLICY_DOF_COUNT; ++i) {
          int motor_idx = policy_joint_to_motor_idx[i];
          if (motor_idx >= 0 && motor_idx < G1_NUM_MOTOR) {
              qj(i) = ms->q.at(motor_idx);
              dqj(i) = ms->dq.at(motor_idx);
          } else {
              // Should not happen if map is correct
              qj(i) = 0.0f; dqj(i) = 0.0f;
          }
      }

      Eigen::Quaternionf current_quat_base(imu_state->quat[0], imu_state->quat[1], imu_state->quat[2], imu_state->quat[3]);
      Eigen::Vector3f current_ang_vel_base(imu_state->omega[0], imu_state->omega[1], imu_state->omega[2]);

      // Calculate derived observations
      Eigen::Vector3f gravity_orientation = getGravityOrientation(current_quat_base);
      Eigen::VectorXf qj_obs = Eigen::VectorXf::Zero(POLICY_DOF_COUNT);
      for(int i=0; i<POLICY_DOF_COUNT; ++i) {
          qj_obs(i) = (qj(i) - default_angles_[i]) * dof_pos_scale_;
      }
      Eigen::VectorXf dqj_obs = dqj * dof_vel_scale_;
      Eigen::Vector3f ang_vel_obs = current_ang_vel_base * ang_vel_scale_;

      // Assemble the single observation vector for this timestep
      Eigen::VectorXf current_torso_obs(TORSO_OBS_SIZE);
      current_torso_obs.segment<3>(0) = ang_vel_obs;
      current_torso_obs.segment<3>(3) = gravity_orientation;
      current_torso_obs.segment(6, POLICY_DOF_COUNT) = qj_obs;
      current_torso_obs.segment(6 + POLICY_DOF_COUNT, POLICY_DOF_COUNT) = dqj_obs;
      current_torso_obs.segment(6 + 2 * POLICY_DOF_COUNT, POLICY_DOF_COUNT) = last_action_; // Use action from *previous* step

      // --- 2. Update Observation History ---
      torso_obs_history_.push_front(current_torso_obs);
      while (torso_obs_history_.size() > HISTORY_LENGTH) {
          torso_obs_history_.pop_back();
      }
      // Initialize history buffer on first run
      if (!history_initialized_) {
          while (torso_obs_history_.size() < HISTORY_LENGTH) {
              torso_obs_history_.push_back(current_torso_obs); // Fill with first valid obs
          }
          if (torso_obs_history_.size() == HISTORY_LENGTH) {
              history_initialized_ = true;
              ROS_INFO("Torso observation history buffer initialized.");
          }
      }

      // --- Calculate and Update Relative Torso XY and Yaw History (PLACEHOLDERS) ---
      // TODO: Implement the actual calculation logic for these values.
      // Eigen::Vector2f current_torso_xy_rel = Eigen::Vector2f::Zero(); // Placeholder
      // float current_torso_yaw_rel = 0.0f; // Placeholder

      // Calculate relative torso command velocities from gamepad
      Eigen::Vector2f current_torso_xy_rel;
      current_torso_xy_rel[0] = 0.5f * gamepad_.ly;
      current_torso_xy_rel[1] = 0.5f * gamepad_.lx * -1.0f;

      float current_torso_yaw_rel = 0.3f * gamepad_.rx * -1.0f;


      torso_xy_history_.push_front(current_torso_xy_rel);
      while (torso_xy_history_.size() > HISTORY_LENGTH) {
          torso_xy_history_.pop_back();
      }
      if (!xy_history_initialized_ && torso_xy_history_.size() == HISTORY_LENGTH) {
          while (torso_xy_history_.size() < HISTORY_LENGTH) {
             torso_xy_history_.push_back(current_torso_xy_rel); // Fill with first valid obs
          }
          if (torso_xy_history_.size() == HISTORY_LENGTH) {
            xy_history_initialized_ = true;
            ROS_INFO("Torso XY relative history buffer initialized.");
          }
      }

      torso_yaw_history_.push_front(current_torso_yaw_rel);
      while (torso_yaw_history_.size() > HISTORY_LENGTH) {
          torso_yaw_history_.pop_back();
      }
       if (!yaw_history_initialized_) {
           while (torso_yaw_history_.size() < HISTORY_LENGTH) {
               torso_yaw_history_.push_back(current_torso_yaw_rel); // Fill with first valid obs
           }
          if (torso_yaw_history_.size() == HISTORY_LENGTH) {
            yaw_history_initialized_ = true;
            ROS_INFO("Torso Yaw relative history buffer initialized.");
          }
      }
      // --- End History Update ---

      // Prepare command buffer
      MotorCommand motor_command_tmp;

      // Calculate history readiness before the switch for potential use in multiple states/logging
      bool all_history_ready = history_initialized_ && xy_history_initialized_ && yaw_history_initialized_;

      // --- 3. State Machine Logic ---
      switch (current_control_state_) {

          case ControlState::INITIAL_HOLD:
              ROS_INFO_THROTTLE(1.0, "State: INITIAL_HOLD. Press START to go to default pose.");
              SetHoldCommand(motor_command_tmp, *ms);
              if (gamepad_.start.pressed) {
                  ROS_INFO("START pressed. Transitioning to GOING_TO_DEFAULT.");
                  current_control_state_ = ControlState::GOING_TO_DEFAULT;
              }

              // Log timing for this state
              {
                  auto loop_end_time = std::chrono::high_resolution_clock::now();
                  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - control_loop_start_time);
                  ROS_INFO_THROTTLE(1.0, "Control Timing (ms): Total: %ld (State: %d)", 
                                  total_duration.count(), static_cast<int>(current_control_state_));
              }
              break;

          case ControlState::GOING_TO_DEFAULT:
              ROS_INFO_THROTTLE(1.0, "State: GOING_TO_DEFAULT. Press A to run policy, SELECT to shutdown.");
              // SetInitPoseCommand(motor_command_tmp); // Remove direct call

              // --- Smooth Transition Logic ---
              if (!is_transitioning_) { // First time entering or previous transition finished
                  ROS_INFO("Starting transition to default pose over %.1f seconds.", transition_duration_.toSec());
                  is_transitioning_ = true;
                  transition_start_time_ = ros::Time::now();
                  // Store starting positions
                  for(int i=0; i<G1_NUM_MOTOR; ++i) transition_start_q_(i) = ms->q.at(i);

                  // Calculate and store target default positions
                  for (int policy_idx = 0; policy_idx < POLICY_DOF_COUNT; ++policy_idx) {
                       int motor_idx = policy_joint_to_motor_idx[policy_idx];
                       if (motor_idx >= 0 && motor_idx < G1_NUM_MOTOR) {
                           // Use init_angles instead of default_angles_ for the target pose
                           transition_target_q_(motor_idx) = init_angles[policy_idx];
                       }
                   }
                   for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) {
                        if (motor_to_policy_joint_idx[motor_idx] == -1) { // Non-policy joint
                            auto it = non_policy_default_dof_pos.find(motor_idx);
                            if (it != non_policy_default_dof_pos.end()) {
                                // Use default position if specified for this non-policy joint
                                transition_target_q_(motor_idx) = it->second;
                            } else {
                                // Otherwise, hold current measured position
                                transition_target_q_(motor_idx) = ms->q.at(motor_idx);
                            }
                        }
                    }
              }

              if (is_transitioning_) {
                  ros::Time now = ros::Time::now();
                  ros::Duration elapsed_time = now - transition_start_time_;
                  float alpha = std::min(1.0f, static_cast<float>(elapsed_time.toSec() / transition_duration_.toSec()));

                  // Interpolate and set command
                  for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) {
                      motor_command_tmp.q_target.at(motor_idx) = transition_start_q_(motor_idx) + alpha * (transition_target_q_(motor_idx) - transition_start_q_(motor_idx));
                      motor_command_tmp.dq_target.at(motor_idx) = 0.0f;
                      motor_command_tmp.kp.at(motor_idx) = Kp[motor_idx];
                      motor_command_tmp.kd.at(motor_idx) = Kd[motor_idx];
                      motor_command_tmp.tau_ff.at(motor_idx) = 0.0f;
                  }

                  if (alpha >= 1.0f) {
                      ROS_INFO("Transition to default pose complete.");
                      is_transitioning_ = false; // Transition finished
                  }
              } else { // Transition finished, hold the final default pose
                  for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) {
                      motor_command_tmp.q_target.at(motor_idx) = transition_target_q_(motor_idx);
                      motor_command_tmp.dq_target.at(motor_idx) = 0.0f;
                      motor_command_tmp.kp.at(motor_idx) = Kp[motor_idx];
                      motor_command_tmp.kd.at(motor_idx) = Kd[motor_idx];
                      motor_command_tmp.tau_ff.at(motor_idx) = 0.0f;
                  }
              }
              // --- End Smooth Transition Logic ---

              // if (gamepad_.A.on_press) {
              if (gamepad_.A.pressed) {
                  ROS_INFO("A pressed. Transitioning to RUNNING_POLICY.");
                  is_transitioning_ = false; // Ensure transition flag is reset
                  // Reset history flags to force initialization when policy starts
                  history_initialized_ = false;
                  xy_history_initialized_ = false;
                  yaw_history_initialized_ = false;
                  torso_obs_history_.clear();
                  torso_xy_history_.clear();
                  torso_yaw_history_.clear();
                  last_action_.setZero(); // Reset last action
                  current_control_state_ = ControlState::RUNNING_POLICY;
              // } else if (gamepad_.select.on_press) {
              } else if (gamepad_.select.pressed) {
                  ROS_INFO("SELECT pressed. Transitioning to SHUTDOWN.");
                  is_transitioning_ = false; // Ensure transition flag is reset
                  current_control_state_ = ControlState::SHUTDOWN;
              }

              // Log timing for this state
              {
                  auto loop_end_time = std::chrono::high_resolution_clock::now();
                  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - control_loop_start_time);
                  ROS_INFO_THROTTLE(1.0, "Control Timing (ms): Total: %ld (State: %d)", 
                                  total_duration.count(), static_cast<int>(current_control_state_));
              }
              break;

          case ControlState::RUNNING_POLICY:
          {
              // --- Run Policy Inference --- (Only if history is ready)
              bool all_history_ready = history_initialized_ && xy_history_initialized_ && yaw_history_initialized_;

              // --- Check for Policy Toggle (B Button press - Manual Rising Edge Detection) ---
              // if (gamepad_.B.on_press) { // Use on_press for toggle action
              //     use_heightmap_policy_ = !use_heightmap_policy_;
              //     ROS_INFO("Policy toggled. Now using: %s Policy", use_heightmap_policy_ ? "Heightmap" : "Default");
              // }
              bool B_is_pressed = gamepad_.B.pressed;
              if (B_is_pressed && !B_was_pressed_) { // Rising edge detected
                  use_heightmap_policy_ = !use_heightmap_policy_;
                  ROS_INFO("Policy toggled. Now using: %s Policy", use_heightmap_policy_ ? "Heightmap" : "Default");
              }
              B_was_pressed_ = B_is_pressed; // Update previous state for next cycle

              if (all_history_ready) {
                 // Log which policy is active
                 ROS_INFO_THROTTLE(1.0, "State: RUNNING_POLICY (%s Policy, History Ready). Press SELECT to shutdown.",
                                   use_heightmap_policy_ ? "Heightmap" : "Default");

                 // --- Select active model ---
                 torch::jit::script::Module& active_module = use_heightmap_policy_ ? heightmap_module_ : default_module_;

                 // Declare timing points here
                 std::chrono::high_resolution_clock::time_point inference_start_time, inference_end_time;

                 try {
                      // Prepare history_torso_real tensor (flattened)
                      std::vector<float> history_flat_vec;
                      history_flat_vec.reserve(HISTORY_LENGTH * TORSO_OBS_SIZE);
                      for (const auto& obs_vec : torso_obs_history_) { // Deque front is newest
                           history_flat_vec.insert(history_flat_vec.end(), obs_vec.data(), obs_vec.data() + obs_vec.size());
                      }
                      // Reverse the vector so oldest observation is first, matching Python's roll axis=0?
                      // std::reverse(history_flat_vec.begin(), history_flat_vec.end()); // Needs testing if policy expects oldest first

                      if (history_flat_vec.size() != HISTORY_LENGTH * TORSO_OBS_SIZE) {
                           ROS_ERROR("History vector size mismatch! Expected %d, got %zu. Holding.",
                                     HISTORY_LENGTH * TORSO_OBS_SIZE, history_flat_vec.size());
                           // Fallback to holding current position
                           for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) motor_command_tmp.q_target.at(motor_idx) = ms->q.at(motor_idx);
                      } else {
                          torch::Tensor history_torso_real = torch::from_blob(history_flat_vec.data(), {1, HISTORY_LENGTH * TORSO_OBS_SIZE}, torch::kFloat32).clone();

                          // Prepare other inputs (Placeholders - TODO: Implement real calculations)
                          // Single step relative values
                          torch::Tensor torso_xy_rel_tensor = torch::from_blob(current_torso_xy_rel.data(), {1, 2}, torch::kFloat32).clone();
                          torch::Tensor torso_yaw_rel_tensor = torch::from_blob(&current_torso_yaw_rel, {1, 1}, torch::kFloat32).clone();

                          // Prepare history_torso_xy_rel tensor
                          std::vector<float> history_xy_flat_vec;
                          history_xy_flat_vec.reserve(HISTORY_LENGTH * 2);
                          for (const auto& xy_vec : torso_xy_history_) { // Deque front is newest
                              history_xy_flat_vec.push_back(xy_vec[0]);
                              history_xy_flat_vec.push_back(xy_vec[1]);
                          }
                          torch::Tensor history_torso_xy_rel_tensor = torch::from_blob(history_xy_flat_vec.data(), {1, HISTORY_LENGTH * 2}, torch::kFloat32).clone();

                           // Prepare history_torso_yaw_rel tensor
                          std::vector<float> history_yaw_flat_vec;
                          history_yaw_flat_vec.reserve(HISTORY_LENGTH);
                          for (const float& yaw_val : torso_yaw_history_) { // Deque front is newest
                              history_yaw_flat_vec.push_back(yaw_val);
                          }
                          torch::Tensor history_torso_yaw_rel_tensor = torch::from_blob(history_yaw_flat_vec.data(), {1, HISTORY_LENGTH}, torch::kFloat32).clone();


                          // Prepare heightmap tensor
                          torch::Tensor terrain_height_noisy = torch::from_blob(heights_.data(), {1, GRID_Y_SIDE_LENGTH, GRID_X_SIDE_LENGTH}, torch::kFloat32).clone().transpose(1, 2).contiguous();

                          // Prepare upper body joint targets tensor
                          // Create a non-const copy because from_blob needs non-const void*
                          std::vector<float> upper_body_targets_copy = default_upper_body_targets;
                          torch::Tensor upper_body_joint_targets_tensor = torch::from_blob(
                              upper_body_targets_copy.data(), // Use data from the copy
                              {1, static_cast<long int>(upper_body_targets_copy.size())}, // Cast size
                              torch::kFloat32).clone();

                          // Create input dictionary (ensure keys match the trained model!)
                          c10::Dict<std::string, torch::Tensor> inputs;
                          inputs.insert("history_torso_real", history_torso_real);
                          // inputs.insert("torso_xy_rel", torso_xy_rel); // OLD single step -> Re-adding below
                          // inputs.insert("torso_yaw_rel", torso_yaw_rel); // OLD single step -> Re-adding below
                          inputs.insert("torso_xy_rel", torso_xy_rel_tensor);          // Current step rel XY
                          inputs.insert("torso_yaw_rel", torso_yaw_rel_tensor);         // Current step rel Yaw
                          inputs.insert("history_torso_xy_rel", history_torso_xy_rel_tensor); // NEW history
                          inputs.insert("history_torso_yaw_rel", history_torso_yaw_rel_tensor); // NEW history
                          inputs.insert("terrain_height_noisy", terrain_height_noisy);
                          inputs.insert("upper_body_joint_targets", upper_body_joint_targets_tensor); // NEW input
                          // Add other necessary inputs here...

                          // Inference
                          inference_start_time = std::chrono::high_resolution_clock::now(); // Assign time
                          torch::jit::IValue output_ivalue = active_module.forward({inputs});
                          inference_end_time = std::chrono::high_resolution_clock::now(); // Assign time
                          torch::Tensor action_tensor = output_ivalue.toTensor().to(torch::kCPU).squeeze();

                          if (action_tensor.dim() != 1 || action_tensor.size(0) != POLICY_DOF_COUNT) {
                               ROS_ERROR("Policy output tensor has incorrect dimensions! Expected [%d], got shape %s. Holding.",
                                         POLICY_DOF_COUNT, c10::str(action_tensor.sizes()).c_str());
                               for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) motor_command_tmp.q_target.at(motor_idx) = ms->q.at(motor_idx);
                          } else {
                               // Store action for next step's observation
                               memcpy(last_action_.data(), action_tensor.data_ptr<float>(), POLICY_DOF_COUNT * sizeof(float));
                               // TODO: Add action clipping if needed: last_action_ = last_action_.cwiseMax(-clip).cwiseMin(clip);

                               // Calculate target DoF positions
                               Eigen::VectorXf target_dof_pos(POLICY_DOF_COUNT);
                               Eigen::Map<Eigen::VectorXf> default_angles_map(default_angles_.data(), POLICY_DOF_COUNT);
                               target_dof_pos = default_angles_map + last_action_ * action_scale_;

                               // Set commands for policy-controlled DOFs
                               for (int policy_idx = 0; policy_idx < POLICY_DOF_COUNT; ++policy_idx) {
                                   int motor_idx = policy_joint_to_motor_idx[policy_idx];
                                   if (motor_idx >= 0 && motor_idx < G1_NUM_MOTOR) {
                                        motor_command_tmp.q_target.at(motor_idx) = target_dof_pos(policy_idx);
                                        motor_command_tmp.dq_target.at(motor_idx) = 0.0f; // Target velocity 0 for PD control
                                        motor_command_tmp.kp.at(motor_idx) = Kp[motor_idx];
                                        motor_command_tmp.kd.at(motor_idx) = Kd[motor_idx];
                                        motor_command_tmp.tau_ff.at(motor_idx) = 0.0f;
                                   }
                               }
                               // Set commands for non-policy DOFs (Hold current position or use default)
                                for (int motor_idx = 0; motor_idx < G1_NUM_MOTOR; ++motor_idx) {
                                    if (motor_to_policy_joint_idx[motor_idx] == -1) { // If not controlled by policy
                                        auto it = non_policy_default_dof_pos.find(motor_idx);
                                        if (it != non_policy_default_dof_pos.end()) {
                                            // Use default position if specified for this non-policy joint
                                            motor_command_tmp.q_target.at(motor_idx) = it->second;
                                        } else {
                                            // Otherwise, hold current measured position
                                            motor_command_tmp.q_target.at(motor_idx) = ms->q.at(motor_idx);
                                        }
                                        motor_command_tmp.dq_target.at(motor_idx) = 0.0f;
                                        motor_command_tmp.kp.at(motor_idx) = Kp[motor_idx]; // Use default Kp/Kd
                                        motor_command_tmp.kd.at(motor_idx) = Kd[motor_idx];
                                        motor_command_tmp.tau_ff.at(motor_idx) = 0.0f;
                                    }
                                }
                               ROS_INFO_THROTTLE(0.5, "Policy Active (A pressed)");
                          }
                      }
                  } catch (const c10::Error& e) {
                      ROS_ERROR("TorchScript inference error: %s. Holding default pose.", e.what());
                      SetInitPoseCommand(motor_command_tmp);
                  } catch (const std::exception& e) {
                       ROS_ERROR("Standard exception during inference: %s. Holding default pose.", e.what());
                       SetInitPoseCommand(motor_command_tmp);
                  }

                  // --- Detailed Timing Log --- (Only after attempting inference)
                  auto loop_end_time = std::chrono::high_resolution_clock::now();
                  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - control_loop_start_time);
                  // Check if inference times were actually set (i.e., no exception before assignment)
                  if (inference_start_time.time_since_epoch().count() != 0) { 
                      auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end_time - inference_start_time); 
                      auto precise_pre_inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_start_time - control_loop_start_time);
                      ROS_INFO_THROTTLE(1.0, "Control Timing (ms): Pre-Inf: %ld | Inf: %ld | Total: %ld", 
                                        precise_pre_inference_duration.count(), 
                                        inference_duration.count(), 
                                        total_duration.count());
                  } else {
                       // Inference didn't run successfully, log only total
                       ROS_INFO_THROTTLE(1.0, "Control Timing (ms): Total: %ld (State: %d, Inference Error/Skipped)", 
                                         total_duration.count(), static_cast<int>(current_control_state_));
                  }
              } else {
                  ROS_INFO_THROTTLE(1.0, "State: RUNNING_POLICY (%s Policy, Initializing History). Press SELECT to shutdown.",
                                    use_heightmap_policy_ ? "Heightmap" : "Default");
                  // History not ready yet, command default pose
                  SetInitPoseCommand(motor_command_tmp);

                  // Log timing for this sub-state
                  auto loop_end_time = std::chrono::high_resolution_clock::now();
                  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end_time - control_loop_start_time);
                  ROS_INFO_THROTTLE(1.0, "Control Timing (ms): Total: %ld (State: %d, History Init)", 
                                   total_duration.count(), static_cast<int>(current_control_state_));
              }

              // Check for shutdown transition
              // if (gamepad_.select.on_press) {
              if (gamepad_.select.pressed) {
                  ROS_INFO("SELECT pressed. Transitioning to SHUTDOWN.");
                  current_control_state_ = ControlState::SHUTDOWN;
              }
          }
              break;

          case ControlState::SHUTDOWN:
              ROS_INFO_THROTTLE(1.0, "State: SHUTDOWN. Holding position and shutting down ROS.");
              SetHoldCommand(motor_command_tmp, *ms);
              if (ros::ok()) { // Check if ROS is still running before shutting down
                 ros::shutdown();
              }
              break;
      }

      // --- 4. Buffer the calculated MotorCommand ---
      motor_command_buffer_.SetData(motor_command_tmp);
  }


  // --- DDS Command Writer (Runs at control_dt_ interval or faster) ---
  // Reads buffered command and sends it via DDS
  void LowCommandWriter() {
      // Get the latest command from the buffer
      const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();

      // Only send if a valid command is buffered
      if (mc) {
          LowCmd_ dds_low_command;
          dds_low_command.mode_pr() = static_cast<uint8_t>(Mode::PR); // Use PR mode (or AB if needed)
          dds_low_command.mode_machine() = mode_machine_; // Pass current machine mode

          for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
              dds_low_command.motor_cmd().at(i).mode() = 1; // 1:Enable PD control
              dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
              dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
              dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
              dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
              dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
          }

          // Calculate and set CRC
          dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
          // Publish command
          lowcmd_publisher_->Write(dds_low_command);
      } else {
          // No command buffered yet (e.g., during initialization)
          // ROS_INFO_THROTTLE(1.0, "No command buffered in LowCommandWriter.");
          // Optionally send a zero/safe command here if needed.
      }
  }


  // --- ROS Callback for Elevation Cloud ---
  // Updates the heightmap grid based on incoming point cloud data
  void ElevationCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    auto callback_start_time = std::chrono::high_resolution_clock::now();

    geometry_msgs::TransformStamped transform_cloud_to_base;
    geometry_msgs::TransformStamped transform_torso_to_base;
    Eigen::Vector3f torso_pos_eigen;
    Eigen::Quaternionf torso_rot_eigen;

    try {
        ros::Time now = ros::Time(0); // Get latest available transform
        // Get transform from cloud frame to the base frame (e.g., odom_corrected)
        transform_cloud_to_base = tf_buffer_->lookupTransform(ROS_BASE_FRAME, msg->header.frame_id, now, ros::Duration(0.1));
        // Get transform from torso frame to the base frame
        transform_torso_to_base = tf_buffer_->lookupTransform(ROS_BASE_FRAME, ROS_TORSO_FRAME, now, ros::Duration(0.1));

        // Convert torso transform to Eigen
        Eigen::Isometry3d torso_transform_iso = tf2::transformToEigen(transform_torso_to_base);
        torso_pos_eigen = torso_transform_iso.translation().cast<float>();
        torso_rot_eigen = Eigen::Quaternionf(torso_transform_iso.rotation().cast<float>());
    } catch (tf2::TransformException &ex) {
        ROS_WARN_THROTTLE(1.0, "Heightmap TF lookup failed: %s. Skipping heightmap update.", ex.what());
        return;
    }

    // Convert ROS PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    // Remove NaN points
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    if (cloud->empty()) {
        // ROS_WARN_THROTTLE(1.0, "Point cloud is empty after removing NaNs. Skipping heightmap update.");
        return; // Keep existing heights if cloud is temporarily empty
    }

    // Transform point cloud to the base frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3d transform_eigen_double = tf2::transformToEigen(transform_cloud_to_base);
    Eigen::Affine3f transform_eigen_float = transform_eigen_double.cast<float>();
    pcl::transformPointCloud(*cloud, *world_cloud, transform_eigen_float);

    // Build a 2D KdTree from the transformed world cloud for faster searching
    pcl::PointCloud<pcl::PointXY>::Ptr cloud_xy(new pcl::PointCloud<pcl::PointXY>);
    cloud_xy->points.resize(world_cloud->points.size());
    for(size_t i = 0; i < world_cloud->points.size(); ++i) {
        cloud_xy->points[i].x = world_cloud->points[i].x;
        cloud_xy->points[i].y = world_cloud->points[i].y;
    }

    pcl::KdTreeFLANN<pcl::PointXY>::Ptr kdtree_2d(new pcl::KdTreeFLANN<pcl::PointXY>());
    if (!cloud_xy->empty()) {
      kdtree_2d->setInputCloud(cloud_xy);
    } else {
       // ROS_WARN_THROTTLE(1.0, "2D projected cloud is empty, cannot build KdTree. Skipping heightmap update.");
      return; // Keep existing heights if cloud is bad
    }

    // Update the heightmap grid using the processed cloud and robot pose
    updateHeightmap(world_cloud, torso_pos_eigen, torso_rot_eigen, kdtree_2d);

    // printHeightmap(); // Optional: print for debugging

    auto callback_end_time = std::chrono::high_resolution_clock::now();
    auto callback_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end_time - callback_start_time);
    ROS_INFO_THROTTLE(1.0, "ElevationCloudCallback duration: %ld ms", callback_duration_ms.count());
  }
  // --- End Elevation Cloud Callback ---

  // --- Heightmap Helper Functions (Unchanged) ---
  void createElevationGrid() {
    local_query_points_.resize(grid_points_count_, 2);

     // 11 / 2 * 0.1 = 0.55
    float half_x = GRID_X_SIDE_LENGTH / 2.0 * GRID_X_RESOLUTION;
    float half_y = GRID_Y_SIDE_LENGTH / 2.0 * GRID_Y_RESOLUTION;

    int point_idx = 0;
    for (int j = 0; j < GRID_Y_SIDE_LENGTH; ++j) { // Corresponds to Y index
        float y = -half_y + (j+0.5) * GRID_Y_RESOLUTION;
        for (int i = 0; i < GRID_X_SIDE_LENGTH; ++i) { // Corresponds to X index
            float x = -half_x + (i+0.5) * GRID_X_RESOLUTION;
            local_query_points_(point_idx, 0) = x;
            local_query_points_(point_idx, 1) = y;
            point_idx++;
        }
    }
    if(point_idx != grid_points_count_) {
        ROS_ERROR("Grid point count mismatch! Expected %d, got %d", grid_points_count_, point_idx);
    }
  }

  Eigen::MatrixXf getGlobalQueryPoints(const Eigen::Vector3f& torso_pos, const Eigen::Quaternionf& torso_rot) {
    Eigen::MatrixXf global_points(grid_points_count_, 2);
    Eigen::Matrix3f rot_matrix_float = torso_rot.toRotationMatrix();
    float yaw = std::atan2(rot_matrix_float(1, 0), rot_matrix_float(0, 0));
    Eigen::Matrix2f rotm_yaw_only;
    rotm_yaw_only << std::cos(yaw), -std::sin(yaw),
                     std::sin(yaw),  std::cos(yaw);

    Eigen::Vector2f torso_xy = torso_pos.head<2>();
    // Rotate local points by yaw and translate by torso XY position
    global_points = (rotm_yaw_only * local_query_points_.transpose()).transpose().rowwise() + torso_xy.transpose();
    return global_points;
  }

  // Interpolates height at a query point using Inverse Distance Weighting (IDW) from nearest neighbors
 float interpolateHeightIDW(const Eigen::Vector2f& query_point_xy,
                           const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& original_cloud, // Original 3D cloud
                           float torso_z, // Torso Z position in the base frame
                           const pcl::KdTreeFLANN<pcl::PointXY>::Ptr& kdtree_2d) // Pre-built 2D tree
  {
    if (!kdtree_2d || !original_cloud || original_cloud->empty()) {
        // ROS_WARN_THROTTLE(1.0, "Invalid input to interpolateHeightIDW.");
        return KNN_DEFAULT_HEIGHT_OFFSET; // Use default if inputs are bad
    }

    std::vector<int> pointIdxNKNSearch(KNN_K);
    std::vector<float> pointNKNSquaredDistance(KNN_K);

    pcl::PointXY searchPoint_xy;
    searchPoint_xy.x = query_point_xy.x();
    searchPoint_xy.y = query_point_xy.y();

    // Search for K nearest neighbors in the 2D projection
    int found_neighbors = kdtree_2d->nearestKSearch(searchPoint_xy, KNN_K, pointIdxNKNSearch, pointNKNSquaredDistance);

    if (found_neighbors > 0) {
        // Basic validation of indices (should correspond to original_cloud)
        for(int idx : pointIdxNKNSearch) {
            if (idx < 0 || idx >= original_cloud->points.size()) {
                ROS_ERROR("Invalid index %d from kNN search (cloud size %zu)", idx, original_cloud->points.size());
                return KNN_DEFAULT_HEIGHT_OFFSET; // Invalid index -> use default
            }
        }

        // Check distance to the *closest* neighbor
        if (std::sqrt(pointNKNSquaredDistance[0]) > KNN_MAX_DISTANCE) {
            // ROS_WARN_THROTTLE(1.0, "Closest neighbor too far (%.2f m > %.2f m) for query point (%.2f, %.2f)",
            //                   std::sqrt(pointNKNSquaredDistance[0]), KNN_MAX_DISTANCE, query_point_xy.x(), query_point_xy.y());
            return KNN_DEFAULT_HEIGHT_OFFSET; // Too far, use default
        }

        float total_weight = 0.0f;
        float weighted_sum_z = 0.0f;
        const float epsilon = 1e-6f; // Small value to prevent division by zero

        for (int i = 0; i < found_neighbors; ++i) {
            float dist_sq = pointNKNSquaredDistance[i];
            // If query point is exactly a cloud point
            if (dist_sq < epsilon * epsilon) {
                // Return height relative to torso Z
                return torso_z - original_cloud->points[pointIdxNKNSearch[i]].z;
            }
            float dist = std::sqrt(dist_sq);
            float weight = 1.0f / (dist + epsilon); // IDW weight
            weighted_sum_z += original_cloud->points[pointIdxNKNSearch[i]].z * weight;
            total_weight += weight;
        }

        if (total_weight > epsilon) {
            float interpolated_z_world = weighted_sum_z / total_weight;
            // Return height relative to the torso's Z position
            return torso_z - interpolated_z_world;
        } else {
            ROS_WARN_THROTTLE(1.0, "Interpolation resulted in zero total weight. Using default offset.");
            return KNN_DEFAULT_HEIGHT_OFFSET;
        }

    } else {
        // No neighbors found within search radius
        // ROS_WARN_THROTTLE(1.0, "No neighbors found for query point (%.2f, %.2f)", query_point_xy.x(), query_point_xy.y());
        return KNN_DEFAULT_HEIGHT_OFFSET; // Use default
    }
  }

  // Updates the heightmap grid values based on the latest point cloud and robot pose
  void updateHeightmap(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& world_cloud,
                         const Eigen::Vector3f& torso_pos,
                         const Eigen::Quaternionf& torso_rot,
                         const pcl::KdTreeFLANN<pcl::PointXY>::Ptr& kdtree_2d) // Pass the 2D tree
  {
    if (!world_cloud || world_cloud->empty() || !kdtree_2d) {
        // ROS_WARN_THROTTLE(1.0, "Skipping heightmap update due to invalid input cloud/tree.");
        return; // Keep existing heights if data is bad
    }

    // Get query points in the global frame (base frame)
    Eigen::MatrixXf global_query_points_xy = getGlobalQueryPoints(torso_pos, torso_rot);

    // Iterate through the grid and interpolate height for each point
    int point_idx = 0;
    for (int j = 0; j < GRID_Y_SIDE_LENGTH; ++j) { // Row index (Y)
        for (int i = 0; i < GRID_X_SIDE_LENGTH; ++i) { // Column index (X)
             if (point_idx < global_query_points_xy.rows()) { // Bounds check
                Eigen::Vector2f query_xy = global_query_points_xy.row(point_idx);
                // Interpolate height at the query point and store it relative to torso Z
                heights_(j, i) = interpolateHeightIDW(query_xy, world_cloud, torso_pos.z(), kdtree_2d);
             } else {
                  ROS_ERROR("Point index %d out of bounds for global query points (%ld rows)", point_idx, global_query_points_xy.rows());
             }
            point_idx++;
        }
    }
  }

  // Optional: Prints the current heightmap to ROS_INFO
  void printHeightmap() {
    std::stringstream ss;
    ss << "Current Heightmap (" << GRID_Y_SIDE_LENGTH << "x" << GRID_X_SIDE_LENGTH << ") Relative to Torso Z:\n";
    ss << std::fixed << std::setprecision(3);
    for (int j = 0; j < GRID_Y_SIDE_LENGTH; ++j) {
        for (int i = 0; i < GRID_X_SIDE_LENGTH; ++i) {
            ss << std::setw(8) << heights_(j, i);
        }
        ss << "\n";
    }
    ROS_INFO_STREAM_THROTTLE(1.0, "\n" << ss.str()); // Throttle to avoid spamming logs
  }
  // --- End Heightmap Helpers ---

}; // End class VideomimicInferenceReal

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: g1_ankle_policy_ros_example network_interface [ros_args]" << std::endl;
    exit(1);
  }
  std::string networkInterface = argv[1];

  // Initialize ROS
  ros::init(argc, argv, "g1_ankle_policy_ros_node");
  ros::NodeHandle n;
  ROS_INFO("ROS node initialized.");

  try {
    // Create the main class instance
    VideomimicInferenceReal custom(networkInterface, &n);
    ROS_INFO("G1 Policy ROS Example started. Waiting for ROS shutdown (Ctrl+C).");

    // Use AsyncSpinner to handle ROS callbacks (e.g., PointCloud) concurrently
    // Number of threads should be sufficient for subscribers + potentially other background tasks.
    ros::AsyncSpinner spinner(2); // Use 2 threads (e.g., 1 for Cloud CB, 1 for others)
    spinner.start();

    // Keep the main thread alive until ROS is shut down
    ros::waitForShutdown();

    ROS_INFO("ROS shutdown request received. Exiting.");

  } catch (const std::exception& e) {
      ROS_FATAL("Unhandled exception in main: %s", e.what());
      return 1;
  } catch (...) {
      ROS_FATAL("Unknown unhandled exception in main.");
      return 1;
  }

  return 0;
}
