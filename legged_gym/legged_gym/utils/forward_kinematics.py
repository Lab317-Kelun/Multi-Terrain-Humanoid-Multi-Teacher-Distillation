# Forward Kinematics for AMP
# 基于G1真实URDF的完整Forward Kinematics实现
# 与MimicKit的AMP实现保持一致

import torch
import numpy as np
from isaacgym.torch_utils import quat_rotate, quat_mul, quat_from_angle_axis, quat_from_euler_xyz


class G1ForwardKinematics:
    """
    G1机器人的Forward Kinematics计算
    
    基于G1 URDF的真实运动学链，精确计算关键身体部位的位置
    所有参数从 g1.urdf 中提取
    """
    
    def __init__(self, device='cuda'):
        """
        初始化G1机器人的运动学参数
        
        所有数值直接从 /home/cft/kelun/Humanoid-Terrain-Bench/legged_gym/resources/robots/g1_description/g1.urdf 提取
        """
        self.device = device
        
        # ===== Torso位置 (从pelvis出发) =====
        # waist_yaw_joint + waist_roll_link + torso_link
        self.torso_offset_from_pelvis = torch.tensor([
            -0.0039635,  # waist_roll_link的x偏移
            0.0,
            0.044        # waist_roll_link的z偏移
        ], device=device)
        
        # ===== 左腿运动学参数 (从URDF精确提取) =====
        # Hip pitch joint: pelvis -> left_hip_pitch_link
        self.left_hip_pitch_offset = torch.tensor([0.0, 0.064452, -0.1027], device=device)
        
        # Hip roll joint: left_hip_pitch_link -> left_hip_roll_link
        self.left_hip_roll_offset = torch.tensor([0.0, 0.052, -0.030465], device=device)
        self.left_hip_roll_rpy = torch.tensor([0.0, -0.1749, 0.0], device=device)  # -10度预旋转
        
        # Hip yaw joint: left_hip_roll_link -> left_hip_yaw_link
        self.left_hip_yaw_offset = torch.tensor([0.025001, 0.0, -0.12412], device=device)
        
        # Knee joint: left_hip_yaw_link -> left_knee_link
        self.left_knee_offset = torch.tensor([-0.078273, 0.0021489, -0.17734], device=device)
        self.left_knee_rpy = torch.tensor([0.0, 0.1749, 0.0], device=device)  # 10度预旋转
        
        # Ankle pitch joint: left_knee_link -> left_ankle_pitch_link
        self.left_ankle_pitch_offset = torch.tensor([0.0, -9.4445e-05, -0.30001], device=device)
        
        # Ankle roll joint: left_ankle_pitch_link -> left_ankle_roll_link
        self.left_ankle_roll_offset = torch.tensor([0.0, 0.0, -0.017558], device=device)
        
        # 脚底偏移（从ankle_roll_link到实际接触点）
        self.left_foot_contact_offset = torch.tensor([0.0, 0.0, -0.04], device=device)
        
        # ===== 右腿运动学参数 =====
        self.right_hip_pitch_offset = torch.tensor([0.0, -0.064452, -0.1027], device=device)
        self.right_hip_roll_offset = torch.tensor([0.0, -0.052, -0.030465], device=device)
        self.right_hip_roll_rpy = torch.tensor([0.0, -0.1749, 0.0], device=device)
        self.right_hip_yaw_offset = torch.tensor([0.025001, 0.0, -0.12412], device=device)
        self.right_knee_offset = torch.tensor([-0.078273, -0.0021489, -0.17734], device=device)
        self.right_knee_rpy = torch.tensor([0.0, 0.1749, 0.0], device=device)
        self.right_ankle_pitch_offset = torch.tensor([0.0, 9.4445e-05, -0.30001], device=device)
        self.right_ankle_roll_offset = torch.tensor([0.0, 0.0, -0.017558], device=device)
        self.right_foot_contact_offset = torch.tensor([0.0, 0.0, -0.04], device=device)
        
        # ===== 左臂运动学参数 =====
        # Shoulder pitch: torso_link -> left_shoulder_pitch_link
        self.left_shoulder_pitch_offset = torch.tensor([0.0039563, 0.10022, 0.24778], device=device)
        self.left_shoulder_pitch_rpy = torch.tensor([0.27931, 5.4949e-05, -0.00019159], device=device)
        
        # Shoulder roll: left_shoulder_pitch_link -> left_shoulder_roll_link
        self.left_shoulder_roll_offset = torch.tensor([0.0, 0.038, -0.013831], device=device)
        self.left_shoulder_roll_rpy = torch.tensor([-0.27925, 0.0, 0.0], device=device)
        
        # Shoulder yaw: left_shoulder_roll_link -> left_shoulder_yaw_link
        self.left_shoulder_yaw_offset = torch.tensor([0.0, 0.00624, -0.1032], device=device)
        
        # Elbow: left_shoulder_yaw_link -> left_elbow_link
        self.left_elbow_offset = torch.tensor([0.015783, 0.0, -0.080518], device=device)
        
        # Wrist: left_elbow_link -> ... -> left_hand_palm_link
        # 简化：直接到手掌中心
        self.left_wrist_to_palm_offset = torch.tensor([
            0.1 + 0.038 + 0.046 + 0.0415,  # wrist_roll + wrist_pitch + wrist_yaw + palm
            0.00188791 + 0.003,
            -0.01
        ], device=device)
        
        # ===== 右臂运动学参数 =====
        self.right_shoulder_pitch_offset = torch.tensor([0.0039563, -0.10021, 0.24778], device=device)
        self.right_shoulder_pitch_rpy = torch.tensor([-0.27931, 5.4949e-05, 0.00019159], device=device)
        self.right_shoulder_roll_offset = torch.tensor([0.0, -0.038, -0.013831], device=device)
        self.right_shoulder_roll_rpy = torch.tensor([0.27925, 0.0, 0.0], device=device)
        self.right_shoulder_yaw_offset = torch.tensor([0.0, -0.00624, -0.1032], device=device)
        self.right_elbow_offset = torch.tensor([0.015783, 0.0, -0.080518], device=device)
        self.right_wrist_to_palm_offset = torch.tensor([
            0.1 + 0.038 + 0.046 + 0.0415,
            -(0.00188791 + 0.003),
            -0.01
        ], device=device)
        
        print(f"[FK] G1 Forward Kinematics初始化完成 (基于真实URDF)")
        print(f"[FK] 腿部总长度: ~0.82m, 臂展: ~0.5m")
    
    def compute_key_body_positions(self, root_pos, root_rot, dof_pos):
        """
        计算关键身体部位的世界坐标位置
        
        Args:
            root_pos: [batch, 3] root位置（世界坐标，pelvis）
            root_rot: [batch, 4] root旋转（四元数）
            dof_pos: [batch, 27] 关节角度
            
            G1机器人DOF顺序（从humanoid_beamdojo_config.py）:
            0-5: 左腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
            6-11: 右腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
            12: waist_yaw
            13-19: 左臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
            20-26: 右臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
            
        Returns:
            key_body_pos: [batch, 5, 3] 关键身体部位位置
                0: torso (躯干中心)
                1: left_hand (左手掌中心)
                2: right_hand (右手掌中心)
                3: left_foot (左脚底中心)
                4: right_foot (右脚底中心)
        """
        batch_size = root_pos.shape[0]
        key_body_pos = torch.zeros(batch_size, 5, 3, device=self.device)
        
        # ===== 1. Torso位置 =====
        # pelvis -> waist_yaw -> waist_roll -> torso
        waist_yaw = dof_pos[:, 12] if dof_pos.shape[1] > 12 else torch.zeros(batch_size, device=self.device)
        torso_pos = self._compute_torso_position(root_pos, root_rot, waist_yaw)
        key_body_pos[:, 0, :] = torso_pos
        
        # ===== 2. 左手位置 =====
        # torso -> shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow -> wrist -> hand
        left_hand_pos = self._compute_left_hand_position(
            root_pos, root_rot,
            dof_pos[:, 12],  # waist_yaw
            dof_pos[:, 13],  # left_shoulder_pitch
            dof_pos[:, 14],  # left_shoulder_roll
            dof_pos[:, 15],  # left_shoulder_yaw
            dof_pos[:, 16],  # left_elbow
        )
        key_body_pos[:, 1, :] = left_hand_pos
        
        # ===== 3. 右手位置 =====
        right_hand_pos = self._compute_right_hand_position(
            root_pos, root_rot,
            dof_pos[:, 12],  # waist_yaw
            dof_pos[:, 20],  # right_shoulder_pitch
            dof_pos[:, 21],  # right_shoulder_roll
            dof_pos[:, 22],  # right_shoulder_yaw
            dof_pos[:, 23],  # right_elbow
        )
        key_body_pos[:, 2, :] = right_hand_pos
        
        # ===== 4. 左脚位置 =====
        # pelvis -> hip_pitch -> hip_roll -> hip_yaw -> knee -> ankle_pitch -> ankle_roll -> foot
        left_foot_pos = self._compute_left_foot_position(
            root_pos, root_rot,
            dof_pos[:, 0],   # left_hip_pitch
            dof_pos[:, 1],   # left_hip_roll
            dof_pos[:, 2],   # left_hip_yaw
            dof_pos[:, 3],   # left_knee
            dof_pos[:, 4],   # left_ankle_pitch
            dof_pos[:, 5],   # left_ankle_roll
        )
        key_body_pos[:, 3, :] = left_foot_pos
        
        # ===== 5. 右脚位置 =====
        right_foot_pos = self._compute_right_foot_position(
            root_pos, root_rot,
            dof_pos[:, 6],   # right_hip_pitch
            dof_pos[:, 7],   # right_hip_roll
            dof_pos[:, 8],   # right_hip_yaw
            dof_pos[:, 9],   # right_knee
            dof_pos[:, 10],  # right_ankle_pitch
            dof_pos[:, 11],  # right_ankle_roll
        )
        key_body_pos[:, 4, :] = right_foot_pos
        
        return key_body_pos
    
    def _compute_torso_position(self, root_pos, root_rot, waist_yaw):
        """
        计算Torso位置
        运动学链: pelvis -> waist_yaw -> waist_roll_link -> torso_link
        """
        batch_size = root_pos.shape[0]
        
        # Waist yaw旋转
        waist_yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        waist_yaw_rot = quat_from_angle_axis(waist_yaw, waist_yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(root_rot, waist_yaw_rot)
        
        # Torso偏移
        torso_offset = self.torso_offset_from_pelvis.unsqueeze(0).expand(batch_size, -1)
        torso_pos = root_pos + quat_rotate(current_rot, torso_offset)
        
        return torso_pos
    
    def _compute_left_foot_position(self, root_pos, root_rot, hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll):
        """
        计算左脚位置
        运动学链: pelvis -> hip_pitch -> hip_roll -> hip_yaw -> knee -> ankle_pitch -> ankle_roll -> foot
        """
        batch_size = root_pos.shape[0]
        current_pos = root_pos
        current_rot = root_rot
        
        # Hip pitch joint
        offset = self.left_hip_pitch_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        pitch_rot = quat_from_angle_axis(hip_pitch, pitch_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, pitch_rot)
        
        # Hip roll joint (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.left_hip_roll_rpy[0].unsqueeze(0).expand(batch_size),
            self.left_hip_roll_rpy[1].unsqueeze(0).expand(batch_size),
            self.left_hip_roll_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.left_hip_roll_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        roll_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        roll_rot = quat_from_angle_axis(hip_roll, roll_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, roll_rot)
        
        # Hip yaw joint
        offset = self.left_hip_yaw_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        yaw_rot = quat_from_angle_axis(hip_yaw, yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, yaw_rot)
        
        # Knee joint (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.left_knee_rpy[0].unsqueeze(0).expand(batch_size),
            self.left_knee_rpy[1].unsqueeze(0).expand(batch_size),
            self.left_knee_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.left_knee_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        knee_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        knee_rot = quat_from_angle_axis(knee, knee_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, knee_rot)
        
        # Ankle pitch joint
        offset = self.left_ankle_pitch_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        ankle_pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        ankle_pitch_rot = quat_from_angle_axis(ankle_pitch, ankle_pitch_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, ankle_pitch_rot)
        
        # Ankle roll joint
        offset = self.left_ankle_roll_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        ankle_roll_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        ankle_roll_rot = quat_from_angle_axis(ankle_roll, ankle_roll_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, ankle_roll_rot)
        
        # Foot contact point
        offset = self.left_foot_contact_offset.unsqueeze(0).expand(batch_size, -1)
        foot_pos = current_pos + quat_rotate(current_rot, offset)
        
        return foot_pos
    
    def _compute_right_foot_position(self, root_pos, root_rot, hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll):
        """
        计算右脚位置（与左脚对称）
        """
        batch_size = root_pos.shape[0]
        current_pos = root_pos
        current_rot = root_rot
        
        # Hip pitch
        offset = self.right_hip_pitch_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        pitch_rot = quat_from_angle_axis(hip_pitch, pitch_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, pitch_rot)
        
        # Hip roll (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.right_hip_roll_rpy[0].unsqueeze(0).expand(batch_size),
            self.right_hip_roll_rpy[1].unsqueeze(0).expand(batch_size),
            self.right_hip_roll_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.right_hip_roll_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        roll_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        roll_rot = quat_from_angle_axis(hip_roll, roll_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, roll_rot)
        
        # Hip yaw
        offset = self.right_hip_yaw_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        yaw_rot = quat_from_angle_axis(hip_yaw, yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, yaw_rot)
        
        # Knee (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.right_knee_rpy[0].unsqueeze(0).expand(batch_size),
            self.right_knee_rpy[1].unsqueeze(0).expand(batch_size),
            self.right_knee_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.right_knee_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        knee_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        knee_rot = quat_from_angle_axis(knee, knee_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, knee_rot)
        
        # Ankle pitch
        offset = self.right_ankle_pitch_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        ankle_pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        ankle_pitch_rot = quat_from_angle_axis(ankle_pitch, ankle_pitch_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, ankle_pitch_rot)
        
        # Ankle roll
        offset = self.right_ankle_roll_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        ankle_roll_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        ankle_roll_rot = quat_from_angle_axis(ankle_roll, ankle_roll_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, ankle_roll_rot)
        
        # Foot contact
        offset = self.right_foot_contact_offset.unsqueeze(0).expand(batch_size, -1)
        foot_pos = current_pos + quat_rotate(current_rot, offset)
        
        return foot_pos
    
    def _compute_left_hand_position(self, root_pos, root_rot, waist_yaw, shoulder_pitch, shoulder_roll, shoulder_yaw, elbow):
        """
        计算左手位置
        运动学链: pelvis -> waist_yaw -> torso -> shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow -> wrist -> hand
        """
        batch_size = root_pos.shape[0]
        
        # 先到达torso
        waist_yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        waist_yaw_rot = quat_from_angle_axis(waist_yaw, waist_yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(root_rot, waist_yaw_rot)
        torso_offset = self.torso_offset_from_pelvis.unsqueeze(0).expand(batch_size, -1)
        current_pos = root_pos + quat_rotate(current_rot, torso_offset)
        
        # Shoulder pitch (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.left_shoulder_pitch_rpy[0].unsqueeze(0).expand(batch_size),
            self.left_shoulder_pitch_rpy[1].unsqueeze(0).expand(batch_size),
            self.left_shoulder_pitch_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.left_shoulder_pitch_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        pitch_rot = quat_from_angle_axis(shoulder_pitch, pitch_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, pitch_rot)
        
        # Shoulder roll (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.left_shoulder_roll_rpy[0].unsqueeze(0).expand(batch_size),
            self.left_shoulder_roll_rpy[1].unsqueeze(0).expand(batch_size),
            self.left_shoulder_roll_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.left_shoulder_roll_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        roll_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        roll_rot = quat_from_angle_axis(shoulder_roll, roll_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, roll_rot)
        
        # Shoulder yaw
        offset = self.left_shoulder_yaw_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        yaw_rot = quat_from_angle_axis(shoulder_yaw, yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, yaw_rot)
        
        # Elbow
        offset = self.left_elbow_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        elbow_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        elbow_rot = quat_from_angle_axis(elbow, elbow_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, elbow_rot)
        
        # Wrist to palm (simplified)
        offset = self.left_wrist_to_palm_offset.unsqueeze(0).expand(batch_size, -1)
        hand_pos = current_pos + quat_rotate(current_rot, offset)
        
        return hand_pos
    
    def _compute_right_hand_position(self, root_pos, root_rot, waist_yaw, shoulder_pitch, shoulder_roll, shoulder_yaw, elbow):
        """
        计算右手位置（与左手对称）
        """
        batch_size = root_pos.shape[0]
        
        # Torso
        waist_yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        waist_yaw_rot = quat_from_angle_axis(waist_yaw, waist_yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(root_rot, waist_yaw_rot)
        torso_offset = self.torso_offset_from_pelvis.unsqueeze(0).expand(batch_size, -1)
        current_pos = root_pos + quat_rotate(current_rot, torso_offset)
        
        # Shoulder pitch (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.right_shoulder_pitch_rpy[0].unsqueeze(0).expand(batch_size),
            self.right_shoulder_pitch_rpy[1].unsqueeze(0).expand(batch_size),
            self.right_shoulder_pitch_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.right_shoulder_pitch_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        pitch_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        pitch_rot = quat_from_angle_axis(shoulder_pitch, pitch_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, pitch_rot)
        
        # Shoulder roll (with pre-rotation)
        pre_rot = quat_from_euler_xyz(
            self.right_shoulder_roll_rpy[0].unsqueeze(0).expand(batch_size),
            self.right_shoulder_roll_rpy[1].unsqueeze(0).expand(batch_size),
            self.right_shoulder_roll_rpy[2].unsqueeze(0).expand(batch_size)
        )
        current_rot = quat_mul(current_rot, pre_rot)
        offset = self.right_shoulder_roll_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        roll_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        roll_rot = quat_from_angle_axis(shoulder_roll, roll_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, roll_rot)
        
        # Shoulder yaw
        offset = self.right_shoulder_yaw_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        yaw_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        yaw_rot = quat_from_angle_axis(shoulder_yaw, yaw_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, yaw_rot)
        
        # Elbow
        offset = self.right_elbow_offset.unsqueeze(0).expand(batch_size, -1)
        current_pos = current_pos + quat_rotate(current_rot, offset)
        elbow_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        elbow_rot = quat_from_angle_axis(elbow, elbow_axis.unsqueeze(0).expand(batch_size, -1))
        current_rot = quat_mul(current_rot, elbow_rot)
        
        # Wrist to palm
        offset = self.right_wrist_to_palm_offset.unsqueeze(0).expand(batch_size, -1)
        hand_pos = current_pos + quat_rotate(current_rot, offset)
        
        return hand_pos

