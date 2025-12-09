import numpy as np
import pybullet as p
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import time


class GraspState(Enum):
    """抓取状态机状态"""
    IDLE = auto()                    # 空闲状态
    SEARCHING = auto()               # 搜索目标
    APPROACHING = auto()             # 接近目标
    ALIGNING = auto()                # 对准目标
    PRE_GRASP = auto()              # 预抓取位置
    DESCENDING = auto()             # 下降抓取
    GRASPING = auto()               # 闭合夹爪
    LIFTING = auto()                # 提升物体
    TRANSPORTING = auto()           # 运输物体
    PLACING = auto()                # 放置物体
    RELEASING = auto()              # 释放物体
    RETREATING = auto()             # 撤退
    COMPLETED = auto()              # 完成
    ERROR = auto()                  # 错误状态


@dataclass
class ControlConfig:
    """控制参数配置"""
    # 视觉伺服参数
    servo_gain_p: float = 0.8           # 位置比例增益
    servo_gain_d: float = 0.1           # 位置微分增益
    servo_max_vel: float = 0.5          # 最大线速度 (m/s)
    servo_threshold: float = 0.01       # 到达阈值 (m)
    
    # 抓取参数
    approach_height: float = 0.15       # 接近高度 (物体上方)
    grasp_height_offset: float = 0.0    # 抓取高度偏移
    lift_height: float = 0.15           # 提升高度
    grasp_settling_time: float = 0.5    # 夹爪稳定时间
    
    # 安全参数
    workspace_min: np.ndarray = None    # 工作空间最小边界
    workspace_max: np.ndarray = None    # 工作空间最大边界
    joint_velocity_limit: float = 2.0   # 关节速度限制
    
    def __post_init__(self):
        if self.workspace_min is None:
            self.workspace_min = np.array([0.2, -0.8, 0.0])
        if self.workspace_max is None:
            self.workspace_max = np.array([1.0, 0.8, 1.0])


class PBVSController:
    """
    基于位置的视觉伺服控制器 (Position-Based Visual Servoing)
    
    使用目标在机器人基坐标系中的位置进行控制
    """
    
    def __init__(self, config: ControlConfig = None):
        """
        初始化PBVS控制器
        
        Args:
            config: 控制参数配置
        """
        self.config = config if config else ControlConfig()
        
        # 控制状态
        self.target_position = None
        self.target_orientation = None
        self.previous_error = np.zeros(3)
        self.integral_error = np.zeros(3)
        
        # 时间记录
        self.last_update_time = None
    
    def set_target(self, position: np.ndarray, orientation: np.ndarray = None):
        """
        设置目标位姿
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标姿态 (四元数) [x, y, z, w]
        """
        self.target_position = np.array(position)
        if orientation is not None:
            self.target_orientation = np.array(orientation)
        
        # 重置误差积分
        self.integral_error = np.zeros(3)
    
    def compute_velocity(self, current_position: np.ndarray, 
                         dt: float = None) -> Tuple[np.ndarray, float]:
        """
        计算期望速度
        
        Args:
            current_position: 当前末端位置 [x, y, z]
            dt: 时间步长
            
        Returns:
            velocity: 期望线速度 [vx, vy, vz]
            error_norm: 位置误差范数
        """
        if self.target_position is None:
            return np.zeros(3), float('inf')
        
        # 计算位置误差
        error = self.target_position - current_position
        error_norm = np.linalg.norm(error)
        
        # 计算时间步长
        current_time = time.time()
        if dt is None:
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
            else:
                dt = 0.01
        self.last_update_time = current_time
        
        # PD控制
        if dt > 0:
            error_derivative = (error - self.previous_error) / dt
        else:
            error_derivative = np.zeros(3)
        
        velocity = (self.config.servo_gain_p * error + 
                   self.config.servo_gain_d * error_derivative)
        
        # 限制速度
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > self.config.servo_max_vel:
            velocity = velocity * self.config.servo_max_vel / vel_norm
        
        self.previous_error = error.copy()
        
        return velocity, error_norm
    
    def compute_target_pose(self, current_position: np.ndarray,
                           dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        计算期望目标位姿 (用于位置控制)
        
        Args:
            current_position: 当前末端位置
            dt: 时间步长
            
        Returns:
            target_pos: 期望位置
            target_orn: 期望姿态
            reached: 是否到达目标
        """
        velocity, error_norm = self.compute_velocity(current_position, dt)
        
        # 增量位置
        target_pos = current_position + velocity * dt
        
        # 工作空间约束
        target_pos = np.clip(target_pos, 
                            self.config.workspace_min, 
                            self.config.workspace_max)
        
        # 检查是否到达目标
        reached = error_norm < self.config.servo_threshold
        
        return target_pos, self.target_orientation, reached
    
    def is_target_reached(self, current_position: np.ndarray) -> bool:
        """检查是否到达目标"""
        if self.target_position is None:
            return False
        error = np.linalg.norm(self.target_position - current_position)
        return error < self.config.servo_threshold
    
    def reset(self):
        """重置控制器状态"""
        self.target_position = None
        self.target_orientation = None
        self.previous_error = np.zeros(3)
        self.integral_error = np.zeros(3)
        self.last_update_time = None


class IBVSController:
    """
    基于图像的视觉伺服控制器 (Image-Based Visual Servoing)
    
    使用图像特征误差直接计算控制命令
    """
    
    def __init__(self, image_width: int = 640, image_height: int = 480,
                 focal_length: float = 500.0, gain: float = 0.5):
        """
        初始化IBVS控制器
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            focal_length: 焦距 (像素)
            gain: 控制增益
        """
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.gain = gain
        
        # 图像中心
        self.cx = image_width / 2
        self.cy = image_height / 2
    
    def compute_velocity(self, image_error: np.ndarray, 
                        depth: float = 0.5) -> np.ndarray:
        """
        根据图像误差计算相机速度
        
        Args:
            image_error: 图像特征误差 [ex, ey] (像素)
            depth: 目标深度 (米)
            
        Returns:
            velocity: 相机速度 [vx, vy, vz, wx, wy, wz]
        """
        ex, ey = image_error
        
        # 简化的IBVS: 仅控制x, y平移
        # 图像雅可比矩阵的简化形式
        vx = -self.gain * ex / self.focal_length * depth
        vy = -self.gain * ey / self.focal_length * depth
        vz = 0  # 可以根据目标大小调整
        
        return np.array([vx, vy, vz, 0, 0, 0])


class GripperController:
    """夹爪控制器"""
    
    def __init__(self, robot_id: int, gripper_joints: List[int] = None):
        """
        初始化夹爪控制器
        
        Args:
            robot_id: 机器人ID
            gripper_joints: 夹爪关节索引 (默认为Panda夹爪)
        """
        self.robot_id = robot_id
        self.gripper_joints = gripper_joints if gripper_joints else [9, 10]
        
        # 夹爪参数 (Panda默认值)
        self.open_position = 0.04
        self.close_position = 0.01
        self.max_force = 20.0
        self.grasp_force = 5.0
    
    def open(self, width: float = None):
        """打开夹爪"""
        target_pos = width if width else self.open_position
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id, joint,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=self.max_force
            )
    
    def close(self, force: float = None):
        """关闭夹爪"""
        grasp_force = force if force else self.grasp_force
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id, joint,
                p.POSITION_CONTROL,
                targetPosition=self.close_position,
                force=grasp_force
            )
    
    def set_width(self, width: float, force: float = None):
        """设置夹爪宽度"""
        f = force if force else self.max_force
        for joint in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id, joint,
                p.POSITION_CONTROL,
                targetPosition=width / 2,  # 每个手指移动一半
                force=f
            )
    
    def get_width(self) -> float:
        """获取当前夹爪宽度"""
        widths = [p.getJointState(self.robot_id, j)[0] for j in self.gripper_joints]
        return sum(widths)
    
    def is_gripping(self, threshold: float = 0.005) -> bool:
        """检查是否抓住物体"""
        width = self.get_width()
        return width > threshold and width < self.open_position * 0.9


class RobotArmController:
    """
    机器人手臂控制器
    
    封装逆运动学和关节控制
    """
    
    def __init__(self, robot_id: int, ee_link: int = 11, n_joints: int = 7):
        """
        初始化机械臂控制器
        
        Args:
            robot_id: 机器人ID
            ee_link: 末端执行器链接索引
            n_joints: 关节数量
        """
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.n_joints = n_joints
        
        # 获取关节限制
        self._get_joint_limits()
        
        # 默认姿态
        self.home_pose = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
        
        # 控制参数
        self.joint_forces = [87.0] * n_joints
    
    def _get_joint_limits(self):
        """获取关节限制"""
        joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self.n_joints)]
        self.lower_limits = [info[8] for info in joint_infos]
        self.upper_limits = [info[9] for info in joint_infos]
        self.joint_ranges = [self.upper_limits[i] - self.lower_limits[i] 
                            for i in range(self.n_joints)]
        self.rest_poses = [0.0] * self.n_joints
    
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取末端执行器位姿"""
        state = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        pos = np.array(state[4])
        orn = np.array(state[5])
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos, orn, rot
    
    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节角度"""
        return np.array([p.getJointState(self.robot_id, i)[0] 
                        for i in range(self.n_joints)])
    
    def compute_ik(self, target_pos: np.ndarray, 
                   target_orn: np.ndarray = None) -> np.ndarray:
        """
        计算逆运动学
        
        Args:
            target_pos: 目标位置
            target_orn: 目标姿态 (四元数)
            
        Returns:
            关节角度
        """
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        
        joint_angles = p.calculateInverseKinematics(
            self.robot_id, self.ee_link,
            target_pos, target_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
            residualThreshold=1e-4
        )[:self.n_joints]
        
        return np.array(joint_angles)
    
    def set_joint_positions(self, joint_positions: np.ndarray, 
                           velocities: np.ndarray = None):
        """设置关节位置"""
        if velocities is not None:
            p.setJointMotorControlArray(
                self.robot_id,
                range(self.n_joints),
                p.POSITION_CONTROL,
                targetPositions=joint_positions,
                targetVelocities=velocities,
                forces=self.joint_forces
            )
        else:
            p.setJointMotorControlArray(
                self.robot_id,
                range(self.n_joints),
                p.POSITION_CONTROL,
                targetPositions=joint_positions,
                forces=self.joint_forces
            )
    
    def move_to_pose(self, target_pos: np.ndarray, target_orn: np.ndarray = None):
        """移动到目标位姿"""
        joint_angles = self.compute_ik(target_pos, target_orn)
        self.set_joint_positions(joint_angles)
    
    def move_to_home(self):
        """移动到初始位置"""
        self.set_joint_positions(self.home_pose)
    
    def reset_to_home(self):
        """直接重置到初始位置 (无动画)"""
        for i, q in enumerate(self.home_pose):
            p.resetJointState(self.robot_id, i, q)


class GraspStateMachine:
    """
    抓取状态机
    
    管理抓取任务的完整流程
    """
    
    def __init__(self, arm_controller: RobotArmController,
                 gripper_controller: GripperController,
                 config: ControlConfig = None):
        """
        初始化抓取状态机
        
        Args:
            arm_controller: 机械臂控制器
            gripper_controller: 夹爪控制器
            config: 控制配置
        """
        self.arm = arm_controller
        self.gripper = gripper_controller
        self.config = config if config else ControlConfig()
        
        # 状态
        self.state = GraspState.IDLE
        self.previous_state = None
        
        # 目标
        self.target_position = None
        self.target_orientation = None
        self.place_position = None
        self.place_orientation = None
        
        # PBVS控制器
        self.pbvs = PBVSController(self.config)
        
        # 约束ID (用于固定物体到末端)
        self.grasp_constraint_id = None
        self.grasped_object_id = None
        
        # 状态计时器
        self.state_start_time = None
        self.state_timeout = 5.0  # 状态超时时间 (秒)
        
        # 回调函数
        self.on_state_change: Optional[Callable] = None
    
    def set_target(self, position: np.ndarray, orientation: np.ndarray = None,
                   object_id: int = None):
        """设置抓取目标"""
        self.target_position = np.array(position)
        if orientation is not None:
            self.target_orientation = np.array(orientation)
        else:
            self.target_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        self.grasped_object_id = object_id
    
    def set_place_target(self, position: np.ndarray, orientation: np.ndarray = None):
        """设置放置目标"""
        self.place_position = np.array(position)
        if orientation is not None:
            self.place_orientation = np.array(orientation)
        else:
            self.place_orientation = self.target_orientation
    
    def start(self):
        """开始抓取任务"""
        if self.target_position is None:
            print("[StateMachine] Error: No target set!")
            self.state = GraspState.ERROR
            return
        
        self._transition_to(GraspState.APPROACHING)
    
    def update(self, detected_position: np.ndarray = None) -> GraspState:
        """
        更新状态机
        
        Args:
            detected_position: 感知模块检测到的目标位置 (可选)
            
        Returns:
            当前状态
        """
        # 更新目标位置 (如果检测到且在搜索/接近阶段)
        if detected_position is not None and self.state in [
            GraspState.SEARCHING, GraspState.APPROACHING
        ]:
            self.target_position = detected_position
        
        # 获取当前末端位姿
        ee_pos, ee_orn, _ = self.arm.get_ee_pose()
        
        # 状态超时检查
        if self._state_elapsed() > self.state_timeout and self.state not in [
            GraspState.IDLE, GraspState.COMPLETED, GraspState.ERROR
        ]:
            print(f"[StateMachine] Timeout in state {self.state.name}, forcing transition")
            # 超时后强制进入下一状态
            self._handle_timeout()
        
        # 状态处理
        if self.state == GraspState.IDLE:
            pass
        
        elif self.state == GraspState.SEARCHING:
            # 搜索状态: 等待目标检测
            if self.target_position is not None:
                self._transition_to(GraspState.APPROACHING)
        
        elif self.state == GraspState.APPROACHING:
            # 接近状态: 移动到目标上方
            approach_pos = self.target_position.copy()
            approach_pos[2] += self.config.approach_height
            
            # 直接移动到目标位置
            self.arm.move_to_pose(approach_pos, self.target_orientation)
            
            # 检查是否到达 (使用更宽松的阈值)
            error = np.linalg.norm(approach_pos - ee_pos)
            threshold = self.config.servo_threshold * 2
            
            # 调试输出
            if self._state_elapsed() > 1.0 and int(self._state_elapsed() * 10) % 10 == 0:
                print(f"  [APPROACHING] Error={error:.4f}m, Threshold={threshold:.4f}m, EE={ee_pos[:2]}, Target={approach_pos[:2]}")
            
            if error < threshold:
                self._transition_to(GraspState.PRE_GRASP)
        
        elif self.state == GraspState.PRE_GRASP:
            # 预抓取: 打开夹爪，保持位置
            approach_pos = self.target_position.copy()
            approach_pos[2] += self.config.approach_height
            self.arm.move_to_pose(approach_pos, self.target_orientation)
            self.gripper.open()
            
            if self._state_elapsed() > 0.5:
                self._transition_to(GraspState.DESCENDING)
        
        elif self.state == GraspState.DESCENDING:
            # 下降到抓取位置
            grasp_pos = self.target_position.copy()
            grasp_pos[2] += self.config.grasp_height_offset
            
            self.arm.move_to_pose(grasp_pos, self.target_orientation)
            
            error = np.linalg.norm(grasp_pos - ee_pos)
            threshold = self.config.servo_threshold * 2
            
            # 调试输出
            if self._state_elapsed() > 1.0 and int(self._state_elapsed() * 10) % 10 == 0:
                print(f"  [DESCENDING] Error={error:.4f}m, Threshold={threshold:.4f}m")
            
            if error < threshold:
                self._transition_to(GraspState.GRASPING)
        
        elif self.state == GraspState.GRASPING:
            # 保持位置并关闭夹爪
            grasp_pos = self.target_position.copy()
            grasp_pos[2] += self.config.grasp_height_offset
            self.arm.move_to_pose(grasp_pos, self.target_orientation)
            
            self.gripper.close()
            
            # 创建约束 (模拟抓取)
            if self.grasp_constraint_id is None and self.grasped_object_id is not None:
                self._create_grasp_constraint()
            
            if self._state_elapsed() > self.config.grasp_settling_time:
                self._transition_to(GraspState.LIFTING)
        
        elif self.state == GraspState.LIFTING:
            # 提升物体
            lift_pos = self.target_position.copy()
            lift_pos[2] += self.config.lift_height + self.config.approach_height
            
            self.arm.move_to_pose(lift_pos, self.target_orientation)
            
            error = np.linalg.norm(lift_pos - ee_pos)
            if error < self.config.servo_threshold * 2:
                if self.place_position is not None:
                    self._transition_to(GraspState.TRANSPORTING)
                else:
                    self._transition_to(GraspState.COMPLETED)
        
        elif self.state == GraspState.TRANSPORTING:
            # 运输到放置位置上方
            transport_pos = self.place_position.copy()
            transport_pos[2] += self.config.approach_height
            
            self.arm.move_to_pose(transport_pos, self.place_orientation)
            
            error = np.linalg.norm(transport_pos - ee_pos)
            if error < self.config.servo_threshold * 2:
                self._transition_to(GraspState.PLACING)
        
        elif self.state == GraspState.PLACING:
            # 放置物体
            self.arm.move_to_pose(self.place_position, self.place_orientation)
            
            error = np.linalg.norm(self.place_position - ee_pos)
            if error < self.config.servo_threshold * 2:
                self._transition_to(GraspState.RELEASING)
        
        elif self.state == GraspState.RELEASING:
            # 保持位置并释放物体
            self.arm.move_to_pose(self.place_position, self.place_orientation)
            self._remove_grasp_constraint()
            self.gripper.open()
            
            if self._state_elapsed() > 0.5:
                self._transition_to(GraspState.RETREATING)
        
        elif self.state == GraspState.RETREATING:
            # 撤退
            retreat_pos = self.place_position.copy()
            retreat_pos[2] += self.config.approach_height
            
            self.arm.move_to_pose(retreat_pos, self.place_orientation)
            
            error = np.linalg.norm(retreat_pos - ee_pos)
            if error < self.config.servo_threshold * 2:
                self._transition_to(GraspState.COMPLETED)
        
        elif self.state == GraspState.COMPLETED:
            pass
        
        elif self.state == GraspState.ERROR:
            pass
        
        return self.state
    
    def _handle_timeout(self):
        """处理状态超时"""
        # 根据当前状态决定下一步动作
        state_transitions = {
            GraspState.APPROACHING: GraspState.PRE_GRASP,
            GraspState.PRE_GRASP: GraspState.DESCENDING,
            GraspState.DESCENDING: GraspState.GRASPING,
            GraspState.GRASPING: GraspState.LIFTING,
            GraspState.LIFTING: GraspState.TRANSPORTING if self.place_position is not None else GraspState.COMPLETED,
            GraspState.TRANSPORTING: GraspState.PLACING,
            GraspState.PLACING: GraspState.RELEASING,
            GraspState.RELEASING: GraspState.RETREATING,
            GraspState.RETREATING: GraspState.COMPLETED,
        }
        
        if self.state in state_transitions:
            self._transition_to(state_transitions[self.state])
    
    def _transition_to(self, new_state: GraspState):
        """状态转换"""
        self.previous_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        self.pbvs.reset()
        
        print(f"[StateMachine] {self.previous_state.name} -> {new_state.name}")
        
        if self.on_state_change:
            self.on_state_change(self.previous_state, new_state)
    
    def _state_elapsed(self) -> float:
        """获取当前状态持续时间"""
        if self.state_start_time is None:
            return 0.0
        return time.time() - self.state_start_time
    
    def _create_grasp_constraint(self):
        """创建抓取约束"""
        if self.grasped_object_id is None:
            return
        
        ee_state = p.getLinkState(self.arm.robot_id, self.arm.ee_link)
        ee_orn = ee_state[1]
        
        self.grasp_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.arm.robot_id,
            parentLinkIndex=self.arm.ee_link,
            childBodyUniqueId=self.grasped_object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=ee_orn,
            childFrameOrientation=[0, 0, 0, 1]
        )
    
    def _remove_grasp_constraint(self):
        """移除抓取约束"""
        if self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = None
    
    def reset(self):
        """重置状态机"""
        self._remove_grasp_constraint()
        self.state = GraspState.IDLE
        self.previous_state = None
        self.target_position = None
        self.target_orientation = None
        self.place_position = None
        self.place_orientation = None
        self.grasped_object_id = None
        self.state_start_time = None
        self.pbvs.reset()
    
    def is_completed(self) -> bool:
        """检查是否完成"""
        return self.state == GraspState.COMPLETED
    
    def is_error(self) -> bool:
        """检查是否出错"""
        return self.state == GraspState.ERROR


class VisualServoController:
    """
    视觉伺服控制器
    
    整合感知和控制，实现闭环视觉伺服
    """
    
    def __init__(self, arm_controller: RobotArmController,
                 gripper_controller: GripperController,
                 config: ControlConfig = None):
        """
        初始化视觉伺服控制器
        
        Args:
            arm_controller: 机械臂控制器
            gripper_controller: 夹爪控制器
            config: 控制配置
        """
        self.arm = arm_controller
        self.gripper = gripper_controller
        self.config = config if config else ControlConfig()
        
        # PBVS控制器
        self.pbvs = PBVSController(self.config)
        
        # 目标位置 (视觉检测更新)
        self.target_position = None
        self.target_orientation = None
        
        # 跟踪误差记录
        self.position_errors = []
        self.timestamps = []
    
    def update_target_from_perception(self, target_position: np.ndarray,
                                      target_orientation: np.ndarray = None):
        """从感知模块更新目标位置"""
        self.target_position = target_position
        if target_orientation is not None:
            self.target_orientation = target_orientation
    
    def servo_step(self, dt: float = 0.01) -> Tuple[bool, float]:
        """
        执行一步视觉伺服
        
        Returns:
            reached: 是否到达目标
            error: 当前位置误差
        """
        if self.target_position is None:
            return False, float('inf')
        
        ee_pos, ee_orn, _ = self.arm.get_ee_pose()
        
        self.pbvs.set_target(self.target_position, self.target_orientation)
        target_pos, target_orn, reached = self.pbvs.compute_target_pose(ee_pos, dt)
        
        self.arm.move_to_pose(target_pos, target_orn)
        
        error = np.linalg.norm(self.target_position - ee_pos)
        
        # 记录误差
        self.position_errors.append(error)
        self.timestamps.append(time.time())
        
        return reached, error
    
    def get_tracking_errors(self) -> Tuple[List[float], List[float]]:
        """获取跟踪误差历史"""
        return self.timestamps, self.position_errors
    
    def reset(self):
        """重置控制器"""
        self.target_position = None
        self.target_orientation = None
        self.pbvs.reset()
        self.position_errors = []
        self.timestamps = []

