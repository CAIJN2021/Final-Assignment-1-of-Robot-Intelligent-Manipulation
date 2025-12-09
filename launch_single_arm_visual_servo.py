import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
import signal
import sys

# 导入自定义模块
from lib.perception_module import PerceptionModule, TargetState
from lib.control_module import (
    RobotArmController, GripperController, GraspStateMachine,
    VisualServoController, ControlConfig, GraspState
)
from lib.visualization_module import (
    DashboardVisualizer, TrajectoryVisualizer, VisualizationConfig
)


# ============== 全局配置 ==============
class SystemConfig:
    """系统配置"""
    # 仿真参数
    sim_dt = 0.01                    # 仿真时间步长
    control_freq = 100               # 控制频率 Hz
    
    # 桌子参数
    table_position = [0.5, 0.0, 0.0]  # 桌子位置
    table_height = 0.62               # 桌面高度
    
    # 机器人参数 - 放置在桌子上
    robot_base_position = [0.2, 0.0, 0.62]  # 机器人放在桌子上
    
    # 相机参数
    camera_width = 640
    camera_height = 480
    camera_fov = 60.0
    
    # ArUco标记参数
    marker_length = 0.04             # 标记边长 (米)
    
    # 抓取目标参数
    cube_size = 0.05                   # 立方体边长 (米)
    cube_initial_offset = [0.5, 0.15, 0.0]  # 立方体相对于机器人基座的偏移 [x, y, z]
                                            # x: 前后 (0.5m前方), y: 左右 (0.15m右侧), z: 上下 (0m)
    
    # 放置位置偏移 (相对于机器人基座)
    place_offset = [0.5, -0.15, 0.0]   # 放置位置相对于机器人基座的偏移 [x, y, z]
                                        # x: 前后 (0.5m前方), y: 左右 (-0.15m左侧), z: 上下 (0m)


# ============== 仿真环境设置 ==============
class SimulationEnvironment:
    """仿真环境管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.robot_id = None
        self.table_id = None
        self.cube_id = None
        self.target_box_id = None  # 目标盒子
        
    def setup(self):
        """设置仿真环境"""
        print("[System] Initializing PyBullet simulation...")
        
        # 连接PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.config.sim_dt)
        
        # 设置相机视角 - 调整为更好的观察角度
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=135,
            cameraPitch=-25,
            cameraTargetPosition=[0.5, 0.0, 0.7]
        )
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 加载桌子
        print(f"[System] Loading table at {self.config.table_position}")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=self.config.table_position
        )
        
        # 加载机器人 - 放在桌子上
        print(f"[System] Loading robot at {self.config.robot_base_position}")
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.config.robot_base_position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        
        # 加载ArUco立方体 - 立方体偏移相对于机器人基座
        cube_world_pos = [
            self.config.robot_base_position[0] + self.config.cube_initial_offset[0],
            self.config.robot_base_position[1] + self.config.cube_initial_offset[1],
            self.config.table_height + self.config.cube_size / 2 + self.config.cube_initial_offset[2]
        ]
        
        print(f"[System] Loading cube at {cube_world_pos}")
        self.cube_id = self._load_aruco_cube(cube_world_pos)
        
        # 加载目标盒子 - 放置位置
        box_world_pos = [
            self.config.robot_base_position[0] + self.config.place_offset[0],
            self.config.robot_base_position[1] + self.config.place_offset[1],
            self.config.table_height + self.config.place_offset[2]
        ]
        print(f"[System] Loading target box at {box_world_pos}")
        self.target_box_id = self._load_target_box(box_world_pos)
        
        print(f"[System] Robot ID: {self.robot_id}")
        print(f"[System] Table ID: {self.table_id}")
        print(f"[System] Cube ID: {self.cube_id}")
        print(f"[System] Target Box ID: {self.target_box_id}")
        
        return self.robot_id, self.table_id, self.cube_id
    
    def _load_aruco_cube(self, position):
        """加载ArUco立方体"""
        scale_to_meters = self.config.cube_size / 0.05
        
        cube_id = p.loadURDF(
            "aruco_cube_description/urdf/aruco.urdf",
            position,
            globalScaling=scale_to_meters
        )
        
        # 加载纹理
        texture_id = p.loadTexture("aruco_cube_description/materials/textures/aruco_0_with_border.png")
        p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)
        
        return cube_id
    
    def _load_target_box(self, position):
        """加载目标盒子"""
        box_id = p.loadURDF(
            "target_box_description/urdf/target_box.urdf",
            position,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True  # 盒子固定不动
        )
        return box_id
    
    def get_cube_position(self) -> np.ndarray:
        """获取立方体当前位置"""
        pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        return np.array(pos)
    
    def reset_cube(self):
        """重置立方体位置"""
        cube_pos = [
            self.config.robot_base_position[0] + self.config.cube_initial_offset[0],
            self.config.robot_base_position[1] + self.config.cube_initial_offset[1],
            self.config.table_height + self.config.cube_size / 2
        ]
        p.resetBasePositionAndOrientation(
            self.cube_id, cube_pos,
            p.getQuaternionFromEuler([0, 0, 0])
        )
    
    def step(self):
        """执行一步仿真"""
        p.stepSimulation()
    
    def close(self):
        """关闭仿真"""
        p.disconnect()


# ============== 主系统类 ==============
class SingleArmVisualServoSystem:
    """
    单臂视觉伺服抓取系统
    
    集成感知、控制、可视化模块的完整系统
    """
    
    def __init__(self, config: SystemConfig = None):
        """初始化系统"""
        self.config = config if config else SystemConfig()
        
        # 环境
        self.env = SimulationEnvironment(self.config)
        
        # 模块 (稍后初始化)
        self.perception = None
        self.arm_controller = None
        self.gripper_controller = None
        self.state_machine = None
        self.servo_controller = None
        self.visualizer = None
        self.trajectory_vis = None
        
        # 状态
        self.is_running = False
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n[System] Received interrupt signal, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self):
        """初始化所有模块"""
        print("\n" + "="*60)
        print("  Single Arm Visual Servo Grasping System")
        print("="*60 + "\n")
        
        # 设置仿真环境
        robot_id, table_id, cube_id = self.env.setup()
        
        # 初始化机械臂控制器
        print("[System] Initializing arm controller...")
        self.arm_controller = RobotArmController(robot_id, ee_link=11, n_joints=7)
        
        # 设置初始姿态
        home_pose = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]
        self.arm_controller.home_pose = home_pose
        self.arm_controller.reset_to_home()
        
        # 初始化夹爪控制器
        print("[System] Initializing gripper controller...")
        self.gripper_controller = GripperController(robot_id)
        self.gripper_controller.open()
        
        # 初始化感知模块
        print("[System] Initializing perception module...")
        self.perception = PerceptionModule(
            robot_id, ee_link=11,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            camera_fov=self.config.camera_fov,
            marker_length=self.config.marker_length
        )
        
        # 初始化控制配置
        control_config = ControlConfig(
            servo_gain_p=1.5,
            servo_gain_d=0.1,
            servo_max_vel=0.3,
            servo_threshold=0.02,  # 更宽松的阈值
            approach_height=0.15,
            grasp_height_offset=0.02,  # 抓取时稍微高一点
            lift_height=0.15,
            grasp_settling_time=0.8,  # 夹爪稳定时间
            workspace_min=np.array([0.2, -0.6, self.config.table_height]),
            workspace_max=np.array([1.0, 0.6, self.config.table_height + 0.6])
        )
        
        # 初始化视觉伺服控制器
        print("[System] Initializing visual servo controller...")
        self.servo_controller = VisualServoController(
            self.arm_controller, self.gripper_controller, control_config
        )
        
        # 初始化状态机
        print("[System] Initializing grasp state machine...")
        self.state_machine = GraspStateMachine(
            self.arm_controller, self.gripper_controller, control_config
        )
        
        # 初始化可视化
        print("[System] Initializing visualization...")
        vis_config = VisualizationConfig(
            camera_display_width=640,
            camera_display_height=480
        )
        self.visualizer = DashboardVisualizer(vis_config)
        self.trajectory_vis = TrajectoryVisualizer()
        
        # 稳定仿真
        print("[System] Stabilizing simulation...")
        for _ in range(200):
            self.env.step()
            time.sleep(0.001)
        
        print("[System] Initialization complete!\n")
        self.is_running = True
    
    def run_visual_servo_demo(self):
        """
        运行视觉伺服跟踪演示
        
        机器人持续跟踪ArUco标记
        """
        print("\n" + "-"*50)
        print("  Visual Servo Tracking Demo")
        print("-"*50)
        print("Press 'q' to quit, 'r' to reset, 's' to save\n")
        
        self.visualizer.start()
        
        # 移动到观察位置 - 在立方体上方
        cube_pos = self.env.get_cube_position()
        observe_pos = np.array([cube_pos[0] - 0.1, cube_pos[1], cube_pos[2] + 0.25])
        observe_orn = p.getQuaternionFromEuler([0, np.pi * 0.85, 0])
        
        print(f"[Demo] Cube position: {cube_pos}")
        print("[Demo] Moving to observation position...")
        self._move_to_position(observe_pos, observe_orn, duration=2.0)
        
        print("[Demo] Starting visual servo tracking...\n")
        
        iteration = 0
        
        while self.is_running:
            iteration += 1
            
            # 更新感知
            target_state = self.perception.update()
            
            # 获取可视化图像
            vis_image = self.perception.get_visualization()
            
            # 当前末端位置
            ee_pos, ee_orn, ee_rot = self.arm_controller.get_ee_pose()
            
            # 计算误差
            position_error = None
            xyz_error = None
            
            if target_state.detected and target_state.position_base is not None:
                # 更新目标 (添加偏移以保持安全距离)
                target_pos = target_state.position_base.copy()
                target_pos[2] += 0.12  # 保持在目标上方
                
                self.servo_controller.update_target_from_perception(
                    target_pos, observe_orn
                )
                
                # 执行伺服步骤
                reached, position_error = self.servo_controller.servo_step(self.config.sim_dt)
                
                # 计算XYZ误差
                xyz_error = target_pos - ee_pos
                
                # 绘制目标标记
                self.trajectory_vis.draw_target_marker(target_state.position_base)
                
                if iteration % 50 == 0:
                    print(f"[Tracking] Target: {target_state.position_base}, Error: {position_error:.4f}m")
            
            
            # 更新可视化
            detection_info = {
                'detected': target_state.detected,
                'position': target_state.position_base
            }
            state_info = {
                'state': 'TRACKING' if target_state.detected else 'SEARCHING',
                'target_id': target_state.marker_id,
                'timestamp': time.time()
            }
            
            self.visualizer.update(
                vis_image, detection_info, state_info,
                position_error, xyz_error
            )
            
            # 仿真步进
            self.env.step()
            
            # 检查键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_demo()
            elif key == ord('s'):
                self.visualizer.save_results('visual_servo')
            
            # 控制频率
            time.sleep(self.config.sim_dt)
        
        # 显示统计
        self.visualizer.print_statistics()
    
    def run_grasp_demo(self):
        """
        运行抓取演示
        
        使用状态机执行完整的抓取流程
        """
        print("\n" + "-"*50)
        print("  Visual Servo Grasp Demo")
        print("-"*50)
        print("Press 'q' to quit\n")
        
        self.visualizer.start()
        
        # 移动到初始观察位置
        cube_pos = self.env.get_cube_position()
        observe_pos = np.array([cube_pos[0] - 0.1, cube_pos[1], cube_pos[2] + 0.25])
        observe_orn = p.getQuaternionFromEuler([0, np.pi * 0.85, 0])
        
        print(f"[Demo] Cube position: {cube_pos}")
        print("[Demo] Moving to observation position...")
        self._move_to_position(observe_pos, observe_orn, duration=2.0)
        
        # 等待检测目标
        print("[Demo] Searching for target...")
        target_found = False
        search_timeout = 5.0
        search_start = time.time()
        
        while not target_found and (time.time() - search_start) < search_timeout:
            target_state = self.perception.update()
            vis_image = self.perception.get_visualization()
            
            if target_state.detected:
                target_found = True
                print(f"[Demo] Target found! ID: {target_state.marker_id}")
                print(f"[Demo] Position: {target_state.position_base}")
            
            # 更新可视化
            detection_info = {'detected': target_state.detected}
            state_info = {'state': 'SEARCHING'}
            self.visualizer.update(vis_image, detection_info, state_info)
            
            self.env.step()
            time.sleep(self.config.sim_dt)
        
        if not target_found:
            print("[Demo] Target not found, aborting.")
            return
        
        # 设置抓取目标
        grasp_target = target_state.position_base.copy()
        grasp_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        
        # 设置放置位置（盒子内部）
        # 盒子底板厚度0.005m，方块放在盒子内部中心
        box_bottom_height = 0.005
        place_pos = np.array([
            self.config.robot_base_position[0] + self.config.place_offset[0],
            self.config.robot_base_position[1] + self.config.place_offset[1],
            self.config.table_height + box_bottom_height + self.config.cube_size / 2 + 0.01  # 稍高一点释放
        ])
        
        print(f"[Demo] Grasp target: {grasp_target}")
        print(f"[Demo] Place position: {place_pos}")
        
        self.state_machine.set_target(grasp_target, grasp_orn, self.env.cube_id)
        self.state_machine.set_place_target(place_pos, grasp_orn)
        self.state_machine.start()
        
        print("[Demo] Starting grasp sequence...\n")
        
        # 执行状态机
        iteration = 0
        last_state = None
        while self.is_running and not self.state_machine.is_completed():
            iteration += 1
            
            # 更新感知
            target_state = self.perception.update()
            vis_image = self.perception.get_visualization()
            
            # 获取当前位置
            ee_pos, _, ee_rot = self.arm_controller.get_ee_pose()
            
            # 更新状态机
            detected_pos = target_state.position_base if target_state.detected else None
            current_state = self.state_machine.update(detected_pos)
            
            # 状态改变时打印信息
            if current_state != last_state:
                print(f"[StateMachine] State: {current_state.name}")
                last_state = current_state
            
            # 计算误差
            position_error = None
            xyz_error = None
            if self.state_machine.target_position is not None:
                error_vec = self.state_machine.target_position - ee_pos
                position_error = np.linalg.norm(error_vec)
                xyz_error = error_vec
                
                # 每50次迭代打印一次状态
                if iteration % 50 == 0:
                    print(f"[Debug] State={current_state.name}, Error={position_error:.4f}m, EE={ee_pos}, Target={self.state_machine.target_position}")
            
            
            # 更新可视化
            detection_info = {
                'detected': target_state.detected,
                'position': target_state.position_base
            }
            state_info = {
                'state': current_state.name,
                'target_id': target_state.marker_id,
                'timestamp': time.time()
            }
            
            self.visualizer.update(
                vis_image, detection_info, state_info,
                position_error, xyz_error
            )
            
            # 仿真步进
            self.env.step()
            
            # 检查键盘
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("[Demo] Reset requested...")
                self._reset_demo()
                return  # 退出当前抓取序列
            
            time.sleep(self.config.sim_dt)
        
        if self.state_machine.is_completed():
            print("\n[Demo] ✅ Grasp sequence completed successfully!")
            print("[Demo] Returning to home position...")
            
            # 返回初始位置
            self._return_to_home()
        
        # 显示统计
        self.visualizer.print_statistics()
    
    def _move_to_position(self, target_pos: np.ndarray, 
                          target_orn: np.ndarray,
                          duration: float = 2.0):
        """平滑移动到目标位置"""
        ee_pos, _, _ = self.arm_controller.get_ee_pose()
        
        steps = int(duration / self.config.sim_dt)
        
        for i in range(steps):
            t = (i + 1) / steps
            # 使用平滑插值 (minimum jerk)
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            
            current_target = ee_pos + s * (target_pos - ee_pos)
            self.arm_controller.move_to_pose(current_target, target_orn)
            
            self.env.step()
            time.sleep(self.config.sim_dt * 0.5)
    
    def _return_to_home(self):
        """返回初始位置"""
        # 计算home位置对应的末端执行器位置
        # 先设置关节到home pose以获取末端位置
        for i, q in enumerate(self.arm_controller.home_pose):
            p.resetJointState(self.arm_controller.robot_id, i, q)
        
        # 获取home位置的末端执行器位姿
        self.env.step()
        home_ee_pos, home_ee_orn, _ = self.arm_controller.get_ee_pose()
        
        # 恢复当前关节状态（因为我们只是用来计算home位置）
        current_joints = self.arm_controller.get_joint_positions()
        for i, q in enumerate(current_joints):
            p.resetJointState(self.arm_controller.robot_id, i, q)
        
        # 平滑移动到home位置
        print(f"  Moving to home position: {home_ee_pos}")
        self._move_to_position(home_ee_pos, home_ee_orn, duration=3.0)
        
        # 最后使用关节控制确保精确到达
        print("  Fine-tuning to home pose...")
        for _ in range(100):
            self.arm_controller.set_joint_positions(self.arm_controller.home_pose)
            self.env.step()
            time.sleep(self.config.sim_dt * 0.5)
        
        print("  ✅ Returned to home position")
    
    def _reset_demo(self):
        """重置演示"""
        print("[System] Resetting...")
        
        # 重置机器人
        self.arm_controller.reset_to_home()
        self.gripper_controller.open()
        
        # 重置立方体位置
        self.env.reset_cube()
        
        # 重置状态机
        if self.state_machine:
            self.state_machine.reset()
        
        # 重置感知
        self.perception.reset_filter()
        
        # 重置可视化
        self.visualizer.reset()
        p.removeAllUserDebugItems()  # 清除所有调试绘制
        
        # 稳定仿真
        for _ in range(100):
            self.env.step()
        
        print("[System] Reset complete.\n")
    
    def shutdown(self):
        """关闭系统"""
        print("[System] Shutting down...")
        
        self.is_running = False
        
        # 保存结果
        if self.visualizer:
            self.visualizer.save_results('visual_servo_final')
            self.visualizer.stop()
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        # 关闭仿真
        self.env.close()
        
        print("[System] Shutdown complete.")
    
    def run(self, mode: str = 'tracking'):
        """
        运行系统
        
        Args:
            mode: 运行模式
                - 'tracking': 视觉伺服跟踪演示
                - 'grasp': 抓取演示
        """
        try:
            self.initialize()
            
            if mode == 'tracking':
                self.run_visual_servo_demo()
            elif mode == 'grasp':
                self.run_grasp_demo()
            else:
                print(f"Unknown mode: {mode}")
                print("Available modes: 'tracking', 'grasp'")
            
        except Exception as e:
            print(f"[System] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()


# ============== 主函数 ==============
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Single Arm Visual Servo Grasping System'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='tracking',
        choices=['tracking', 'grasp'],
        help='Running mode: tracking (visual servo) or grasp (pick-and-place)'
    )
    
    args = parser.parse_args()
    
    # 创建并运行系统
    system = SingleArmVisualServoSystem()
    system.run(mode=args.mode)


if __name__ == "__main__":
    main()
