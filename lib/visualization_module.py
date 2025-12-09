import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import time
import threading


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 窗口设置
    camera_window_name: str = "Camera View"
    dashboard_window_name: str = "Visual Servo Dashboard"
    
    # 图像显示
    camera_display_width: int = 640
    camera_display_height: int = 480
    
    # 误差曲线
    error_history_length: int = 500
    error_plot_update_interval: float = 0.1  # 秒
    
    # 颜色主题
    primary_color: Tuple[int, int, int] = (0, 200, 100)    # BGR
    secondary_color: Tuple[int, int, int] = (255, 165, 0)  # BGR
    error_color: Tuple[int, int, int] = (0, 0, 255)        # BGR
    success_color: Tuple[int, int, int] = (0, 255, 0)      # BGR
    
    # 字体
    font_scale: float = 0.6
    font_thickness: int = 2


class CameraVisualizer:
    """
    相机图像可视化器
    
    显示相机图像和检测结果
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """初始化相机可视化器"""
        self.config = config if config else VisualizationConfig()
        self.window_created = False
    
    def show(self, image: np.ndarray, detection_info: Dict = None,
             state_info: Dict = None) -> np.ndarray:
        """
        显示相机图像
        
        Args:
            image: RGB图像
            detection_info: 检测信息 (可选)
            state_info: 状态信息 (可选)
            
        Returns:
            带标注的图像
        """
        # 转换为BGR (OpenCV格式)
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 调整大小
        display_image = cv2.resize(
            display_image,
            (self.config.camera_display_width, self.config.camera_display_height)
        )
        
        # 添加状态信息面板
        if state_info:
            display_image = self._draw_state_panel(display_image, state_info)
        
        # 添加检测信息
        if detection_info:
            display_image = self._draw_detection_info(display_image, detection_info)
        
        # 添加边框和标题
        display_image = self._draw_frame(display_image)
        
        # 显示
        if not self.window_created:
            cv2.namedWindow(self.config.camera_window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
        
        cv2.imshow(self.config.camera_window_name, display_image)
        cv2.waitKey(1)
        
        return display_image
    
    def _draw_state_panel(self, image: np.ndarray, state_info: Dict) -> np.ndarray:
        """绘制状态信息面板"""
        h, w = image.shape[:2]
        
        # 半透明背景
        overlay = image.copy()
        panel_height = 80
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (30, 30, 30), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # 状态文本
        y_offset = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 当前状态
        state = state_info.get('state', 'UNKNOWN')
        state_color = self.config.success_color if state == 'COMPLETED' else \
                     self.config.error_color if state == 'ERROR' else \
                     self.config.primary_color
        cv2.putText(image, f"State: {state}", (10, y_offset),
                   font, self.config.font_scale, state_color, self.config.font_thickness)
        
        # 目标信息
        y_offset += 25
        if 'target_id' in state_info:
            cv2.putText(image, f"Target ID: {state_info['target_id']}", (10, y_offset),
                       font, self.config.font_scale, self.config.secondary_color, 1)
        
        # 误差信息
        if 'error' in state_info:
            error = state_info['error']
            error_text = f"Error: {error:.4f} m"
            cv2.putText(image, error_text, (w - 180, y_offset),
                       font, self.config.font_scale, 
                       self.config.success_color if error < 0.02 else self.config.error_color, 1)
        
        # 时间戳
        y_offset += 25
        timestamp = state_info.get('timestamp', time.time())
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        cv2.putText(image, f"Time: {time_str}", (10, y_offset),
                   font, 0.5, (200, 200, 200), 1)
        
        return image
    
    def _draw_detection_info(self, image: np.ndarray, detection_info: Dict) -> np.ndarray:
        """绘制检测信息"""
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 检测状态
        detected = detection_info.get('detected', False)
        status_text = "TARGET LOCKED" if detected else "SEARCHING..."
        status_color = self.config.success_color if detected else self.config.secondary_color
        
        # 在底部显示
        cv2.putText(image, status_text, (w//2 - 80, h - 20),
                   font, self.config.font_scale, status_color, self.config.font_thickness)
        
        # 位置信息
        if detected and 'position' in detection_info:
            pos = detection_info['position']
            pos_text = f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            cv2.putText(image, pos_text, (10, h - 20),
                       font, 0.5, self.config.primary_color, 1)
        
        return image
    
    def _draw_frame(self, image: np.ndarray) -> np.ndarray:
        """绘制边框"""
        h, w = image.shape[:2]
        
        # 边框
        cv2.rectangle(image, (0, 0), (w-1, h-1), self.config.primary_color, 2)
        
        # 角落装饰
        corner_length = 30
        corners = [
            ((0, 0), (corner_length, 0), (0, corner_length)),
            ((w-1, 0), (w-1-corner_length, 0), (w-1, corner_length)),
            ((0, h-1), (corner_length, h-1), (0, h-1-corner_length)),
            ((w-1, h-1), (w-1-corner_length, h-1), (w-1, h-1-corner_length))
        ]
        
        for corner, p1, p2 in corners:
            cv2.line(image, corner, p1, self.config.primary_color, 3)
            cv2.line(image, corner, p2, self.config.primary_color, 3)
        
        return image
    
    def close(self):
        """关闭窗口"""
        cv2.destroyWindow(self.config.camera_window_name)
        self.window_created = False


class ErrorPlotter:
    """
    误差曲线绘制器
    
    实时显示跟踪误差
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """初始化误差绘制器"""
        self.config = config if config else VisualizationConfig()
        
        # 数据缓冲
        self.position_errors = deque(maxlen=self.config.error_history_length)
        self.timestamps = deque(maxlen=self.config.error_history_length)
        self.xyz_errors = {
            'x': deque(maxlen=self.config.error_history_length),
            'y': deque(maxlen=self.config.error_history_length),
            'z': deque(maxlen=self.config.error_history_length)
        }
        
        # 绘图设置
        self.fig = None
        self.axes = None
        self.lines = {}
        self.start_time = None
        
        # 更新控制
        self.last_update_time = 0
        
        # 是否已初始化
        self.initialized = False
    
    def initialize(self):
        """初始化绘图"""
        plt.ion()  # 交互模式
        
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.suptitle('Visual Servo Tracking Error', fontsize=14, fontweight='bold')
        
        # 总误差图
        self.axes[0].set_xlabel('Time (s)')
        self.axes[0].set_ylabel('Position Error (m)')
        self.axes[0].set_title('Total Position Error')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_ylim(0, 0.2)
        self.lines['total'], = self.axes[0].plot([], [], 'b-', linewidth=2, label='Error')
        self.lines['threshold'] = self.axes[0].axhline(y=0.01, color='g', linestyle='--', 
                                                        label='Threshold')
        self.axes[0].legend(loc='upper right')
        
        # XYZ分量误差图
        self.axes[1].set_xlabel('Time (s)')
        self.axes[1].set_ylabel('Error (m)')
        self.axes[1].set_title('XYZ Component Errors')
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].set_ylim(-0.1, 0.1)
        self.lines['x'], = self.axes[1].plot([], [], 'r-', linewidth=1.5, label='X')
        self.lines['y'], = self.axes[1].plot([], [], 'g-', linewidth=1.5, label='Y')
        self.lines['z'], = self.axes[1].plot([], [], 'b-', linewidth=1.5, label='Z')
        self.axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        
        self.start_time = time.time()
        self.initialized = True
    
    def add_error(self, total_error: float, xyz_error: np.ndarray = None):
        """
        添加误差数据
        
        Args:
            total_error: 总位置误差
            xyz_error: XYZ分量误差 [ex, ey, ez]
        """
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
        
        rel_time = current_time - self.start_time
        
        self.timestamps.append(rel_time)
        self.position_errors.append(total_error)
        
        if xyz_error is not None:
            self.xyz_errors['x'].append(xyz_error[0])
            self.xyz_errors['y'].append(xyz_error[1])
            self.xyz_errors['z'].append(xyz_error[2])
    
    def update(self):
        """更新绘图"""
        if not self.initialized:
            self.initialize()
        
        current_time = time.time()
        if current_time - self.last_update_time < self.config.error_plot_update_interval:
            return
        
        self.last_update_time = current_time
        
        if len(self.timestamps) < 2:
            return
        
        times = list(self.timestamps)
        
        # 更新总误差曲线
        self.lines['total'].set_data(times, list(self.position_errors))
        
        # 更新XYZ分量曲线
        if len(self.xyz_errors['x']) > 0:
            self.lines['x'].set_data(times, list(self.xyz_errors['x']))
            self.lines['y'].set_data(times, list(self.xyz_errors['y']))
            self.lines['z'].set_data(times, list(self.xyz_errors['z']))
        
        # 自动调整坐标轴
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        
        # 刷新
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def save(self, filename: str = 'tracking_error.png'):
        """保存图像"""
        if self.fig:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Error plot saved to {filename}")
    
    def close(self):
        """关闭绘图"""
        if self.fig:
            plt.close(self.fig)
            self.initialized = False
    
    def reset(self):
        """重置数据"""
        self.position_errors.clear()
        self.timestamps.clear()
        for key in self.xyz_errors:
            self.xyz_errors[key].clear()
        self.start_time = None


class DashboardVisualizer:
    """
    综合仪表盘可视化器
    
    整合所有可视化功能
    """
    
    def __init__(self, config: VisualizationConfig = None, enable_error_plot: bool = False):
        """初始化仪表盘"""
        self.config = config if config else VisualizationConfig()
        self.enable_error_plot = enable_error_plot
        
        # 子可视化器
        self.camera_vis = CameraVisualizer(config)
        self.error_plotter = ErrorPlotter(config) if enable_error_plot else None
        
        # 状态
        self.is_running = False
        self.statistics = {
            'total_frames': 0,
            'detected_frames': 0,
            'min_error': float('inf'),
            'max_error': 0,
            'avg_error': 0,
            'error_sum': 0
        }
    
    def start(self):
        """启动可视化"""
        self.is_running = True
        if self.error_plotter:
            self.error_plotter.initialize()
    
    def update(self, camera_image: np.ndarray = None,
               detection_info: Dict = None,
               state_info: Dict = None,
               position_error: float = None,
               xyz_error: np.ndarray = None):
        """
        更新仪表盘
        
        Args:
            camera_image: 相机图像
            detection_info: 检测信息
            state_info: 状态信息
            position_error: 位置误差
            xyz_error: XYZ分量误差
        """
        if not self.is_running:
            return
        
        # 更新统计
        self.statistics['total_frames'] += 1
        
        if detection_info and detection_info.get('detected', False):
            self.statistics['detected_frames'] += 1
        
        if position_error is not None:
            self.statistics['min_error'] = min(self.statistics['min_error'], position_error)
            self.statistics['max_error'] = max(self.statistics['max_error'], position_error)
            self.statistics['error_sum'] += position_error
            self.statistics['avg_error'] = self.statistics['error_sum'] / self.statistics['total_frames']
            
            # 添加误差数据
            if self.error_plotter:
                self.error_plotter.add_error(position_error, xyz_error)
        
        # 添加统计信息到状态
        if state_info is None:
            state_info = {}
        state_info['error'] = position_error if position_error else 0
        
        # 更新相机视图
        if camera_image is not None:
            self.camera_vis.show(camera_image, detection_info, state_info)
        
        # 更新误差曲线
        if self.error_plotter:
            self.error_plotter.update()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        detection_rate = (self.statistics['detected_frames'] / 
                         max(self.statistics['total_frames'], 1))
        return {
            **self.statistics,
            'detection_rate': detection_rate
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("Visual Servo Statistics")
        print("="*50)
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Detection Rate: {stats['detection_rate']*100:.1f}%")
        print(f"Min Error: {stats['min_error']:.4f} m")
        print(f"Max Error: {stats['max_error']:.4f} m")
        print(f"Avg Error: {stats['avg_error']:.4f} m")
        print("="*50 + "\n")
    
    def save_results(self, prefix: str = 'visual_servo'):
        """保存结果"""
        if self.error_plotter:
            self.error_plotter.save(f'{prefix}_error.png')
        self.print_statistics()
    
    def stop(self):
        """停止可视化"""
        self.is_running = False
        self.camera_vis.close()
        if self.error_plotter:
            self.error_plotter.close()
    
    def reset(self):
        """重置可视化"""
        if self.error_plotter:
            self.error_plotter.reset()
        self.statistics = {
            'total_frames': 0,
            'detected_frames': 0,
            'min_error': float('inf'),
            'max_error': 0,
            'avg_error': 0,
            'error_sum': 0
        }


class TrajectoryVisualizer:
    """
    轨迹可视化器
    
    在PyBullet中绘制3D轨迹
    """
    
    def __init__(self, line_width: float = 2.0, 
                 lifetime: float = 0.0,
                 max_points: int = 1000):
        """
        初始化轨迹可视化器
        
        Args:
            line_width: 线宽
            lifetime: 线段存活时间 (0表示永久)
            max_points: 最大点数
        """
        import pybullet as p
        self.p = p
        
        self.line_width = line_width
        self.lifetime = lifetime
        self.max_points = max_points
        
        # 轨迹点
        self.trajectory_points: List[np.ndarray] = []
        self.target_points: List[np.ndarray] = []
        
        # 颜色
        self.actual_color = [0, 1, 0]     # 绿色: 实际轨迹
        self.target_color = [1, 0, 0]     # 红色: 目标轨迹
        self.error_color = [1, 1, 0]      # 黄色: 误差线
    
    def add_point(self, actual_pos: np.ndarray, target_pos: np.ndarray = None):
        """添加轨迹点"""
        # 绘制实际轨迹
        if len(self.trajectory_points) > 0:
            self.p.addUserDebugLine(
                self.trajectory_points[-1], actual_pos,
                self.actual_color, self.line_width, self.lifetime
            )
        self.trajectory_points.append(actual_pos.copy())
        
        # 绘制目标轨迹
        if target_pos is not None:
            if len(self.target_points) > 0:
                self.p.addUserDebugLine(
                    self.target_points[-1], target_pos,
                    self.target_color, self.line_width * 0.5, self.lifetime
                )
            self.target_points.append(target_pos.copy())
            
            # 绘制误差线
            self.p.addUserDebugLine(
                actual_pos, target_pos,
                self.error_color, 1, 0.1  # 短暂显示
            )
        
        # 限制点数
        if len(self.trajectory_points) > self.max_points:
            self.trajectory_points.pop(0)
        if len(self.target_points) > self.max_points:
            self.target_points.pop(0)
    
    def draw_coordinate_frame(self, position: np.ndarray, 
                              rotation_matrix: np.ndarray,
                              axis_length: float = 0.1):
        """绘制坐标系"""
        origin = position
        
        # X轴 - 红
        self.p.addUserDebugLine(
            origin, origin + axis_length * rotation_matrix[:, 0],
            [1, 0, 0], 2, 0.1
        )
        # Y轴 - 绿
        self.p.addUserDebugLine(
            origin, origin + axis_length * rotation_matrix[:, 1],
            [0, 1, 0], 2, 0.1
        )
        # Z轴 - 蓝
        self.p.addUserDebugLine(
            origin, origin + axis_length * rotation_matrix[:, 2],
            [0, 0, 1], 2, 0.1
        )
    
    def draw_target_marker(self, position: np.ndarray, size: float = 0.05):
        """绘制目标标记"""
        half = size / 2
        
        # 绘制十字
        self.p.addUserDebugLine(
            [position[0] - half, position[1], position[2]],
            [position[0] + half, position[1], position[2]],
            [1, 0, 0], 3, 0.5
        )
        self.p.addUserDebugLine(
            [position[0], position[1] - half, position[2]],
            [position[0], position[1] + half, position[2]],
            [0, 1, 0], 3, 0.5
        )
        self.p.addUserDebugLine(
            [position[0], position[1], position[2] - half],
            [position[0], position[1], position[2] + half],
            [0, 0, 1], 3, 0.5
        )
    
    def clear(self):
        """清除轨迹"""
        self.trajectory_points.clear()
        self.target_points.clear()
        self.p.removeAllUserDebugItems()

