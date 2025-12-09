import cv2
import cv2.aruco as aruco
import numpy as np
import pybullet as p
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """目标类型枚举"""
    ARUCO_MARKER = "aruco_marker"
    COLOR_OBJECT = "color_object"
    UNKNOWN = "unknown"


@dataclass
class TargetState:
    """目标状态数据类"""
    detected: bool = False
    target_type: TargetType = TargetType.UNKNOWN
    position_camera: Optional[np.ndarray] = None  # 相机坐标系下的位置
    position_base: Optional[np.ndarray] = None    # 机器人基坐标系下的位置
    orientation_camera: Optional[np.ndarray] = None  # 相机坐标系下的姿态 (旋转向量)
    orientation_base: Optional[np.ndarray] = None    # 机器人基坐标系下的姿态
    marker_id: Optional[int] = None
    confidence: float = 0.0
    timestamp: float = 0.0


class CameraSimulator:
    """
    PyBullet相机仿真器
    
    从机器人末端执行器获取虚拟相机图像
    """
    
    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 60.0,
                 near_val: float = 0.01, far_val: float = 5.0):
        """
        初始化相机仿真器
        
        Args:
            width: 图像宽度
            height: 图像高度
            fov_deg: 视场角(度)
            near_val: 近裁剪面
            far_val: 远裁剪面
        """
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.near_val = near_val
        self.far_val = far_val
        
        # 计算相机内参
        fov_rad = np.deg2rad(fov_deg)
        self.fy = 0.5 * height / np.tan(fov_rad / 2)
        self.fx = self.fy
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((5,), dtype=np.float64)
        
        # 投影矩阵
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov_deg, aspect=width/height, 
            nearVal=near_val, farVal=far_val
        )
    
    def get_image_from_ee(self, robot_id: int, ee_link: int,
                          camera_offset: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从末端执行器获取相机图像
        
        Args:
            robot_id: 机器人ID
            ee_link: 末端执行器链接索引
            camera_offset: 相机相对于末端执行器的偏移 [x, y, z]
            
        Returns:
            rgb_image: RGB图像
            depth_image: 深度图像
        """
        # 获取末端执行器位姿
        state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
        ee_pos = np.array(state[4])
        ee_orn = np.array(state[5])
        ee_rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        
        # 相机位置 (默认相机位于末端执行器)
        if camera_offset is not None:
            camera_pos = ee_pos + ee_rot @ camera_offset
        else:
            camera_pos = ee_pos
        
        # 相机朝向 (沿着末端执行器的z轴)
        camera_target = camera_pos + ee_rot[:, 2]  # z轴方向
        camera_up = -ee_rot[:, 0]  # 使用负x轴作为上方向
        
        # 计算视图矩阵
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up
        )
        
        # 获取相机图像
        img_arr = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 处理RGB图像
        rgb = np.reshape(img_arr[2], (self.height, self.width, 4))
        rgb = rgb[:, :, :3].astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        
        # 处理深度图像
        depth_buffer = np.reshape(img_arr[3], (self.height, self.width))
        depth = self.far_val * self.near_val / (self.far_val - (self.far_val - self.near_val) * depth_buffer)
        
        return rgb, depth
    
    def get_ee_pose(self, robot_id: int, ee_link: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取末端执行器位姿"""
        state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
        pos = np.array(state[4])
        orn = np.array(state[5])
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos, orn, rot


class TargetDetector:
    """
    目标检测器
    
    支持ArUco标记检测和颜色物体检测
    """
    
    def __init__(self, camera: CameraSimulator, marker_length: float = 0.04,
                 aruco_dict_type: int = aruco.DICT_4X4_100):
        """
        初始化目标检测器
        
        Args:
            camera: 相机仿真器
            marker_length: ArUco标记边长(米)
            aruco_dict_type: ArUco字典类型
        """
        self.camera = camera
        self.marker_length = marker_length
        
        # ArUco检测器设置
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = aruco.DetectorParameters()
        
        # ArUco标记的3D角点坐标 (标记坐标系)
        half_len = marker_length / 2.0
        self.marker_obj_points = np.array([
            [-half_len,  half_len, 0],
            [ half_len,  half_len, 0],
            [ half_len, -half_len, 0],
            [-half_len, -half_len, 0]
        ], dtype=np.float32)
        
        # 颜色检测参数 (HSV范围)
        self.color_ranges = {
            'red': ((0, 100, 100), (10, 255, 255)),
            'green': ((35, 100, 100), (85, 255, 255)),
            'blue': ((100, 100, 100), (130, 255, 255)),
        }
    
    def detect_aruco(self, image: np.ndarray) -> List[Dict]:
        """
        检测ArUco标记
        
        Args:
            image: RGB图像
            
        Returns:
            检测结果列表, 每个结果包含:
            - marker_id: 标记ID
            - corners: 角点坐标
            - rvec: 旋转向量
            - tvec: 平移向量
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict)
        
        results = []
        if ids is not None:
            for i, marker_id in enumerate(ids):
                img_points = corners[i][0].astype(np.float32)
                
                # 使用PnP求解位姿
                success, rvec, tvec = cv2.solvePnP(
                    self.marker_obj_points,
                    img_points,
                    self.camera.camera_matrix,
                    self.camera.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    results.append({
                        'marker_id': marker_id[0],
                        'corners': corners[i],
                        'rvec': rvec,
                        'tvec': tvec,
                        'confidence': 1.0
                    })
        
        return results
    
    def detect_color_object(self, image: np.ndarray, color: str = 'red',
                           min_area: int = 500) -> List[Dict]:
        """
        检测颜色物体
        
        Args:
            image: RGB图像
            color: 目标颜色
            min_area: 最小面积阈值
            
        Returns:
            检测结果列表
        """
        if color not in self.color_ranges:
            return []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower, upper = self.color_ranges[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 计算质心
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    results.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'contour': contour,
                        'confidence': min(area / 10000, 1.0)
                    })
        
        return results
    
    def draw_detection(self, image: np.ndarray, aruco_results: List[Dict] = None,
                       color_results: List[Dict] = None) -> np.ndarray:
        """
        在图像上绘制检测结果
        """
        output = image.copy()
        
        # 绘制ArUco检测结果
        if aruco_results:
            for result in aruco_results:
                corners = result['corners']
                marker_id = result['marker_id']
                rvec = result['rvec']
                tvec = result['tvec']
                
                # 绘制标记边框
                pts = corners[0].astype(np.int32)
                cv2.polylines(output, [pts], True, (0, 255, 0), 2)
                
                # 绘制坐标轴
                self._draw_axes(output, rvec, tvec)
                
                # 显示ID和距离
                center = np.mean(pts, axis=0).astype(int)
                distance = np.linalg.norm(tvec)
                cv2.putText(output, f"ID:{marker_id} D:{distance:.2f}m",
                           (center[0] - 40, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制颜色检测结果
        if color_results:
            for result in color_results:
                x, y, w, h = result['bbox']
                cx, cy = result['center']
                cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)
        
        # 绘制十字准心
        self._draw_crosshair(output)
        
        return output
    
    def _draw_axes(self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                   axis_length: float = None):
        """绘制坐标轴"""
        if axis_length is None:
            axis_length = self.marker_length * 1.5
        
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])
        
        imgpts, _ = cv2.projectPoints(
            axis_points, rvec, tvec,
            self.camera.camera_matrix, self.camera.dist_coeffs
        )
        imgpts = imgpts.astype(int).reshape(-1, 2)
        
        origin = tuple(imgpts[0])
        cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 2)  # X - 红
        cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 2)  # Y - 绿
        cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 2)  # Z - 蓝
    
    def _draw_crosshair(self, image: np.ndarray, color: Tuple = (0, 255, 255),
                        length: int = 20, thickness: int = 2):
        """绘制十字准心"""
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        cv2.line(image, (cx - length, cy), (cx + length, cy), color, thickness)
        cv2.line(image, (cx, cy - length), (cx, cy + length), color, thickness)


class CoordinateTransformer:
    """
    坐标系变换器
    
    实现相机坐标系到机器人基坐标系的变换
    """
    
    def __init__(self, T_ee_camera: np.ndarray = None):
        """
        初始化坐标变换器
        
        Args:
            T_ee_camera: 末端执行器到相机的变换矩阵 (4x4)
        """
        if T_ee_camera is None:
            # 默认变换: 相机与末端执行器对齐
            self.T_ee_camera = np.array([
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
        else:
            self.T_ee_camera = T_ee_camera
    
    def camera_to_base(self, point_camera: np.ndarray, 
                       ee_pos: np.ndarray, ee_orn: np.ndarray) -> np.ndarray:
        """
        将相机坐标系中的点变换到机器人基坐标系
        
        Args:
            point_camera: 相机坐标系中的点 [x, y, z]
            ee_pos: 末端执行器位置 [x, y, z]
            ee_orn: 末端执行器姿态 (四元数)
            
        Returns:
            机器人基坐标系中的点 [x, y, z]
        """
        # 构建基座标系到末端执行器的变换矩阵
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = rot_matrix
        T_base_ee[:3, 3] = ee_pos
        
        # 相机坐标系中的点 (齐次坐标)
        point_camera_h = np.append(point_camera.flatten(), 1.0)
        
        # 变换: 基座标系 <- 末端执行器 <- 相机
        point_base_h = T_base_ee @ self.T_ee_camera @ point_camera_h
        
        return point_base_h[:3]
    
    def base_to_camera(self, point_base: np.ndarray,
                       ee_pos: np.ndarray, ee_orn: np.ndarray) -> np.ndarray:
        """
        将机器人基坐标系中的点变换到相机坐标系
        """
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = rot_matrix
        T_base_ee[:3, 3] = ee_pos
        
        # 逆变换
        T_camera_base = np.linalg.inv(self.T_ee_camera) @ np.linalg.inv(T_base_ee)
        
        point_base_h = np.append(point_base.flatten(), 1.0)
        point_camera_h = T_camera_base @ point_base_h
        
        return point_camera_h[:3]
    
    def rvec_to_quaternion(self, rvec: np.ndarray) -> np.ndarray:
        """旋转向量转四元数"""
        rot_matrix, _ = cv2.Rodrigues(rvec)
        # 使用scipy转换
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rot_matrix)
        return r.as_quat()  # [x, y, z, w]


class PerceptionModule:
    """
    感知模块主类
    
    集成相机、检测器和坐标变换功能
    """
    
    def __init__(self, robot_id: int, ee_link: int,
                 camera_width: int = 640, camera_height: int = 480,
                 camera_fov: float = 60.0, marker_length: float = 0.04):
        """
        初始化感知模块
        
        Args:
            robot_id: 机器人ID
            ee_link: 末端执行器链接索引
            camera_width: 相机图像宽度
            camera_height: 相机图像高度
            camera_fov: 相机视场角
            marker_length: ArUco标记边长
        """
        self.robot_id = robot_id
        self.ee_link = ee_link
        
        # 初始化子模块
        self.camera = CameraSimulator(camera_width, camera_height, camera_fov)
        self.detector = TargetDetector(self.camera, marker_length)
        self.transformer = CoordinateTransformer()
        
        # 状态变量
        self.current_image = None
        self.current_depth = None
        self.target_state = TargetState()
        
        # 滤波器状态 (简单低通滤波)
        self._filtered_position = None
        self._filter_alpha = 0.3  # 滤波系数
    
    def update(self, target_marker_id: int = None) -> TargetState:
        """
        更新感知状态
        
        Args:
            target_marker_id: 目标标记ID (None表示检测任意标记)
            
        Returns:
            目标状态
        """
        import time
        
        # 获取相机图像
        self.current_image, self.current_depth = self.camera.get_image_from_ee(
            self.robot_id, self.ee_link
        )
        
        # 检测ArUco标记
        aruco_results = self.detector.detect_aruco(self.current_image)
        
        # 获取末端执行器位姿
        ee_pos, ee_orn, _ = self.camera.get_ee_pose(self.robot_id, self.ee_link)
        
        # 更新目标状态
        self.target_state = TargetState()
        self.target_state.timestamp = time.time()
        
        if aruco_results:
            # 筛选目标标记
            target_result = None
            if target_marker_id is not None:
                for result in aruco_results:
                    if result['marker_id'] == target_marker_id:
                        target_result = result
                        break
            else:
                target_result = aruco_results[0]  # 使用第一个检测到的标记
            
            if target_result:
                self.target_state.detected = True
                self.target_state.target_type = TargetType.ARUCO_MARKER
                self.target_state.marker_id = target_result['marker_id']
                self.target_state.confidence = target_result['confidence']
                
                # 相机坐标系位姿
                self.target_state.position_camera = target_result['tvec'].flatten()
                self.target_state.orientation_camera = target_result['rvec'].flatten()
                
                # 转换到基座标系
                position_base = self.transformer.camera_to_base(
                    target_result['tvec'], ee_pos, ee_orn
                )
                
                # 应用低通滤波
                if self._filtered_position is None:
                    self._filtered_position = position_base
                else:
                    self._filtered_position = (
                        self._filter_alpha * position_base +
                        (1 - self._filter_alpha) * self._filtered_position
                    )
                
                self.target_state.position_base = self._filtered_position.copy()
                self.target_state.orientation_base = self.transformer.rvec_to_quaternion(
                    target_result['rvec']
                )
        else:
            self.target_state.detected = False
            # 保持上一次的滤波位置
            if self._filtered_position is not None:
                self.target_state.position_base = self._filtered_position.copy()
        
        return self.target_state
    
    def get_visualization(self) -> np.ndarray:
        """
        获取可视化图像
        
        Returns:
            带有检测结果标注的图像
        """
        if self.current_image is None:
            return None
        
        aruco_results = self.detector.detect_aruco(self.current_image)
        vis_image = self.detector.draw_detection(self.current_image, aruco_results)
        
        # 添加状态信息
        if self.target_state.detected:
            pos = self.target_state.position_base
            info_text = f"Target: ID={self.target_state.marker_id}"
            cv2.putText(vis_image, info_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if pos is not None:
                pos_text = f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                cv2.putText(vis_image, pos_text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(vis_image, "No Target Detected", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_image
    
    def get_image_error(self) -> Optional[np.ndarray]:
        """
        计算图像空间误差 (用于IBVS)
        
        Returns:
            图像中心到目标中心的像素误差 [ex, ey]
        """
        if not self.target_state.detected:
            return None
        
        aruco_results = self.detector.detect_aruco(self.current_image)
        if not aruco_results:
            return None
        
        # 找到目标标记
        target_result = None
        for result in aruco_results:
            if result['marker_id'] == self.target_state.marker_id:
                target_result = result
                break
        
        if target_result is None:
            return None
        
        # 计算标记中心
        corners = target_result['corners'][0]
        marker_center = np.mean(corners, axis=0)
        
        # 图像中心
        image_center = np.array([self.camera.width / 2, self.camera.height / 2])
        
        # 像素误差
        error = marker_center - image_center
        
        return error
    
    def reset_filter(self):
        """重置滤波器状态"""
        self._filtered_position = None

