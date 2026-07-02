import sys
import cv2
import re
import webbrowser
import numpy as np
import io
import os
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QMessageBox, QTextEdit, QFrame, QListWidget, QGroupBox,
    QTabWidget, QLineEdit, QColorDialog, QScrollArea, QShortcut
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QMimeData, QUrl, QCoreApplication, QBuffer, QIODevice
from pyzbar.pyzbar import decode, ZBarSymbol
import math

# 适应高DPI设备
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)


def is_url(text):
    return re.match(r'https?://[\w./?=&-]+', text)


def enhance_qr_image(img):
    """
    预处理图像以提高二维码识别率。
    包括灰度转换、Otsu自适应阈值化、闭运算和锐化。
    适用于光照相对均匀的情况。
    """
    if img is None or img.size == 0:
        return None

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 自适应阈值化 (Otsu's Binarization)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 形态学操作：闭运算连接断裂线条和填充小孔
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. 锐化 (可选，对模糊图像有帮助)
    sharpened = cv2.filter2D(morphed, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    return sharpened


def enhance_qr_image_adaptive(img):
    """
    使用自适应阈值化增强图像，适用于光照不均匀或存在阴影的情况（如纸张弯曲）。
    """
    if img is None or img.size == 0:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自适应阈值化: ADAPTIVE_THRESH_GAUSSIAN_C 更平滑，对噪声敏感度低
    # blockSize: 区域大小，奇数，例如 15
    # C: 常量，从均值或加权均值中减去的值
    binary_adaptive = cv2.adaptiveThreshold(gray, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 2)  # blockSize 15, C 2

    # 形态学操作和锐化与 enhance_qr_image 保持一致
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    sharpened = cv2.filter2D(morphed, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    return sharpened


def find_finder_patterns_custom(binary_img):
    """
    在二值图像中查找QR码的三个查找器模式（Finder Patterns）。
    使用轮廓检测和几何特性（三层嵌套结构，1:1:3:1:1比例）。
    返回找到的三个查找器模式的中心点和大小。

    对于边缘有弧度的QR码，轮廓可能不是完美的正方形，但内部的比例关系通常保持。
    此函数对几何容差的调整需要非常谨慎，以避免误识别。
    """
    if binary_img is None or binary_img.size == 0:
        return []

    # 查找所有轮廓
    # RETR_TREE 对于查找嵌套轮廓非常有用
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选潜在的查找器模式
    potential_patterns = []
    min_area = 50  # 最小轮廓面积，可根据实际QR码大小调整
    max_area_ratio = 0.05  # 轮廓最大面积占图像总面积的比例，防止大噪声

    img_area = binary_img.shape[0] * binary_img.shape[1]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > img_area * max_area_ratio:
            continue

        # 近似多边形，查找矩形
        # 对于有弧度的二维码，epsilon可以适当调大一些，允许更大的近似误差，但需谨慎
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 必须是近似四边形
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            # 检查宽高比是否接近1，并且是凸包
            # 对于有弧度的二维码，0.7 <= aspect_ratio <= 1.3 可能会更合适，但会增加误识别
            if 0.8 <= aspect_ratio <= 1.2 and cv2.isContourConvex(approx):
                # 检查三层嵌套结构
                # hierarchy[0][i] = [next, previous, child, parent]
                current_level_has_child = hierarchy[0][i][2] != -1
                current_level_has_parent = hierarchy[0][i][3] != -1

                if current_level_has_child and current_level_has_parent:  # Current is inner-most square
                    child_idx = hierarchy[0][i][2]  # This is wrong, hierarchy[0][i][2] is the *first child*
                    # A finder pattern is usually a black square, surrounded by a white square, surrounded by a black square.
                    # So, we need to find an outer contour, which has a child, and that child has a child.
                    # Or, start from an inner contour and find its two parents.

                    # Let's refine the nesting check: find a contour that has a child, and its child has a parent (itself)
                    # and the original contour also has a parent (the outer black square).

                    # More robust way: Find contours that are themselves a child, and have a child, and their parent has a parent.
                    # This implies a chain: Grandparent (outer black) -> Parent (white) -> Current (inner black)
                    if hierarchy[0][i][3] != -1:  # Has a parent (white square)
                        parent_idx = hierarchy[0][i][3]
                        if hierarchy[0][parent_idx][3] != -1:  # Parent has a parent (outer black square)
                            grandparent_idx = hierarchy[0][parent_idx][3]

                            # Ensure the sizes are nesting correctly
                            gp_x, gp_y, gp_w, gp_h = cv2.boundingRect(contours[grandparent_idx])
                            p_x, p_y, p_w, p_h = cv2.boundingRect(contours[parent_idx])

                            # Check for reasonable size progression for nesting
                            # Outer square (grandparent) should be approx 3x inner black square (current) or 7x module size
                            # Inner square (current) should be approx 1x module size
                            # These ratios (3x, 5x, 7x module size) are critical for robust detection

                            # Simple nesting size check:
                            if gp_w > p_w and p_w > w and gp_h > p_h and p_h > h:
                                # Further verify with 1:1:3:1:1 ratio
                                # For curved QR codes, the ratio check's tolerance (`tolerance` variable in `verify_finder_pattern_ratio`)
                                # 可能需要适当放宽。
                                if verify_finder_pattern_ratio(binary_img, (x + w // 2, y + h // 2), w):
                                    potential_patterns.append({
                                        'center': (x + w // 2, y + h // 2),
                                        'size': max(w, h),
                                        'contour': cnt
                                    })

    # 从潜在模式中筛选出3个最符合条件的查找器模式
    # 策略：选择大小相似且距离合适的三个，构成近似直角三角形
    if len(potential_patterns) >= 3:
        # 为了避免过于复杂的算法，这里选择了一种简化的筛选，实际产品级代码需要更精细的几何验证
        # 遍历所有三元组，寻找最接近直角三角形的组合
        best_patterns = []
        min_error = float('inf')

        for i in range(len(potential_patterns)):
            for j in range(i + 1, len(potential_patterns)):
                for k in range(j + 1, len(potential_patterns)):
                    p1_obj, p2_obj, p3_obj = potential_patterns[i], potential_patterns[j], potential_patterns[k]
                    p1, p2, p3 = p1_obj['center'], p2_obj['center'], p3_obj['center']

                    dists_sq = sorted([
                        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2,
                        (p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2,
                        (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2
                    ])

                    # 检查是否接近直角三角形 a^2 + b^2 = c^2
                    # 引入一定的容差
                    tolerance_ratio = 0.15  # 15% 容差，比之前略大，以容忍轻微弧度

                    if dists_sq[2] > 0:  # 避免除以零
                        error = abs((dists_sq[0] + dists_sq[1]) - dists_sq[2]) / dists_sq[2]
                        if error < tolerance_ratio and error < min_error:
                            min_error = error
                            best_patterns = [p1_obj, p2_obj, p3_obj]

        if len(best_patterns) == 3:
            # 找到直角点 (通常是左上角)
            p_centers = [np.array(p['center']) for p in best_patterns]
            # 计算两两距离
            dist01 = np.linalg.norm(p_centers[0] - p_centers[1])
            dist02 = np.linalg.norm(p_centers[0] - p_centers[2])
            dist12 = np.linalg.norm(p_centers[1] - p_centers[2])

            # 找到最长边（斜边），其对面的点就是直角点
            if dist01 > dist02 and dist01 > dist12:
                # 0-1 是斜边，2 是直角点
                top_left_idx = 2
            elif dist02 > dist01 and dist02 > dist12:
                # 0-2 是斜边，1 是直角点
                top_left_idx = 1
            else:
                # 1-2 是斜边，0 是直角点
                top_left_idx = 0

            p_tl = p_centers[top_left_idx]
            other_points = [p for i, p in enumerate(p_centers) if i != top_left_idx]

            # 确定右上和左下（通过x轴比较）
            if other_points[0][0] > other_points[1][0]:
                p_tr = other_points[0]
                p_bl = other_points[1]
            else:
                p_tr = other_points[1]
                p_bl = other_points[0]

            # 返回按顺序的四个点 (top_left, top_right, bottom_right, bottom_left)
            # 为了deskew_qr_code，需要返回四个点
            # 估算第四个角点 (右下)
            p_br = p_tr + p_bl - p_tl

            # 返回四个角点，以便deskew_qr_code可以直接使用
            return np.float32([p_tl, p_tr, p_br, p_bl])
    return []


def verify_finder_pattern_ratio(binary_img, center, size):
    """
    通过线扫描验证查找器模式的1:1:3:1:1黑白模块比例。
    从中心点沿四个方向（水平、垂直、对角线）进行扫描。
    对于有弧度的二维码，比例的容差可以适当放宽。
    """
    cx, cy = int(center[0]), int(center[1])
    half_size = int(size / 2)

    # 简单的边界检查
    if not (0 <= cx < binary_img.shape[1] and 0 <= cy < binary_img.shape[0]):
        return False

    # 定义扫描线步长，以中心点为原点
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # (dx, dy)

    # 期望的比例模式 (1:1:3:1:1)
    expected_ratios = np.array([1, 1, 3, 1, 1], dtype=np.float32)
    total_expected_sum = np.sum(expected_ratios)

    for dx, dy in directions:
        segments = []

        # 扫描两个方向
        for direction_factor in [-1, 1]:
            current_segments = []
            current_color = -1
            current_length = 0

            # 确保扫描范围在图像内
            for step in range(0, half_size):  # 扫描半个finder pattern区域
                px, py = cx + direction_factor * step * dx, cy + direction_factor * step * dy
                if not (0 <= px < binary_img.shape[1] and 0 <= py < binary_img.shape[0]):
                    break  # 超出图像边界

                pixel_val = binary_img[py, px]

                if current_color == -1:
                    current_color = pixel_val

                if pixel_val == current_color:
                    current_length += 1
                else:
                    current_segments.append(current_length)
                    current_color = pixel_val
                    current_length = 1
            if current_length > 0:
                current_segments.append(current_length)

            # 对于反方向扫描，需要反转段的顺序
            if direction_factor == -1:
                current_segments.reverse()
            segments.extend(current_segments)

        # 组合正反两个方向的扫描结果，并处理中心点的重叠
        # 通常中心点会被扫描两次，需要合并
        # 这里简化为直接检查segments，需要确保至少有5段

        # 尝试匹配1:1:3:1:1模式
        # 这是一个简化的匹配逻辑，更鲁棒的实现会使用滑动窗口或更复杂的模式识别
        # 找到最长的连续5个段，且最中间的一个段是最大的（3的比例）

        # 确保至少有5个段
        if len(segments) >= 5:
            for i in range(len(segments) - 4):
                current_pattern_segments = np.array(segments[i:i + 5], dtype=np.float32)

                # Finder pattern的中心是黑白黑 (1:1:3:1:1)。
                # 所以中间的3部分应该是 Black-White-Black, 总共5个部分。
                # 检查中间段是否是最大的
                if np.argmax(current_pattern_segments) == 2:  # 索引2是第三个段，即1:1:3中的3
                    current_sum = np.sum(current_pattern_segments)
                    if current_sum == 0: continue

                    ratios = current_pattern_segments / current_sum

                    # 期望比例的百分比
                    expected_percentages = expected_ratios / total_expected_sum

                    # 检查比例是否在容差范围内
                    # 对于弧度二维码，适当增加容差
                    tolerance = 0.25  # 25% 容差，可以根据实际情况调整

                    match = True
                    for j in range(5):
                        # 检查绝对误差百分比
                        if abs(ratios[j] - expected_percentages[j]) > tolerance * expected_percentages[j]:
                            match = False
                            break

                    if match:
                        return True  # 找到符合条件的模式

    return False  # 未找到符合条件的模式


def deskew_qr_code(img, qr_points, target_size=300):
    """
    根据QR码的四个角点进行透视校正。
    对于有弧度的二维码，此线性透视校正可能无法完美纠正，但仍是尝试的第一步。
    :param img: 原始图像
    :param qr_points: QR码的四个角点，格式为 NumPy 数组 (4, 2)
                      例如 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param target_size: 校正后图像的目标尺寸
    :return: 校正后的图像，如果无法校正则返回None
    """
    if qr_points is None or qr_points.shape != (4, 2):
        return None

    src_pts = np.float32(qr_points)
    dst_pts = np.float32([[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_img = cv2.warpPerspective(img, M, (target_size, target_size))

    return warped_img


class QRScannerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("美观二维码识别系统")
        self.setAcceptDrops(True)
        self.setStyleSheet(self.get_app_stylesheet())
        self.resize(1200, 700)

        self.last_data = ""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self._generated_qr_img_pil = None

        self.init_ui()

        # Ctrl+V 粘贴图片快捷键
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_image)

    def get_app_stylesheet(self):
        # 统一的样式表
        return """
            QWidget {
                background-color: #f8f9fa;
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 16px; /* 增大字号 */
                color: #343a40;
            }
            QLabel {
                color: #495057;
            }
            QGroupBox {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 15px; /* 增大标题与边框的距离 */
                font-size: 18px; /* 增大 GroupBox 标题字号 */
                font-weight: bold;
                color: #212529;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #f8f9fa;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 12px 18px; /* 增大按钮内边距 */
                border-radius: 6px;
                font-size: 16px; /* 增大按钮字号 */
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
            QPushButton#camera_btn {
                background-color: #28a745;
            }
            QPushButton#camera_btn:hover {
                background-color: #218838;
            }
            QPushButton#file_btn {
                background-color: #17a2b8;
            }
            QPushButton#file_btn:hover {
                background-color: #138496;
            }
            QTextEdit, QLineEdit, QListWidget {
                border: 1px solid #ced4da;
                border-radius: 5px;
                padding: 10px; /* 增大文本输入区域内边距 */
                background-color: #ffffff;
                font-size: 15px; /* 增大文本输入区域字号 */
            }
            QListWidget::item {
                padding: 6px; /* 增大列表项内边距 */
                font-size: 15px; /* 增大列表项字号 */
            }
            QListWidget::item:selected {
                background-color: #e9ecef;
                color: #495057;
            }
            QTabWidget::pane {
                border: 1px solid #ced4da;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QTabWidget::tab-bar {
                left: 5px;
            }
            QTabBar::tab {
                background: #e9ecef;
                border: 1px solid #ced4da;
                border-bottom-color: #ced4da;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 10px 20px; /* 增大 Tab 标题内边距 */
                min-width: 90px;
                font-size: 15px; /* 增大 Tab 标题字号 */
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #f1f3f5;
            }
        """

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- 左侧：主预览与识别结果 ---
        left_panel_layout = QVBoxLayout()

        # 摄像头/图片预览
        self.image_label = QLabel("摄像头/图片预览")
        self.image_label.setFixedSize(600, 450)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #adb5bd; border-radius: 8px; background-color: #e9ecef;")
        left_panel_layout.addWidget(self.image_label)

        # 识别结果输出
        recognition_group = QGroupBox("识别结果")
        recognition_layout = QVBoxLayout(recognition_group)
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setFixedHeight(80)
        recognition_layout.addWidget(self.text_output)

        # 复制按钮
        copy_btn = QPushButton("📋 复制结果")
        copy_btn.setStyleSheet("background-color: #6c757d; border-radius: 5px; padding: 8px; font-size: 15px;")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        recognition_layout.addWidget(copy_btn)

        left_panel_layout.addWidget(recognition_group)

        # 操作按钮区
        control_buttons_layout = QHBoxLayout()
        self.camera_btn = QPushButton("📷 打开摄像头")
        self.camera_btn.setObjectName("camera_btn")
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.file_btn = QPushButton("🖼️ 选择本地图片")
        self.file_btn.setObjectName("file_btn")
        self.file_btn.clicked.connect(self.load_image)
        control_buttons_layout.addWidget(self.camera_btn)
        control_buttons_layout.addWidget(self.file_btn)
        left_panel_layout.addLayout(control_buttons_layout)

        main_layout.addLayout(left_panel_layout)

        # --- 右侧：功能区 (Tab Widget) ---
        right_panel_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)

        # --- 识别页面 ---
        scan_tab = QWidget()
        scan_layout = QVBoxLayout(scan_tab)

        qr_display_group = QGroupBox("识别到的二维码区域")
        qr_display_layout = QVBoxLayout(qr_display_group)
        self.qr_label = QLabel("二维码特写")
        self.qr_label.setFixedSize(250, 250)
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setStyleSheet("border: 2px dashed #fd7e14; border-radius: 8px; background-color: #fff;")
        qr_display_layout.addWidget(self.qr_label, alignment=Qt.AlignCenter)
        scan_layout.addWidget(qr_display_group)

        history_group = QGroupBox("识别历史")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.setFixedHeight(150)
        self.history_list.itemDoubleClicked.connect(self.open_history_item)
        history_layout.addWidget(self.history_list)

        # 调整：增加伸缩，让按钮贴底
        history_layout.addStretch()
        clear_history_btn = QPushButton("🗑️ 清空历史")
        clear_history_btn.setStyleSheet("background-color: #dc3545; border-radius: 5px; padding: 8px; font-size: 15px;")
        clear_history_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(clear_history_btn)

        scan_layout.addWidget(history_group)
        scan_layout.addStretch()  # 填充扫描页面的剩余空间

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(scan_tab, "🔍 二维码识别")

        # --- 生成页面 ---
        generate_tab = QWidget()
        generate_layout = QVBoxLayout(generate_tab)

        input_group = QGroupBox("输入内容生成二维码")
        input_layout = QVBoxLayout(input_group)
        self.qr_content_input = QLineEdit()
        self.qr_content_input.setPlaceholderText("在此输入文本或网址...")
        self.qr_content_input.setFixedHeight(40)
        input_layout.addWidget(self.qr_content_input)

        generate_btn = QPushButton("✨ 生成二维码")
        generate_btn.setStyleSheet("background-color: #6f42c1;")
        generate_btn.clicked.connect(self.generate_qr_code)
        input_layout.addWidget(generate_btn)
        generate_layout.addWidget(input_group)

        # 二维码生成预览
        generated_qr_group = QGroupBox("生成预览")
        generated_qr_layout = QVBoxLayout(generated_qr_group)
        self.generated_qr_label = QLabel("生成的二维码将显示在此")
        self.generated_qr_label.setFixedSize(250, 250)
        self.generated_qr_label.setAlignment(Qt.AlignCenter)
        self.generated_qr_label.setStyleSheet("border: 2px solid #007bff; border-radius: 8px; background-color: #fff;")
        generated_qr_layout.addWidget(self.generated_qr_label, alignment=Qt.AlignCenter)

        # 使保存二维码按钮靠近底部
        generated_qr_layout.addStretch()  # 确保预览图下方有可伸缩空间
        save_qr_btn = QPushButton("💾 保存二维码")
        save_qr_btn.clicked.connect(self.save_generated_qr)
        save_qr_btn.setStyleSheet("background-color: #ffc107; color: black; padding: 10px 15px;")
        generated_qr_layout.addWidget(save_qr_btn)

        generate_layout.setContentsMargins(10, 10, 10, 0)
        generate_layout.addWidget(generated_qr_group, 0, Qt.AlignBottom)

        self.tab_widget.addTab(generate_tab, "✏️ 二维码生成")

        right_panel_layout.addWidget(self.tab_widget)
        main_layout.addLayout(right_panel_layout)

    # --- 功能方法 ---
    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_output.toPlainText())
        QMessageBox.information(self, "复制成功", "识别结果已复制到剪贴板！")

    def clear_history(self):
        reply = QMessageBox.question(self, "清空历史", "确定要清空所有识别历史吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_list.clear()
            self.last_data = ""
            QMessageBox.information(self, "操作成功", "识别历史已清空。")

    def open_history_item(self, item):
        data = item.text()
        self.text_output.setText(data)
        QMessageBox.information(self, "历史记录", f"正在显示历史记录：\n{data}")

    def generate_qr_code(self):
        content = self.qr_content_input.text().strip()
        if not content:
            QMessageBox.warning(self, "输入为空", "请输入要生成二维码的内容！")
            self.generated_qr_label.clear()
            self.generated_qr_label.setText("请输入内容...")
            return

        try:
            import qrcode
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            # 关键修改：将字符串编码为UTF-8字节串
            qr.add_data(content)
            qr.make(fit=True)

            img_pil = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            self._generated_qr_img_pil = img_pil

            q_img = QImage(img_pil.tobytes(), img_pil.width, img_pil.height,
                           img_pil.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.generated_qr_label.size(), Qt.KeepAspectRatio)
            self.generated_qr_label.setPixmap(pixmap)

        except ImportError:
            QMessageBox.critical(self, "缺少库", "请安装 'qrcode' 库：\npip install qrcode[pil]")
        except Exception as e:
            QMessageBox.critical(self, "生成失败", f"生成二维码时发生错误: {e}")

    def save_generated_qr(self):
        if self._generated_qr_img_pil is None:
            QMessageBox.warning(self, "无二维码", "请先生成一个二维码再保存。")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "保存二维码图片", "qrcode.png",
                                                   "PNG Images (*.png);;JPG Images (*.jpg);;BMP Images (*.bmp)")
        if file_name:
            try:
                self._generated_qr_img_pil.save(file_name)
                QMessageBox.information(self, "保存成功", f"二维码已保存到：\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存二维码时发生错误: {e}")

    def toggle_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.image_label.clear()
            self.image_label.setText("摄像头/图片预览")
            self.camera_btn.setText("📷 打开摄像头")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头，请检查设备连接。")
                self.cap = None
                return
            self.timer.start(30)
            self.camera_btn.setText("🛑 关闭摄像头")

    def capture_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.display_image(frame)
            self.decode_qr(frame)
        else:
            self.image_label.setText("无法读取摄像头画面")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            image = self.imread_unicode(file_name)
            self.display_image(image)
            success = self.decode_qr(image)
            if not success:
                QMessageBox.information(self, "识别失败", "未识别到二维码，请尝试其他图片。")

    def imread_unicode(self, path):
        with open(path, 'rb') as f:
            bytes_data = f.read()
        image = np.array(Image.open(io.BytesIO(bytes_data)).convert('RGB'))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def display_image(self, img):
        if img is None or img.size == 0:
            self.image_label.setText("无法显示图像")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def decode_qr(self, img):
        if img is None or img.size == 0:
            self.text_output.setText("输入图像无效")
            self.qr_label.clear()
            return False

        original_img = img.copy()  # 保留原始图像副本，用于后续处理

        # --- 1. 超大尺寸图像缩放处理 ---
        # 优化超大尺寸图像的识别速度，防止崩溃
        max_dim = max(img.shape[0], img.shape[1])
        SCALE_THRESHOLD = 1500  # 图像最长边超过此值时进行缩放

        # 记录是否进行了缩放，用于后续坐标转换
        scaled = False
        if max_dim > SCALE_THRESHOLD:
            scale_factor = SCALE_THRESHOLD / max_dim
            img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)),
                             interpolation=cv2.INTER_AREA)
            scaled = True
            print(f"图像已缩放至: {img.shape[1]}x{img.shape[0]}")

        decoded_objs = []
        qr_roi_to_display = None  # 用于显示在二维码特写区域的图像

        # --- 2. 尝试多种预处理和定位策略 ---
        # 策略优先级：
        #   a. 自定义定位 + 常规增强图像 -> pyzbar解码
        #   b. pyzbar直接解码原始图像
        #   c. pyzbar直接解码常规增强图像
        #   d. 自定义定位 + 自适应增强图像 -> pyzbar解码 (新增，处理弧度/不均光照)
        #   e. pyzbar直接解码自适应增强图像
        #   f. 自定义定位 + 反色常规增强图像 -> pyzbar解码
        #   g. pyzbar直接解码反色图像
        #   h. pyzbar直接解码反色常规增强图像
        #   i. 自定义定位 + 反色自适应增强图像 -> pyzbar解码
        #   j. pyzbar直接解码反色自适应增强图像
        #   k. OpenCV自带QRCodeDetector解码 (最终备选)

        processing_attempts = [
            # (处理函数, 是否反色, 描述)
            (enhance_qr_image, False, "自定义定位+常规增强图像"),
            (None, False, "pyzbar原始图像"),  # 直接用img
            (enhance_qr_image, False, "pyzbar常规增强图像"),
            (enhance_qr_image_adaptive, False, "自定义定位+自适应增强图像 (处理弧度/不均光照)"),  # 新增
            (enhance_qr_image_adaptive, False, "pyzbar自适应增强图像 (处理弧度/不均光照)"),  # 新增

            # 反色处理
            (enhance_qr_image, True, "自定义定位+反色常规增强图像"),
            (None, True, "pyzbar反色原始图像"),  # 直接用inv_img
            (enhance_qr_image, True, "pyzbar反色常规增强图像"),
            (enhance_qr_image_adaptive, True, "自定义定位+反色自适应增强图像 (处理弧度/不均光照)"),  # 新增
            (enhance_qr_image_adaptive, True, "pyzbar反色自适应增强图像 (处理弧度/不均光照)"),  # 新增
        ]

        for preproc_func, invert_color, desc in processing_attempts:
            if decoded_objs:  # 如果已经识别到，则跳过后续尝试
                break

            current_img = img.copy()
            if invert_color:
                current_img = cv2.bitwise_not(current_img)

            processed_img = None
            if preproc_func:
                processed_img = preproc_func(current_img.copy())
            else:
                processed_img = current_img  # No specific preprocessing, use current image

            if processed_img is None:
                continue

            # 尝试自定义定位
            if "自定义定位" in desc:
                custom_points = find_finder_patterns_custom(processed_img)
                if custom_points and len(custom_points) == 4:  # 期望返回4个角点
                    # 确保 custom_points 的坐标与当前处理图像的缩放比例匹配
                    warped_img_custom = deskew_qr_code(current_img.copy(), custom_points)

                    if warped_img_custom is not None and warped_img_custom.shape[0] > 0 and warped_img_custom.shape[
                        1] > 0:
                        decoded_by_custom = decode(warped_img_custom, symbols=[ZBarSymbol.QRCODE])
                        if decoded_by_custom:
                            decoded_objs.extend(decoded_by_custom)
                            qr_roi_to_display = warped_img_custom
                            print(f"通过 {desc} 解码成功！")
                            # 立即处理并返回，避免重复识别
                            if decoded_objs:
                                obj = decoded_objs[0]
                                data = obj.data.decode("utf-8")
                                self.process_and_display_result(data, qr_roi_to_display, scaled,
                                                                scale_factor if scaled else 1)
                                return True
            else:  # 直接使用pyzbar解码
                decoded_by_pyzbar = decode(processed_img, symbols=[ZBarSymbol.QRCODE])
                if decoded_by_pyzbar:
                    decoded_objs.extend(decoded_by_pyzbar)
                    print(f"通过 {desc} 解码成功！")
                    # 立即处理并返回
                    if decoded_objs:
                        obj = decoded_objs[0]
                        data = obj.data.decode("utf-8")
                        # 尝试从pyzbar结果中获取ROI
                        roi_from_pyzbar = self._get_roi_from_pyzbar_obj(obj, original_img, scaled,
                                                                        scale_factor if scaled else 1)
                        self.process_and_display_result(data, roi_from_pyzbar, scaled, scale_factor if scaled else 1)
                        return True

        # --- 3. 最后尝试使用OpenCV自带的QRCodeDetector（作为最终备选）---
        if not decoded_objs:
            qrDetector = cv2.QRCodeDetector()
            # 优先使用缩放后的图像进行检测，因为其处理速度更快，但也尝试原始图像以确保最大鲁棒性
            retval_cv, points_cv, straight_qrcode_cv = qrDetector.detectAndDecode(img if scaled else original_img)

            if retval_cv and points_cv is not None and len(points_cv) > 0:
                data = retval_cv
                print("通过OpenCV QRCodeDetector解码成功！")
                if straight_qrcode_cv is not None and straight_qrcode_cv.shape[0] > 0 and straight_qrcode_cv.shape[
                    1] > 0:
                    qr_roi_to_display = straight_qrcode_cv
                self.process_and_display_result(data, qr_roi_to_display, scaled, scale_factor if scaled else 1)
                return True
            else:
                print("OpenCV QRCodeDetector未能检测到或解码二维码。")

        # --- 识别结果处理 ---
        self.qr_label.clear()
        self.qr_label.setText("未识别到二维码")
        self.text_output.setText("未识别到二维码")
        print("所有方法均未能识别到二维码。")
        return False

    def _get_roi_from_pyzbar_obj(self, obj, original_img, scaled, scale_factor):
        """
        根据pyzbar返回的对象获取ROI图像。
        """
        roi_img = None
        if obj.polygon is not None:
            if isinstance(obj.polygon, np.ndarray):
                pts_np = obj.polygon
            else:
                pts_np = np.array([(p.x, p.y) for p in obj.polygon], dtype=np.int32)

            # 如果图像被缩放过，将pyzbar返回的ROI坐标转换回原始图像坐标
            if scaled and scale_factor != 0:
                pts_np = (pts_np / scale_factor).astype(np.int32)

            x, y, w, h = cv2.boundingRect(pts_np)

            # 扩大搜索范围以获取正确的二维码边缘（增加边框）
            padding = 20  # 扩大像素范围
            img_height, img_width = original_img.shape[:2]

            x_clip_start = max(0, x - padding)
            y_clip_start = max(0, y - padding)
            x_clip_end = min(img_width, x + w + padding)
            y_clip_end = min(img_height, y + h + padding)

            clipped_w = x_clip_end - x_clip_start
            clipped_h = y_clip_end - y_clip_start

            if clipped_w > 0 and clipped_h > 0:
                roi_img = original_img[y_clip_start:y_clip_end, x_clip_start:x_clip_end]
                if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                    roi_img = None
            else:
                print(f"警告: obj.polygon 计算出的 ROI 经过裁剪和扩展后宽度或高度为零: w={clipped_w}, h={clipped_h}。")
        return roi_img

    def process_and_display_result(self, data, qr_roi_to_display=None, scaled=False, scale_factor=1.0):
        """
        处理和显示识别结果。
        """
        if data == self.last_data:
            return

        self.last_data = data

        # 更新历史记录
        existing = [self.history_list.item(i).text() for i in range(self.history_list.count())]
        if data not in existing:
            self.history_list.addItem(data)

        # 显示结果文本
        self.text_output.setText(data)

        # 显示二维码特写
        if qr_roi_to_display is not None and qr_roi_to_display.size > 0:
            # 确保ROI图像是RGB，并缩放以适应QLabel
            # 如果ROI是来自pyzbar并且图像之前被缩放过，pyzbar的polygon坐标需要反向缩放
            # 这里已经处理过了，qr_roi_to_display是根据original_img裁剪的

            # 确保ROI图像是RGB
            if len(qr_roi_to_display.shape) == 3:
                roi_rgb = cv2.cvtColor(qr_roi_to_display, cv2.COLOR_BGR2RGB)
            else:
                roi_rgb = cv2.cvtColor(qr_roi_to_display, cv2.COLOR_GRAY2RGB)

            h_roi, w_roi, ch_roi = roi_rgb.shape
            bytes_per_line_roi = ch_roi * w_roi
            roi_qimg = QImage(roi_rgb.data, w_roi, h_roi, bytes_per_line_roi, QImage.Format_RGB888)
            self.qr_label.setPixmap(
                QPixmap.fromImage(roi_qimg).scaled(self.qr_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.qr_label.clear()
            self.qr_label.setText("无法获取二维码区域")

        # 如果是网址，询问是否打开
        if is_url(data):
            reply = QMessageBox.question(self, "打开网址", f"识别到网址：{data}\n是否打开？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                webbrowser.open(data)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                image = self.imread_unicode(file_path)
                self.display_image(image)
                success = self.decode_qr(image)
                if not success:
                    QMessageBox.information(self, "识别失败", "未识别到二维码，请尝试其他图片。")

    def paste_image(self):
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        img_np = None
        # 优先检查图像数据
        if mime.hasImage():
            qimg = clipboard.image()
            buffer = QBuffer()
            buffer.open(QIODevice.ReadWrite)
            qimg.save(buffer, 'PNG')
            pil_img = Image.open(io.BytesIO(buffer.data().data())).convert('RGB')
            img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 再检查文件 URL
        elif mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if os.path.isfile(path):
                    img_np = self.imread_unicode(path)
                    break
        if img_np is not None:
            self.display_image(img_np)
            if not self.decode_qr(img_np):
                QMessageBox.information(self, "识别失败", "未识别到二维码，请尝试其他图片。")
        else:
            QMessageBox.warning(self, "无图像", "剪贴板中没有可用的图片数据！")


if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    window = QRScannerApp()
    window.show()
    sys.exit(app.exec_())