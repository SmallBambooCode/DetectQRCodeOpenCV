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

# 判断字符串是否为网址，正则表达式匹配
def is_url(text):
    return re.match(r'https?://[\w./?=&-]+', text)

# 图像预处理（增强）
def enhance_qr_image(img):
    # 图像预处理以提高二维码识别率。
    # 包括灰度转换、Otsu自适应阈值化、闭运算和锐化。
    # 测试输入
    if img is None or img.size == 0:
        return None
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1.自适应阈值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 2.形态学操作：闭运算连接断裂线条和填充小孔
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 3.锐化（增强模糊图像）
    sharpened = cv2.filter2D(morphed, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    return sharpened

def find_finder_patterns_custom(binary_img):
    """
    在二值图像中查找QR码的三个查找器模式（Finder Patterns）。
    使用轮廓检测和几何特性（三层嵌套结构，1:1:3:1:1比例）。
    返回找到的三个查找器模式的中心点和大小。
    """
    if binary_img is None or binary_img.size == 0:
        return []

    # 查找所有轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 必须是近似正方形 (4个顶点)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            # 检查宽高比是否接近1，并且是凸包
            if 0.8 <= aspect_ratio <= 1.2 and cv2.isContourConvex(approx):
                # 检查三层嵌套结构
                # RETR_TREE 模式下，hierarchy[i][2] 是子轮廓的索引，hierarchy[i][3] 是父轮廓的索引
                # 查找具有两层父级和一层子级（共三层）的轮廓
                # 外 -> 中 -> 内 的结构
                # 检查当前轮廓的父级和父级的父级是否存在，并且有子级
                has_parent = _[0][i][3] != -1
                if has_parent:
                    parent_idx = _[0][i][3]
                    has_grandparent = _[0][parent_idx][3] != -1
                    if has_grandparent:
                        grandparent_idx = _[0][parent_idx][3]

                        # 确保三层是嵌套的，且是正方形
                        gp_x, gp_y, gp_w, gp_h = cv2.boundingRect(contours[grandparent_idx])
                        p_x, p_y, p_w, p_h = cv2.boundingRect(contours[parent_idx])

                        # 简单的尺寸比例检查
                        if gp_w > p_w and p_w > w and gp_h > p_h and p_h > h:
                            # 进一步使用线扫描验证1:1:3:1:1比例
                            if verify_finder_pattern_ratio(binary_img, (x + w // 2, y + h // 2), w):
                                potential_patterns.append({
                                    'center': (x + w // 2, y + h // 2),
                                    'size': max(w, h),
                                    'contour': cnt  # 保存最内层轮廓
                                })

    # 从潜在模式中筛选出3个最符合条件的查找器模式
    # 简单的策略：选取大小相似且距离合适的三个
    # 实际应用中可能需要更复杂的筛选和分组逻辑
    if len(potential_patterns) >= 3:
        # 尝试通过距离和大小筛选
        final_patterns = []
        # 可以根据实际情况设计更复杂的筛选逻辑，例如：
        # 1. 计算所有点对的距离，寻找近似等腰直角三角形
        # 2. 检查finder pattern的相对位置和角度

        # 粗略筛选：选择面积最大的几个正方形，然后尝试组成三元组
        sorted_patterns = sorted(potential_patterns, key=lambda p: p['size'], reverse=True)

        # 尝试寻找三个彼此之间距离合适的点
        if len(sorted_patterns) >= 3:
            # 这是一个非常简化的选择，实际中需要更复杂的几何验证
            # 比如计算两两距离，形成三角形，然后判断是否接近直角等
            # 为了避免过于复杂的算法，这里假设排名前三的如果验证通过，就是我们需要的
            # 实际产品级代码需要实现：计算所有 potential_patterns 的两两距离，
            # 找到构成直角三角形的三个点，并验证它们的相对大小和位置。
            # 为了避免引入过多复杂性，这里简化为直接从潜在模式中选择
            # 考虑到 pyzbar 后续会进行解码，这里找到“差不多”的就行
            final_patterns = sorted_patterns[:3]

            # 进一步验证这三个点是否构成近似的直角三角形
            if len(final_patterns) == 3:
                p1, p2, p3 = final_patterns[0]['center'], final_patterns[1]['center'], final_patterns[2]['center']

                # 计算边长平方
                d12_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                d13_sq = (p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2
                d23_sq = (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2

                # 找到最长边（斜边）
                sides_sq = sorted([d12_sq, d13_sq, d23_sq])

                # 检查是否接近直角三角形 a^2 + b^2 = c^2
                # 引入一定的容差
                tolerance_ratio = 0.1  # 10% 容差
                if abs((sides_sq[0] + sides_sq[1]) - sides_sq[2]) / sides_sq[2] > tolerance_ratio:
                    # 如果不是近似直角三角形，则可能不是正确的Finder Patterns
                    final_patterns = []

        if len(final_patterns) == 3:
            # 返回三个点的坐标 (x,y)
            return [np.array(p['center'], dtype=np.int32) for p in final_patterns]
    return []


def verify_finder_pattern_ratio(binary_img, center, size):
    """
    通过线扫描验证查找器模式的1:1:3:1:1黑白模块比例。
    从中心点沿四个方向（水平、垂直、对角线）进行扫描。
    """
    cx, cy = int(center[0]), int(center[1])
    half_size = int(size / 2)

    # 简单的边界检查
    if not (0 <= cx < binary_img.shape[1] and 0 <= cy < binary_img.shape[0]):
        return False

    # 定义扫描线步长，以中心点为原点
    # 沿着水平、垂直和两个对角线方向扫描
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # (dx, dy)

    # 期望的比例模式 (1:1:3:1:1)
    expected_ratios = [1, 1, 3, 1, 1]
    total_expected_sum = sum(expected_ratios)

    for dx, dy in directions:
        line_segments = []
        current_color = -1
        current_length = 0

        # 向一个方向扫描
        for step in range(-half_size, half_size):  # 扫描整个finder pattern区域
            px, py = cx + step * dx, cy + step * dy
            if not (0 <= px < binary_img.shape[1] and 0 <= py < binary_img.shape[0]):
                continue  # 超出图像边界

            pixel_val = binary_img[py, px]

            if current_color == -1:  # 第一次设置颜色
                current_color = pixel_val

            if pixel_val == current_color:
                current_length += 1
            else:
                line_segments.append(current_length)
                current_color = pixel_val
                current_length = 1

        if current_length > 0:  # 添加最后一个段
            line_segments.append(current_length)

        # 尝试匹配1:1:3:1:1模式
        if len(line_segments) >= 5:
            # 找到中间的3部分，通常是黑-白-黑或者白-黑-白
            # 需要考虑中间最宽的部分是3，两边是1

            # 简化的匹配：在扫描段中查找比例
            # 这部分需要更精确的匹配算法，例如滑动窗口或动态规划
            # 考虑到二维码查找器模式的特性，我们可以寻找连续的5个段，它们的长度比符合要求

            for i in range(len(line_segments) - 4):
                s1, s2, s3, s4, s5 = line_segments[i:i + 5]
                current_sum = s1 + s2 + s3 + s4 + s5

                # 计算实际比例
                ratios = [s1 / current_sum, s2 / current_sum, s3 / current_sum, s4 / current_sum, s5 / current_sum]

                # 期望比例的百分比
                expected_percentages = [x / total_expected_sum for x in expected_ratios]

                # 检查比例是否在容差范围内
                tolerance = 0.3  # 30% 容差

                match = True
                for j in range(5):
                    if abs(ratios[j] - expected_percentages[j]) > tolerance * expected_percentages[j]:
                        match = False
                        break

                if match:
                    return True  # 找到符合条件的模式

    return False  # 未找到符合条件的模式


def deskew_qr_code(img, qr_points, target_size=300):
    """
    根据QR码的四个角点进行透视校正
    :param img: 原始图像
    :param qr_points: QR码的四个角点，格式为 NumPy 数组 (4, 2)
                      例如 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param target_size: 校正后图像的目标尺寸
    :return: 校正后的图像，如果无法校正则返回None
    """
    if qr_points is None or qr_points.shape != (4, 2):
        return None

    # 需要根据三个finder pattern的相对位置确定第四个角点
    # 通常是左上，右上，左下。缺失的右下角点可以通过向量法估算。
    # 假设 qr_points 是三个查找器模式的中心点 (x, y)
    # OpenCV 的 QRCodeDetector 返回的 points_cv 是四个角点
    # 如果这里传入的是三个查找器模式点，需要先计算出第四个点
    # 这是一个简化，如果需要更精确的透视校正，需要明确四个角点

    if len(qr_points) == 3:
        # 假设三个点是 P0 (top-left), P1 (top-right), P2 (bottom-left)
        # 它们可以通过距离判断
        # P0-P1, P0-P2 是两条短边，P1-P2 是长边（斜边）
        # 寻找距离最远的两个点，它们是斜边上的点。第三个点是直角点。

        dists_sq = []
        points_map = {}
        for i in range(3):
            for j in range(i + 1, 3):
                p1 = qr_points[i]
                p2 = qr_points[j]
                dist_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                dists_sq.append((dist_sq, i, j))
                points_map[(i, j)] = (p1, p2)
                points_map[(j, i)] = (p2, p1)

        dists_sq.sort(key=lambda x: x[0], reverse=True)

        # 最长边对应的两个点是 P1, P2 (或反之)，第三个点是 P0
        idx_hyp_1, idx_hyp_2 = dists_sq[0][1], dists_sq[0][2]

        corner_indices = list(range(3))
        corner_indices.remove(idx_hyp_1)
        corner_indices.remove(idx_hyp_2)
        idx_top_left = corner_indices[0]  # 这个点是直角点，通常是左上角

        p_tl = qr_points[idx_top_left]
        p_tr = None
        p_bl = None

        # 确定 p_tr 和 p_bl
        # 比较 p_tl 到其他两点的斜率或方向
        other_points = [qr_points[idx_hyp_1], qr_points[idx_hyp_2]]

        # 通过叉积或简单的x,y比较确定
        if other_points[0][0] > other_points[1][0]:  # x更大的倾向于是右边
            p_tr = other_points[0]
            p_bl = other_points[1]
        else:
            p_tr = other_points[1]
            p_bl = other_points[0]

        # 估算第四个角点 (右下)
        # P_BR = P_TR + P_BL - P_TL
        p_br = p_tr + p_bl - p_tl

        src_pts = np.float32([p_tl, p_tr, p_br, p_bl])
    else:  # 如果传入已经是4个点
        src_pts = np.float32(qr_points)

    dst_pts = np.float32([[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_img = cv2.warpPerspective(img, M, (target_size, target_size))

    return warped_img


class QRScannerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("二维码识别与生成系统")
        self.setAcceptDrops(True)
        self.setStyleSheet(self.get_app_stylesheet())
        self.resize(1200, 700)
        # 设置图标
        self.setWindowIcon(QIcon("favicon.ico"))

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
        if max_dim > SCALE_THRESHOLD:
            scale_factor = SCALE_THRESHOLD / max_dim
            img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)),
                             interpolation=cv2.INTER_AREA)
            print(f"图像已缩放至: {img.shape[1]}x{img.shape[0]}")

        decoded_objs = []
        qr_roi_to_display = None  # 用于显示在二维码特写区域的图像

        # --- 2. 尝试使用自定义定位点查找（最优先） ---
        # 对原始图像进行预处理以获取二值图
        processed_img_for_custom = enhance_qr_image(img.copy())
        if processed_img_for_custom is not None:
            custom_points = find_finder_patterns_custom(processed_img_for_custom)

            if custom_points and len(custom_points) == 3:
                # 假设找到了三个finder patterns，需要计算出第四个点进行透视校正
                # 这里简化的deskew_qr_code已经处理了3个点的情况
                warped_img_custom = deskew_qr_code(img.copy(), np.array(custom_points))

                if warped_img_custom is not None and warped_img_custom.shape[0] > 0 and warped_img_custom.shape[1] > 0:
                    decoded_objs_custom = decode(warped_img_custom, symbols=[ZBarSymbol.QRCODE])
                    if decoded_objs_custom:
                        decoded_objs.extend(decoded_objs_custom)
                        qr_roi_to_display = warped_img_custom  # 使用自定义校正后的图像作为ROI
                        print("通过自定义定位和透视校正解码成功！")
                        # 如果识别成功，可能不需要继续尝试其他方法，但为了鲁棒性，可以继续。
                        # 这里我们只取第一个成功解码的结果
                        if decoded_objs:
                            obj = decoded_objs[0]
                            data = obj.data.decode("utf-8")
                            self.process_and_display_result(data, qr_roi_to_display)
                            return True

        # --- 3. 尝试使用pyzbar直接识别（原始图像和增强图像）---
        # 只指定识别QR码，提高速度
        if not decoded_objs:  # 如果自定义定位没有成功
            decoded_objs_raw = decode(img, symbols=[ZBarSymbol.QRCODE])
            if decoded_objs_raw:
                decoded_objs.extend(decoded_objs_raw)
                print("通过pyzbar原始图像解码成功！")

        if not decoded_objs:  # 如果原始图像未成功，尝试增强图像
            enhanced_img = enhance_qr_image(img.copy())
            if enhanced_img is not None:
                decoded_objs_enhanced = decode(enhanced_img, symbols=[ZBarSymbol.QRCODE])
                if decoded_objs_enhanced:
                    for obj in decoded_objs_enhanced:
                        if obj.data.decode("utf-8") not in [d.data.decode("utf-8") for d in decoded_objs]:
                            decoded_objs.append(obj)
                    print("通过pyzbar增强图像解码成功！")

        # --- 4. 尝试反色图像的自定义定位和pyzbar识别 ---
        if not decoded_objs:
            # 1. 反色
            inv_img = cv2.bitwise_not(img)

            # 尝试反色图像的自定义定位
            processed_img_inv_for_custom = enhance_qr_image(inv_img.copy())  # 注意，enhance_qr_image会再次二值化
            if processed_img_inv_for_custom is not None:
                custom_points_inv = find_finder_patterns_custom(processed_img_inv_for_custom)
                if custom_points_inv and len(custom_points_inv) == 3:
                    warped_img_custom_inv = deskew_qr_code(inv_img.copy(), np.array(custom_points_inv))
                    if warped_img_custom_inv is not None and warped_img_custom_inv.shape[0] > 0 and \
                            warped_img_custom_inv.shape[1] > 0:
                        decoded_objs_custom_inv = decode(warped_img_custom_inv, symbols=[ZBarSymbol.QRCODE])
                        if decoded_objs_custom_inv:
                            decoded_objs.extend(decoded_objs_custom_inv)
                            qr_roi_to_display = warped_img_custom_inv
                            print("通过反色图像的自定义定位和透视校正解码成功！")
                            if decoded_objs:
                                obj = decoded_objs[0]
                                data = obj.data.decode("utf-8")
                                self.process_and_display_result(data, qr_roi_to_display)
                                return True

            # 如果反色自定义定位未成功，尝试直接pyzbar解码反色图及增强反色图
            # 2. 转灰度再二值化 (enhance_qr_image已包含此步骤)
            # 3. 尝试直接解码二值化反色图
            decoded_inv = decode(inv_img, symbols=[ZBarSymbol.QRCODE])
            if decoded_inv:
                decoded_objs.extend(decoded_inv)
                print("通过pyzbar反色图像解码成功！")
            else:
                # 4. 对反色二值图做形态学+锐化，再试一次
                enhanced_inv = enhance_qr_image(inv_img.copy())
                if enhanced_inv is not None:
                    decoded_sharp_inv = decode(enhanced_inv, symbols=[ZBarSymbol.QRCODE])
                    if decoded_sharp_inv:
                        decoded_objs.extend(decoded_sharp_inv)
                        print("通过pyzbar增强反色图像解码成功！")

        # --- 5. 最后尝试使用OpenCV自带的QRCodeDetector（作为备选）---
        # 用户要求保留原初策略，即如果pyzbar未能识别，再尝试OpenCV自带的
        if not decoded_objs:
            qrDetector = cv2.QRCodeDetector()
            retval_cv, points_cv, straight_qrcode_cv = qrDetector.detectAndDecode(original_img)  # 使用原始大图或缩放后的图

            if retval_cv and points_cv is not None and len(points_cv) > 0:
                # opencv detectAndDecode 成功，retval_cv 是解码结果
                # points_cv 是四个角点
                # straight_qrcode_cv 是校正后的二维码图像
                # 这里的 straight_qrcode_cv 已经包含了校正，可以直接使用
                data = retval_cv
                # 为了统一，仍然使用pyzbar解码，确保结果一致性，但这里直接取retval_cv
                # 如果QRCodeDetector直接解码成功，直接使用其结果
                if data:
                    print("通过OpenCV QRCodeDetector解码成功！")
                    # 对于显示ROI，直接使用 straight_qrcode_cv
                    if straight_qrcode_cv is not None and straight_qrcode_cv.shape[0] > 0 and straight_qrcode_cv.shape[
                        1] > 0:
                        qr_roi_to_display = straight_qrcode_cv

                    self.process_and_display_result(data, qr_roi_to_display)
                    return True
                else:
                    print("OpenCV QRCodeDetector检测到二维码但未能解码。")
            else:
                print("OpenCV QRCodeDetector未能检测到二维码。")

        # --- 识别结果处理 ---
        if not decoded_objs:
            self.qr_label.clear()
            self.qr_label.setText("未识别到二维码")
            self.text_output.setText("未识别到二维码")
            print("所有方法均未能识别到二维码。")
            return False

        # 如果pyzbar识别成功 (优先取第一个结果)
        obj = decoded_objs[0]
        data = obj.data.decode("utf-8")

        # 尝试使用 pyzbar 返回的 obj.polygon 截取 ROI
        if obj.polygon is not None:
            if isinstance(obj.polygon, np.ndarray):
                pts_np = obj.polygon
            else:
                pts_np = np.array([(p.x, p.y) for p in obj.polygon], dtype=np.int32)

            # 考虑图像缩放，将pyzbar返回的ROI坐标转换回原始图像坐标
            if 'scale_factor' in locals():
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
                qr_roi_to_display = original_img[y_clip_start:y_clip_end, x_clip_start:x_clip_end]
                if qr_roi_to_display.shape[0] == 0 or qr_roi_to_display.shape[1] == 0:
                    qr_roi_to_display = None  # ROI无效
            else:
                print(f"警告: obj.polygon 计算出的 ROI 经过裁剪和扩展后宽度或高度为零: w={clipped_w}, h={clipped_h}。")

        # 如果 pyzbar 未能提供有效的 ROI 图像，则使用之前可能生成的校正图
        if qr_roi_to_display is None and 'straight_qrcode_cv' in locals() and straight_qrcode_cv is not None and \
                straight_qrcode_cv.shape[0] > 0:
            qr_roi_to_display = straight_qrcode_cv

        self.process_and_display_result(data, qr_roi_to_display)
        return True

    def process_and_display_result(self, data, qr_roi_to_display=None):
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