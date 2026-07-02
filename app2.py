import sys
import cv2
import re
import webbrowser
import numpy as np
import io
import os
import time
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QMessageBox, QTextEdit, QListWidget, QGroupBox, QShortcut,
    QTabWidget, QLineEdit
)
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PyQt5.QtCore import QTimer, Qt, QCoreApplication, QBuffer, QIODevice
from pyzbar.pyzbar import decode, ZBarSymbol

# 适应高DPI设备
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

# 判断字符串是否为网址
def is_url(text):
    return re.match(r'https?://[\w./?=&-]+', text)

# 图像预处理增强
def enhance_qr_image(img):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应阈值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学操作：通过闭运算连接断裂线条和填充小孔
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 锐化
    sharpened = cv2.filter2D(morphed, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    return sharpened


# 根据QR码的四个角点进行透视校正
def deskew_qr_code(img, qr_points):
    # 传入参数合法性检测
    if qr_points is None or qr_points.shape != (4, 2):
        return None
    src_pts = np.float32(qr_points)
    target_size = 300
    # 定义变换后二维码四个角点在300x300图像中的位置
    dst_pts = np.float32([[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]])
    # 计算透视变换矩阵M
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # 应用透视变换矩阵M
    warped_img = cv2.warpPerspective(img, M, (target_size, target_size))
    return warped_img


class QRScannerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("二维码识别与生成系统")
        self.setAcceptDrops(True)
        # 应用样式表
        self.setStyleSheet(self.get_app_stylesheet())
        self.resize(1200, 700)
        # 设置图标
        self.setWindowIcon(QIcon("favicon.ico"))
        # 上次识别的内容
        self.last_data = ""
        # 用于存储上一次在 text_output 中显示的数据，区分当前识别到的数据和已经显示的数据
        self.last_displayed_data = ""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self._generated_qr_img_pil = None

        # 当前识别模式
        self.current_scan_mode = "idle"  # idle, camera, file

        # 摄像头重复识别的间隔时间
        self.camera_scan_interval = 2  # 2秒内不重复提示
        self.last_camera_scan_time = 0  # 上次摄像头成功识别的时间戳

        # 初始化微信扫码引擎
        self.wechat_qr_detector = None
        try:
            # 微信引擎的小型模型文件
            prototxt_path = "models/detect.prototxt"
            caffemodel_path = "models/detect.caffemodel"
            sr_prototxt_path = "models/sr.prototxt"
            sr_caffemodel_path = "models/sr.caffemodel"

            if all(os.path.exists(p) for p in [prototxt_path, caffemodel_path, sr_prototxt_path, sr_caffemodel_path]):
                # 创建微信二维码识别引擎的对象
                self.wechat_qr_detector = cv2.wechat_qrcode_WeChatQRCode(
                    prototxt_path, caffemodel_path, sr_prototxt_path, sr_caffemodel_path
                )
                print("OpenCV WeChat QRCode detector 初始化成功")
            else:
                QMessageBox.warning(self, "模型文件缺失",
                                    "未找到OpenCV微信扫码模型文件。请确保以下文件存在于'models/'目录下：\n"
                                    "detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel\n"
                                    "否则将退回使用Pyzbar和OpenCV自带的识别器。")
        except AttributeError:
            QMessageBox.warning(self, "OpenCV版本问题",
                                "您的OpenCV版本可能不支持wechat_qrcode模块，或未安装opencv-contrib-python。将退回使用Pyzbar和OpenCV自带的识别器。")
            self.wechat_qr_detector = None
        except Exception as e:
            QMessageBox.warning(self, "微信扫码引擎初始化失败",
                                f"初始化OpenCV微信扫码引擎时发生错误: {e}\n将退回使用Pyzbar和OpenCV自带的识别器。")
            self.wechat_qr_detector = None

        # 进一步初始化UI界面
        self.init_ui()
        # Ctrl+V 粘贴图片快捷键
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_image)

    def get_app_stylesheet(self):
        # 统一的样式表
        return """
            QWidget {
                background-color: #f8f9fa;
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 16px;
                color: #343a40;
            }
            QLabel {
                color: #495057;
            }
            QGroupBox {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 15px;
                font-size: 18px;
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
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 16px;
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
                padding: 10px;
                background-color: #ffffff;
                font-size: 15px;
            }
            QListWidget::item {
                padding: 6px;
                font-size: 15px;
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
                padding: 10px 20px;
                min-width: 90px;
                font-size: 15px;
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

        # 左侧是主预览与识别结果
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

        # 右侧：功能区 (Tab Widget)
        right_panel_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)

        # 识别页面
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

        # 生成页面
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

    # 功能方法
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
            self.last_displayed_data = ""  # 清空显示数据
            QMessageBox.information(self, "操作成功", "识别历史已清空。")

    def open_history_item(self, item):
        data = item.text()
        if is_url(data):
            reply = QMessageBox.question(self, "打开网址", f"是否打开历史记录中的网址：{data}？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                webbrowser.open(data)
        else:
            self.text_output.setText(data)
            self.last_displayed_data = data  # 更新上次显示的数据
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
            self.current_scan_mode = "idle"  # 摄像头关闭，模式设为空闲
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头，请检查设备连接。")
                self.cap = None
                return
            # 30ms一次
            self.timer.start(30)
            self.camera_btn.setText("🛑 关闭摄像头")
            self.current_scan_mode = "camera"  # 摄像头打开，模式设为摄像头
            self.last_data = ""  # 切换模式，清空上次数据，确保新模式下的首次识别不会被过滤
            self.last_displayed_data = ""  # 清空显示数据，确保新模式下的首次显示
            self.last_camera_scan_time = 0  # 重置摄像头扫描时间

    def capture_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.display_image(frame)
            self.decode_qr(frame)  # 在摄像头模式下调用decode_qr
        else:
            self.image_label.setText("无法读取摄像头画面")

    def load_image(self):
        # 如果摄像头开启，先关闭它
        if self.cap:
            self.toggle_camera()

        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.current_scan_mode = "file"  # 设置为文件模式
            self.last_data = ""  # 切换模式，清空上次数据
            self.last_displayed_data = ""  # 清空显示数据，确保新文件下的首次显示
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def decode_qr(self, img):
        decoded_data = ""
        decoded_points = None
        roi_img = None

        try:
            # 首次尝试OpenCV自带的QRCodeDetector
            qrDetector = cv2.QRCodeDetector()
            retval_cv, points_cv, straight_qrcode_cv = qrDetector.detectAndDecode(img)
            if retval_cv and points_cv is not None and len(points_cv) > 0:
                decoded_data = retval_cv
                decoded_points = points_cv[0]
                roi_img = straight_qrcode_cv
                print(f"OpenCV自带识别器 (原图) 识别成功: {decoded_data}")  # 调试信息
            else:
                # 如果原图识别失败，尝试对图片反色后再次进行OpenCV自带的QRCodeDetector识别
                inv_img = cv2.bitwise_not(img)
                retval_cv_inv, points_cv_inv, straight_qrcode_cv_inv = qrDetector.detectAndDecode(inv_img)
                if retval_cv_inv and points_cv_inv is not None and len(points_cv_inv) > 0:
                    decoded_data = retval_cv_inv
                    decoded_points = points_cv_inv[0]
                    roi_img = straight_qrcode_cv_inv
                    print(f"OpenCV自带识别器 (反色图) 识别成功: {decoded_data}")  # 调试信息
                elif points_cv is not None and len(points_cv) > 0:
                    # 原始图检测到点但未解码，尝试透视校正
                    warped_img = deskew_qr_code(img, points_cv[0])
                    if warped_img is not None and warped_img.shape[0] > 0 and warped_img.shape[1] > 0:
                        decoded_objs_warped = decode(warped_img, symbols=[ZBarSymbol.QRCODE])
                        if decoded_objs_warped:
                            decoded_data = decoded_objs_warped[0].data.decode("utf-8")
                            decoded_points = np.array([(p.x, p.y) for p in decoded_objs_warped[0].polygon],
                                                      dtype=np.int32)
                            roi_img = warped_img
                            print(f"OpenCV检测+Pyzbar校正识别成功: {decoded_data}")  # 调试信息
                elif points_cv_inv is not None and len(points_cv_inv) > 0:
                    # 反色图检测到点但未解码，尝试透视校正
                    warped_img_inv = deskew_qr_code(inv_img, points_cv_inv[0])
                    if warped_img_inv is not None and warped_img_inv.shape[0] > 0 and warped_img_inv.shape[1] > 0:
                        decoded_objs_warped_inv = decode(warped_img_inv, symbols=[ZBarSymbol.QRCODE])
                        if decoded_objs_warped_inv:
                            decoded_data = decoded_objs_warped_inv[0].data.decode("utf-8")
                            decoded_points = np.array([(p.x, p.y) for p in decoded_objs_warped_inv[0].polygon],
                                                      dtype=np.int32)
                            roi_img = warped_img_inv
                            print(f"OpenCV检测(反色)+Pyzbar校正识别成功: {decoded_data}")  # 调试信息
            # 更新二维码特写区域（如果已识别到decoded_data）
            if decoded_data and decoded_points is not None and roi_img is None:
                # 原始图或反色图识别成功，但roi_img未从straight_qrcode_cv或straight_qrcode_cv_inv中获取
                # 则手动从原图或反色图中裁剪ROI
                target_img = img if not (decoded_data and decoded_points is points_cv_inv[0]) else inv_img
                pts_int = np.intp(decoded_points)
                x, y, w, h = cv2.boundingRect(pts_int)
                img_height, img_width = target_img.shape[:2]

                x_clip_start = max(0, x)
                y_clip_start = max(0, y)
                x_clip_end = min(img_width, x + w)
                y_clip_end = min(img_height, y + h)

                clipped_w = x_clip_end - x_clip_start
                clipped_h = y_clip_end - y_clip_start

                if clipped_w > 0 and clipped_h > 0:
                    roi_img = target_img[y_clip_start:y_clip_end, x_clip_start:x_clip_end]
                else:
                    print(f"警告: 计算 ROI 时裁剪后宽度或高度为零: w={clipped_w}, h={clipped_h}。")

            # 接着尝试使用OpenCV微信扫码引擎
            if self.wechat_qr_detector and not decoded_data:
                try:
                    results, points = self.wechat_qr_detector.detectAndDecode(img)
                    if results and results[0]:
                        decoded_data = results[0]
                        decoded_points = points[0]
                        print(f"微信扫码引擎识别成功: {decoded_data}") # 调试信息
                        if decoded_points is not None and len(decoded_points) > 0:
                            pts_int = np.intp(decoded_points)
                            x, y, w, h = cv2.boundingRect(pts_int)
                            roi_img = img[max(0, y):min(img.shape[0], y + h), max(0, x):min(img.shape[1], x + w)]
                except Exception as e:
                    print(f"微信扫码引擎处理错误: {e}")
                    decoded_data = ""

            # 如果以上所有方法都未识别到，尝试pyzbar
            if not decoded_data:
                decoded_objs_raw = decode(img, symbols=[ZBarSymbol.QRCODE])
                if decoded_objs_raw:
                    decoded_data = decoded_objs_raw[0].data.decode("utf-8")
                    decoded_points = np.array([(p.x, p.y) for p in decoded_objs_raw[0].polygon], dtype=np.int32)
                    print(f"Pyzbar (原始图) 识别成功: {decoded_data}") # 调试信息
                else:
                    enhanced_img = enhance_qr_image(img)
                    decoded_objs_enhanced = decode(enhanced_img, symbols=[ZBarSymbol.QRCODE])
                    if decoded_objs_enhanced:
                        decoded_data = decoded_objs_enhanced[0].data.decode("utf-8")
                        decoded_points = np.array([(p.x, p.y) for p in decoded_objs_enhanced[0].polygon],
                                                  dtype=np.int32)
                        print(f"Pyzbar (增强图) 识别成功: {decoded_data}") # 调试信息
                    else:
                        inv = cv2.bitwise_not(img)
                        gray_inv = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
                        _, bin_inv = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        decoded_inv = decode(bin_inv, symbols=[ZBarSymbol.QRCODE])
                        if decoded_inv:
                            decoded_data = decoded_inv[0].data.decode("utf-8")
                            decoded_points = np.array([(p.x, p.y) for p in decoded_inv[0].polygon], dtype=np.int32)
                            print(f"Pyzbar (反色二值图) 识别成功: {decoded_data}") # 调试信息
                        else:
                            kernel = np.ones((5, 5), np.uint8)
                            morphed_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel)
                            sharpened_inv = cv2.filter2D(morphed_inv, -1,
                                                         np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
                            decoded_sharp_inv = decode(sharpened_inv, symbols=[ZBarSymbol.QRCODE])
                            if decoded_sharp_inv:
                                decoded_data = decoded_sharp_inv[0].data.decode("utf-8")
                                decoded_points = np.array([(p.x, p.y) for p in decoded_sharp_inv[0].polygon],
                                                          dtype=np.int32)
                                print(f"Pyzbar (反色二值增强图) 识别成功: {decoded_data}") # 调试信息

            # 处理识别结果
            if not decoded_data:
                # 如果没有识别到，清空特写区域和文本输出，并返回False
                self.qr_label.clear()
                self.qr_label.setText("未识别到二维码")
                self.text_output.setText("未识别到二维码")
                self.last_displayed_data = ""  # 清空上次显示数据
                return False

            # 更新二维码ROI区域
            # 无论是否重复，特写区域都需要实时更新，保持视觉反馈
            if roi_img is None and decoded_points is not None:
                pts_int = np.intp(decoded_points)
                x, y, w, h = cv2.boundingRect(pts_int)
                img_height, img_width = img.shape[:2]

                x_clip_start = max(0, x)
                y_clip_start = max(0, y)
                x_clip_end = min(img_width, x + w)
                y_clip_end = min(img_height, y + h)

                clipped_w = x_clip_end - x_clip_start
                clipped_h = y_clip_end - y_clip_start

                if clipped_w > 0 and clipped_h > 0:
                    roi_img = img[y_clip_start:y_clip_end, x_clip_start:x_clip_end]
                else:
                    print(f"警告: 计算 ROI 时裁剪后宽度或高度为零: w={clipped_w}, h={clipped_h}。")

            if roi_img is not None and roi_img.shape[0] > 0 and roi_img.shape[1] > 0:
                display_qr_img = cv2.resize(roi_img, (250, 250))
                if len(display_qr_img.shape) == 3:
                    display_qr_img = cv2.cvtColor(display_qr_img, cv2.COLOR_BGR2GRAY)

                roi_qimg = QImage(display_qr_img.data, display_qr_img.shape[1], display_qr_img.shape[0],
                                  display_qr_img.strides[0], QImage.Format_Grayscale8)
                self.qr_label.setPixmap(QPixmap.fromImage(roi_qimg))
            else:
                self.qr_label.clear()
                self.qr_label.setText("无法获取二维码区域")

            # 处理重复二维码逻辑和提示
            # 只有当识别到的数据与上次显示的数据不同时，才更新text_output和考虑弹窗
            should_process_data = True
            if self.current_scan_mode == "camera":
                # 摄像头模式下，如果数据与上次相同，且在间隔时间内，则不进行进一步处理（除了ROI显示）
                if decoded_data == self.last_data and (
                        time.time() - self.last_camera_scan_time < self.camera_scan_interval):
                    should_process_data = False
                else:
                    self.last_camera_scan_time = time.time()  # 更新上次扫描时间

            if should_process_data:
                # 不管哪种模式，如果数据和上次显示的不同，就更新text_output
                if decoded_data != self.last_displayed_data:
                    self.text_output.setText(decoded_data)
                    self.last_displayed_data = decoded_data

                    # 添加到历史记录
                    existing = [self.history_list.item(i).text() for i in range(self.history_list.count())]
                    if decoded_data not in existing:
                        self.history_list.addItem(decoded_data)

                    # 提示网址打开
                    if is_url(decoded_data):
                        # 只有在本地文件模式，或者摄像头模式且是首次识别到该网址时，才提示
                        if self.current_scan_mode == "file" or (self.current_scan_mode == "camera" and decoded_data != self.last_data):  # 摄像头首次识别到新网址才弹窗
                            reply = QMessageBox.question(self, "打开网址", f"识别到网址：{decoded_data}\n是否打开？",
                                                         QMessageBox.Yes | QMessageBox.No)
                            if reply == QMessageBox.Yes:
                                webbrowser.open(decoded_data)
                        else:  # 摄像头模式下重复识别到相同网址，不弹窗，但可以打印日志
                            print(f"摄像头重复识别到网址：{decoded_data} (不弹窗)")
                    else:  # 非网址内容，在本地文件模式下才提示
                        if self.current_scan_mode == "file":
                            QMessageBox.information(self, "识别成功", f"识别内容：\n{decoded_data}")
                        else:
                            print(f"摄像头识别到非网址内容：{decoded_data} (不弹窗)")

                self.last_data = decoded_data  # 更新last_data，用于下次重复判断

            return True

        except Exception as e:
            print(f"识别异常： {e}")
            self.text_output.setText("二维码解析失败")
            self.qr_label.clear()
            # 清空上次显示数据
            self.last_displayed_data = ""
            return False

    # 重写拖拽图片识别事件
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # 拖放后的具体操作函数
    def dropEvent(self, event: QDropEvent):
        # 如果摄像头开启，先关闭它
        if self.cap:
            self.toggle_camera()
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self.current_scan_mode = "file"  # 设置为文件模式
                self.last_data = ""  # 清空上次数据
                self.last_displayed_data = ""  # 清空显示数据
                image = self.imread_unicode(file_path)
                self.display_image(image)
                success = self.decode_qr(image)
                if not success:
                    QMessageBox.information(self, "识别失败", "未识别到二维码，请尝试其他图片。")
                break  # 只处理第一个拖拽的文件

    # 粘贴后的具体操作函数
    def paste_image(self):
        # 如果摄像头开启，先关闭它
        if self.cap:
            self.toggle_camera()
        # 创建QT的剪贴板对象
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
        # 再检查文件URL（例如从文件管理器复制的文件）
        elif mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if os.path.isfile(path):
                    img_np = self.imread_unicode(path)
                    break
        # 如果识别到粘贴的图片是合法的
        if img_np is not None:
            self.current_scan_mode = "file"  # 设置为文件模式
            self.last_data = ""  # 清空上次数据
            self.last_displayed_data = ""  # 清空显示数据
            self.display_image(img_np)
            if not self.decode_qr(img_np):
                QMessageBox.information(self, "识别失败", "未识别到二维码，请尝试其他图片。")
        else:
            QMessageBox.warning(self, "无图像", "剪贴板中没有可用的图片数据！")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QRScannerApp()
    window.show()
    sys.exit(app.exec_())