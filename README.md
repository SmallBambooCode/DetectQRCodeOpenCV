# QR Code Detection System based on OpenCV

基于OpenCV的QR码检测系统的设计与实现

## 简介
本项目为数字图像处理综合实训项目，基于 Python 开发，结合 OpenCV 图像处理算法与 PyQt5 图形界面，实现 QR 码的检测、校正、解码与生成功能。采用多引擎级联识别策略，在保证常规场景识别效率的同时，兼顾模糊、倾斜等复杂场景下的鲁棒性。

## 技术栈
- 开发语言：Python
- 图像处理：OpenCV
- 图形界面：PyQt5
- 条码解码：Pyzbar
- 二维码生成：qrcode
- 增强识别：OpenCV 微信扫码引擎（可选）

## 核心功能
### 二维码识别
- 支持本地图片加载、摄像头实时扫描、剪贴板粘贴、文件拖拽四种输入方式
- 多级识别回退策略：OpenCV 原生检测器 → 透视校正 + Pyzbar 解码 → 微信扫码引擎 → Pyzbar 多级图像增强兜底
- 自动透视校正，将倾斜变形的 QR 码校正为标准正视图
- 识别结果实时展示、历史记录管理、识别到网址可一键跳转
- 摄像头模式下内置重复识别过滤机制，避免重复弹窗

### 二维码生成
- 支持自定义文本/网址生成 QR 码
- 生成结果实时预览，支持导出为 PNG/JPG/BMP 格式图片

## 快速开始
### 安装依赖
```bash
pip install opencv-python opencv-contrib-python PyQt5 pyzbar qrcode[pil] pillow numpy
```

### 运行程序
```bash
python app.py
```

### 可选配置
若需启用微信扫码引擎增强识别能力，请将 `detect.prototxt`、`detect.caffemodel`、`sr.prototxt`、`sr.caffemodel` 模型文件放入项目 `models/` 目录下。

## 运行截图

<img width="2560" height="1368" alt="image" src="https://github.com/user-attachments/assets/9102efe3-f4b2-4735-a8f3-dda55a684314" />
<img width="397" height="213" alt="image" src="https://github.com/user-attachments/assets/13ab4aa6-b391-4b88-a9b9-67209b4ad782" />
<img width="2560" height="1368" alt="image" src="https://github.com/user-attachments/assets/66aae3dd-c9dd-41d9-b683-b3286ec3a982" />
<img width="2560" height="1368" alt="image" src="https://github.com/user-attachments/assets/3967f3dd-34ac-461c-9a7a-a5f6570504ce" />

## 说明
本项目为图像处理综合实训课程实训项目，核心围绕数字图像处理经典算法实现，融合多引擎策略提升实际场景识别效果。代码仅供学习交流，不建议直接抄袭使用！
