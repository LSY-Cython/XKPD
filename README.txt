1.运行环境：
opencv-python 3.4.2.17
pillow 8.2.0
pyyaml 5.4.1
pyqt5 5.15.2
pyqt5-tools 5.15.2
pytorch 1.6.0
pywin32 300
numpy 1.19.2

2.主程序入口代码：MainApp.py    界面设计代码：MainForm.py

3.技术要点：
(1)鼠标滚轮控制图像缩放比例系数
(2)鼠标左键单击控制点位拖动，注意屏幕坐标系、像素坐标系和界面主窗口坐标系对齐
(3)图像资源的统一加载与格式转换：opencv是np.ndarray，pyqt5是QImage
(4)界面显示参数自适应：图像显示容器宽高比自适应图像尺寸，控件大小自适应屏幕分辨率