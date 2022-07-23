import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QFileDialog, QMenu, QAction, QInputDialog, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from MainForm import Ui_Form
from GeometryForm import Ui_GeometryForm
from slot import load_model, draw_curve
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
from inference import handle_single

# 多继承方法实现界面与逻辑分离
class PointDetection(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.model = load_model()
        self.initImage()
        self.initSrc()
        # 声明在图片显示容器中创建右键快捷菜单
        self.box_image.setContextMenuPolicy(Qt.DefaultContextMenu)
    def initImage(self):
        # self.image_w = int(self.box_image.width() * 0.75)  # box_image.geometry固定不变, label_image.geometry在滚轮缩放中实时变化
        # self.image_h = int(self.box_image.height() * 0.84)
        self.left_click = False
        self.startPos = None
        self.label_x = 0
        self.label_y = 0
        self.label_w = int(self.screenWidth * 0.91)
        self.label_h = int(self.screenHeight)
        self.dx = 0
        self.dy = 0
        self.x_start = int(self.screenWidth * 0.09)
        self.x_end = int(self.screenWidth)
        self.select_id = None  # 单击选中的关键点索引
        self.delete_id = []  # 右键选中的待删除关键点索引
        self.alter_id = []  # 右键选中的待修正关键点索引
        self.alter_gt = []  # 右键选中的待修正关键点校准坐标
        self.move_id = []  # 右键选中的待移动关键点索引
        self.move_gt = []  # 右键选中的待移动关键点实时坐标
        self.align_flag = False
        self.move_flag = False
        self.label_image.setGeometry(QtCore.QRect(0, 0, int(self.screenWidth * 0.91), self.screenHeight))
    def initSrc(self):
        self.model_path = ''  # 模型存储绝对路径
        self.image_path = ''  # 图片存储绝对路径
        self.config_path = ''  # 曲线配置文件存储绝对路径
        self.image_src = None  # 加载图像资源统一为QImage格式
        self.raw_img = None  # 原始图像(np.ndarray格式)
        self.point_img = None  # 标注关键点后的图像(np.ndarray格式)
        self.curve_img = None  # 绘制多项式曲线后的图像(np.ndarray格式)
        self.repaint_img = None  # 编辑修改关键点后的图像(np.ndarray格式)
        self.kps = None  # 关键点坐标集合(相对于图像坐标系)
        self.kps_adjust = None  # 关键点实时坐标集合(相对于图像坐标系)
        self.kps_pos = None  # 关键点坐标集合(相对于窗口坐标系)
        self.image_shape = None  # 原始图像尺寸
    def btnOpenModel(self):
        try:
           model_path, _ = QFileDialog.getOpenFileName(self, caption="选择模型", directory="", filter="*.pt *.pth")
           if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
               self.model = load_model(self.model_path)
        except:
           QMessageBox.warning(self, "警告", "深度学习模型加载错误, 请重新选择！")
    def btnOpenImage(self):
        self.initImage()
        self.initSrc()
        try:
            self.image_path, _ = QFileDialog.getOpenFileName(self, caption="选择图片", directory="", filter="*.jpg *.png")
            self.raw_img = cv2.cvtColor(np.asarray(Image.open(self.image_path)), cv2.COLOR_RGB2BGR)
            self.image_shape = self.raw_img.shape
            self.image_w = int(self.box_image.width() * 0.7)
            self.image_h = int((self.image_w*self.image_shape[0])/self.image_shape[1])
            # with open(self.image_path, 'rb') as f:
            #     self.raw_img = QImage.fromData(f.read())
            self.label_image.setPixmap(QPixmap.fromImage(self.qpix_img(self.raw_img)).scaled(self.image_w, self.image_h))
            self.image_src = self.qpix_img(self.raw_img) 
        except:
            QMessageBox.warning(self, "警告", "输入图片加载错误, 请重新选择！")
    def btnKeyPoint(self):
        if self.raw_img is not None:
            self.point_img, kps = handle_single(self.raw_img, self.model, 'cpu')
            self.repaint_img = self.point_img.copy()
            self.image_shape = self.point_img.shape
            self.kps = kps.astype(np.int32)  # np.array([[a,b],[c,d],...,[x,y]])
            self.kps_adjust = self.kps.copy()
            self.image_src = self.qpix_img(self.point_img)
            self.label_image.setPixmap(QPixmap.fromImage(self.image_src).scaled(self.image_w, self.image_h))
        else:
            QMessageBox.warning(self, "警告", "请先打开图片！")
    def btnFitCurve(self):
        if self.repaint_img is not None and self.align_flag is False:
            if self.config_path.endswith(".json"):
                self.curve_img = draw_curve(self.repaint_img, self.kps_adjust, self.config_path)
            else:
                self.curve_img = draw_curve(self.repaint_img, self.kps_adjust, "curve_config.json")
            self.image_src = self.qpix_img(self.curve_img)
            self.label_image.setPixmap(QPixmap.fromImage(self.image_src).scaled(self.image_w, self.image_h))
        elif self.align_flag is True:
            QMessageBox.warning(self, "警告", "请先退出点位编辑！")
        elif self.repaint_img is None:
            QMessageBox.warning(self, "警告", "请先生成关键点检测图片！")
    def btnOpenConfig(self):
        try:
           self.config_path, _ = QFileDialog.getOpenFileName(self, caption="选择配置文件", directory="", filter="*.json")
        except:
           QMessageBox.warning(self, "警告", "曲线配置文件加载错误, 请重新选择！")
    def btnAlignPoint(self):
        if self.point_img is not None:
            self.initImage()
            self.align_flag = True
            self.kps_pos = self.pix2pos(self.kps)
            self.kps_adjust = self.kps.copy()
            self.label_image.setPixmap(QPixmap.fromImage(self.qpix_img(self.point_img)).scaled(self.image_w, self.image_h))
        else:
            QMessageBox.warning(self, "警告", "请先生成关键点检测图片！")
    def btnExitAlter(self):
        self.align_flag = False
    def btnMeasureGeometry(self):
        text, state = QInputDialog.getText(self, "几何数据测量", "请输入三点编号(A B C)")
        if state and self.point_img is not None:
            try:
                gt_a = self.kps_adjust[int(text.split(' ')[0])]
                gt_b = self.kps_adjust[int(text.split(' ')[1])]
                gt_c = self.kps_adjust[int(text.split(' ')[2])]
                self.sub_win = GeometryForm()  # 子窗口实例为主窗口类的变量
                angle_a = self.angle_formula(gt_b - gt_a, gt_c - gt_a)
                angle_b = self.angle_formula(gt_a - gt_b, gt_c - gt_b)
                angle_c = self.angle_formula(gt_a - gt_c, gt_b - gt_c)
                lab = np.linalg.norm(gt_a - gt_b)
                lbc = np.linalg.norm(gt_b - gt_c)
                lac = np.linalg.norm(gt_a - gt_c)
                da = self.distance_formular(gt_b - gt_a, gt_c - gt_a, gt_b - gt_c)
                db = self.distance_formular(gt_a - gt_b, gt_c - gt_b, gt_c - gt_a)
                dc = self.distance_formular(gt_a - gt_c, gt_b - gt_c, gt_b - gt_a)
                self.sub_win.lineEdit_anglea.setText(str(angle_a))
                self.sub_win.lineEdit_angleb.setText(str(angle_b))
                self.sub_win.lineEdit_anglec.setText(str(angle_c))
                self.sub_win.lineEdit_lab.setText(str(np.around(lab, 2)))
                self.sub_win.lineEdit_lbc.setText(str(np.around(lbc, 2)))
                self.sub_win.lineEdit_lac.setText(str(np.around(lac, 2)))
                self.sub_win.lineEdit_da.setText(str(da))
                self.sub_win.lineEdit_db.setText(str(db))
                self.sub_win.lineEdit_dc.setText(str(dc))
                self.sub_win.show()
            except:
                QMessageBox.critical(self, "错误", "关键点编号输入格式错误！")
        elif self.point_img is None:
            QMessageBox.warning(self, "警告", "请先生成关键点检测图片！")
    def btnSaveData(self):
        if self.curve_img is not None and self.align_flag is False:
            conn = sqlite3.connect('xkps.db')
            cursor = conn.cursor()
            insert_sql = '''INSERT INTO info VALUES(?,?,?,?,?)'''  # name text, id text, points blob, image blob, shape blob
            name = self.image_path.split('/')[-1].strip(".JPG")
            id = self.image_path.split('/')[-1].strip(".JPG")
            points = self.kps_adjust
            # image = cv2.cvtColor(np.asarray(Image.open(self.image_path)), cv2.COLOR_RGB2BGR)
            image = self.raw_img  # 保存原图, 便于info数据修改
            shape = np.array(self.image_shape)
            insert_data = (name, id, points, image, shape)
            cursor.execute(insert_sql, insert_data)
            conn.commit()
            QMessageBox.information(self, "提示", f"{id}-{name}已保存！")
            print(f"{datetime.datetime.now()} 数据库info已保存更新")
            conn.close()
        elif self.align_flag is True:
            QMessageBox.warning(self, "警告", "请先退出点位编辑！")
        elif self.curve_img is None:
            QMessageBox.warning(self, "警告", "请先绘制关键点曲线！")
    def btnQueryData(self):
        text, state = QInputDialog.getText(self, "病例数据查询", "请输入病人姓名或编号")
        if state:
            self.initImage()
            self.initSrc()
            conn = sqlite3.connect('xkps.db')
            cursor = conn.cursor()
            select_sql = '''SELECT points, image, shape FROM info WHERE name=? OR id=?'''
            select_data = (text, text)
            select_results = cursor.execute(select_sql, select_data).fetchall()
            if len(select_results) != 0:
                record = select_results[-1]
                # 导出数据后重新初始化加载资源
                kps = np.frombuffer(record[0], dtype=np.int32)
                self.image_path = text + ".JPG"
                self.kps = kps.reshape((int(len(kps)/2), 2))
                self.kps_adjust = self.kps.copy()
                self.image_shape = tuple(np.frombuffer(record[2], dtype=np.int32))
                self.raw_img = np.frombuffer(record[1], dtype=np.uint8).reshape(self.image_shape)
                self.point_img = self.repaint_point(self.raw_img.copy())
                self.repaint_img = self.point_img.copy()
                self.curve_img = draw_curve(self.point_img, self.kps, "curve_config.json")
                self.image_src = self.qpix_img(self.curve_img)  # 当前加载图像资源指向已生成曲线的图片
                self.image_w = int(self.box_image.width() * 0.7)
                self.image_h = int((self.image_w * self.image_shape[0]) / self.image_shape[1])
                self.label_image.setPixmap(QPixmap.fromImage(self.image_src).scaled(self.image_w, self.image_h))
                QMessageBox.information(self, "提示", f"{text}-{text}已打开！")
                print(f"{datetime.datetime.now()} 数据库info已查询更新")
            else:
                QMessageBox.warning(self, "警告", "数据库info中不存在该病例！")
            conn.close()
    def contextMenuEvent(self, e):
        try:
            self.right_menu = QMenu(self)  # 菜单对象
            # 创建菜单选项对象
            self.delete_action = QAction(QIcon("icon/delete.png"), u"删除点位", self)
            self.right_menu.addAction(self.delete_action)
            self.alter_action = QAction(QIcon("icon/alter.png"), u"修改坐标", self)
            self.right_menu.addAction(self.alter_action)
            # 动作触发时连接到槽函数
            self.delete_action.triggered.connect(self.deletePoint)
            self.alter_action.triggered.connect(self.alterPoint)
            pos = [e.pos().x(), e.pos().y()]
            # print(f"当前屏幕cursor坐标为：{pos}")
            if self.point_img is not None and self.align_flag is True and self.select_point(pos):  # 鼠标在关键点上右击时才能显示菜单
                self.right_menu.popup(QPoint(pos[0]+20, pos[1]+40))  # exec_和popup都可以, 移动关键点到右下方显示
        except:
            QMessageBox.warning(self, "警告", "频繁进行右键单击操作！")
    def deletePoint(self):
        self.delete_id.append(self.select_id)
        self.repaint_img = self.repaint_image()
    def alterPoint(self):
        text, state = QInputDialog.getText(self, "精校坐标", "请输入整数坐标值(x y)")
        if state:
            try:
                input_kpx = int(text.split(" ")[0])
                input_kpy = int(text.split(" ")[1])
                if self.select_id in self.move_id:
                    self.move_gt.remove(self.move_gt[self.move_id.index(self.select_id)])
                    self.move_id.remove(self.select_id)  # 先移除修改点索引对应坐标值
                if self.select_id not in self.alter_id:  # 单击选中点可能在移动坐标点但不在精校坐标点中, 故不能用elif
                    self.alter_id.append(self.select_id)
                    self.alter_gt.append([input_kpx, input_kpy])
                if self.select_id in self.alter_id:
                    self.alter_gt[self.alter_id.index(self.select_id)] = [input_kpx, input_kpy]
                self.repaint_img = self.repaint_image()
            except:
                QMessageBox.critical(self, "错误", "整数坐标值输入格式错误！")
    # 滚轮缩放图片, 缩放图片的同时也要缩放QLabel显示区域(WPS图片是以鼠标当前坐标位置为画布中心进行缩放)
    def wheelEvent(self, e):
        if e.angleDelta().y() > 0 and self.image_src is not None and self.align_flag is False:  # 放大图片, 防止没有加载图片时误动滚轮使线程卡死
            self.image_w *= 1.1
            self.image_h *= 1.1
            self.label_w *= 1.1
            self.label_h *= 1.1
            self.label_image.setGeometry(QtCore.QRect(self.label_x + self.dx, self.label_y + self.dy, self.label_w, self.label_h))
            self.label_image.setPixmap(QPixmap.fromImage(self.image_src).scaled(self.image_w, self.image_h))  # 刷写图片
        elif e.angleDelta().y() < 0 and self.image_src is not None and self.align_flag is False:  # 缩小图片
            self.image_w *= 0.9
            self.image_h *= 0.9
            self.label_w *= 0.9
            self.label_h *= 0.9
            self.label_image.setGeometry(QtCore.QRect(self.label_x + self.dx, self.label_y + self.dy, self.label_w, self.label_h))
            self.label_image.setPixmap(QPixmap.fromImage(self.image_src).scaled(self.image_w, self.image_h))
    def mouseMoveEvent(self, e):
        if self.left_click is True and self.align_flag is False:
            self.dx = e.pos().x() - self.startPos.x()
            self.dy = e.pos().y() - self.startPos.y()
            self.label_image.setGeometry(QtCore.QRect(self.label_x+self.dx, self.label_y+self.dy, self.label_w, self.label_h))
        if (self.left_click and self.align_flag and self.move_flag) is True:
            pos = [e.pos().x(), e.pos().y()]
            self.move_gt[self.move_id.index(self.select_id)] = pos
            self.repaint_img = self.repaint_image()
    def mousePressEvent(self, e):
        pos = [e.pos().x(), e.pos().y()]
        self.move_flag = False  # 防止在没有选中关键点时也有拖动效果
        # print(f"鼠标相对于窗口控件的坐标为：{e.pos()}")
        if e.button() == Qt.LeftButton:  # 鼠标左键在图片显示区域按下
            if (self.x_start <= e.pos().x() <= self.x_end) and self.image_src is not None and self.align_flag is False:
                self.left_click = True
                self.startPos = e.pos()
            elif self.select_point(pos) and self.align_flag is True:  # 鼠标左键单击选中关键点
                self.left_click = True
                self.move_flag = True
                if self.select_id in self.alter_id:  # 单击选中点不能同时存在于精校坐标点和移动坐标点中
                    self.alter_gt.remove(self.alter_gt[self.alter_id.index(self.select_id)])
                    self.alter_id.remove(self.select_id)
                if self.select_id not in self.move_id: 
                    self.move_id.append(self.select_id)
                    self.move_gt.append(pos)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:  # 鼠标左键在图片显示区域释放
            if self.x_start <= e.pos().x() <= self.x_end and self.image_src is not None and self.align_flag is False:
                self.left_click = False
                self.label_x += self.dx
                self.label_y += self.dy
    # 图像格式转换：opecv.ndarray转pyqt5.QImage
    def qpix_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, cols, channels = image.shape
        bytesPerLine = channels * cols
        image = QImage(image.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        return image
    def pix2pos(self, kps):
        kps_x = (kps[:, 0]/self.image_shape[1])*self.image_w + self.screenWidth * 0.2265
        kps_y = (kps[:, 1]/self.image_shape[0])*self.image_h + self.screenHeight * (1-self.image_h/self.screenHeight)/2
        kps_pos = np.stack((kps_x, kps_y), axis=-1).astype(np.int32)
        return kps_pos
    def pos2pix(self, pos):
        kps_x = (pos[0] - self.screenWidth * 0.2265)*self.image_shape[1]/self.image_w
        kps_y = (pos[1] - self.screenHeight * (1-self.image_h/self.screenHeight)/2) * self.image_shape[0] / self.image_h
        return int(kps_x), int(kps_y)
    # 重新渲染图像的同时更新关键点坐标集合
    def repaint_image(self):
        # img = Image.open(self.image_path)
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = self.raw_img.copy()
        for i in range(0, len(self.kps), 1):
            int_kpx = self.kps[i][0]
            int_kpy = self.kps[i][1]
            if i in self.delete_id:
                self.kps_adjust[i] = [-20, -20]
                continue
            elif i in self.alter_id:
                alter_index = self.alter_id.index(i)
                int_kpx = self.alter_gt[alter_index][0]
                int_kpy = self.alter_gt[alter_index][1]
                self.kps_adjust[i] = [int_kpx, int_kpy]
                self.kps_pos = self.pix2pos(self.kps_adjust)  # 更新关键点屏幕坐标集合
            elif i in self.move_id:
                move_index = self.move_id.index(i)
                int_kpx, int_kpy = self.pos2pix(self.move_gt[move_index])
                self.kps_adjust[i] = [int_kpx, int_kpy]
                self.kps_pos = self.pix2pos(self.kps_adjust)
            cv2.putText(img, f"{i}", (int_kpx + 5, int_kpy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (232, 9, 52), 4)
            cv2.circle(img, (int_kpx, int_kpy), 15, (38, 46, 222), -1)
        self.label_image.setPixmap(QPixmap.fromImage(self.qpix_img(img)).scaled(self.image_w, self.image_h))
        return img
    # 左右键单击选中关键点
    def select_point(self, pos):
        if self.kps_pos is not None:
            kps_posx = self.kps_pos[:, 0]
            kps_posy = self.kps_pos[:, 1]
            diff_posx = np.where(np.abs(kps_posx - pos[0]) < 5)[0]
            diff_posy = np.where(np.abs(kps_posy - pos[1]) < 5)[0]
            intersection = list(set(diff_posx) & set(diff_posy))
            print(f"单击选中的关键点编号为：{intersection}")
            if len(intersection) != 0:  # 取索引交集, 在关键点半径范围内
                self.select_id = intersection[0]
                return True
            else:
                return False
        else:
            return False
    # 两向量夹角公式
    def angle_formula(self, vector_a, vector_b):
        # 计算L1范数距离(向量长度)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        # 计算点积
        dot = np.dot(vector_a, vector_b)
        # 计算夹角余弦值
        cos_angle = dot/(norm_a*norm_b)
        # 将弧度制转换为角度制
        angle_value = np.rad2deg(np.arccos(cos_angle))
        # 保留小数点后两位
        return np.around(angle_value, 2)
    # 点到直线距离公式
    def distance_formular(self, vector_a, vector_b, vector_c):
        # 通过某顶点的两向量外(叉)积/对边长度
        distance = np.abs(np.cross(vector_a, vector_b))/np.linalg.norm(vector_c)
        return np.around(distance, 2)
    # 数据库中原图导出后重绘关键点
    def repaint_point(self, img):
        for i in range(0, len(self.kps), 1):
            int_kpx = self.kps[i][0]
            int_kpy = self.kps[i][1]
            cv2.putText(img, f"{i}", (int_kpx + 5, int_kpy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (232, 9, 52), 4)
            cv2.circle(img, (int_kpx, int_kpy), 15, (38, 46, 222), -1)
        return img

class GeometryForm(QtWidgets.QWidget, Ui_GeometryForm):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_win = PointDetection()
    win_icon = QIcon("icon/win.png")
    main_win.setWindowIcon(win_icon)
    main_win.show()  # 屏幕分辨率为1920*1080
    sys.exit(app.exec_())
