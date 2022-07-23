import torch
import pose_hrnet
import yaml
import cv2
import numpy as np
from inference import handle_single
import json
from scipy.optimize import curve_fit
import sqlite3

# 加载深度学习模型
def load_model(pth='model-lr-050.pth'):
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = pose_hrnet.get_pose_net(cfg, is_train=True)
    # pth = 'model-lr-050.pth'
    state_dict = torch.load(pth, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict['model'].items()})
    model.eval()
    return model
    # point_img, kps = handle_single("C:/Users/LSY/Desktop/lhwl/xkpd/关键点检测测试图片/CS_0010.JPG", model, 'cpu')
    # draw_curve(point_img, kps, "curve_config.json")

def fit_curve(img, kps_y, kps_x, num):
    # 自定义拟合函数形式
    # def fit_func(x, a, b, c, d):
    #     return a*np.power(x, 3) + b*np.power(x, 2) + c*x + d
    fit_func = np.polyfit(kps_y, kps_x, num)
    # 非线性最小二乘法拟合
    # popt, pcov = curve_fit(fit_func, kps_y, kps_x)
    fit_y = np.linspace(start=np.min(kps_y), stop=np.max(kps_y), num=int(np.max(kps_y)-np.min(kps_y))).astype(np.int32)
    fit_x = np.polyval(fit_func, fit_y).astype(np.int32)
    # contours = np.stack((np.expand_dims(fit_x, axis=0), np.expand_dims(fit_y, axis=0)), axis=-1).astype(np.int32)
    # curve_img = cv2.drawContours(point_img, [contours], -1, (255, 0, 0), 5)
    for i in range(0, len(fit_x)-1, 1):
        cv2.line(img, (fit_x[i], fit_y[i]), (fit_x[i+1], fit_y[i+1]), (64, 240, 60), 5)
    return img

def draw_curve(point_img, kps, file_path='curve_config.json'):
    curve_img = point_img.copy()
    # 解析生成曲线配置文件
    with open(file_path, 'rb') as f:
      js_data = json.loads(f.read())["fit_curve"]
    for item in js_data:
        num = item['value']  # 多项式方程次数
        kps_x = [kps[i, 0] for i in item['points']]
        kps_y = [kps[i, 1] for i in item['points']]
        curve_img = fit_curve(curve_img, kps_y, kps_x, num)
    # cv2.imwrite("result_curve.jpg", curve_img)
    return curve_img

# 创建数据表
def create_database():
    conn = sqlite3.connect('xkps.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE info
                   (name text, id text, points blob, image blob)''')
    # Save (commit) the changes
    conn.commit()
    conn.close()

if __name__ == "__main__":
    load_model()
    # create_database()