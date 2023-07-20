from PySide2.QtWidgets import QApplication, QMessageBox, QMainWindow, QPushButton, QPlainTextEdit, QTableWidgetItem
from PySide2.QtCore import *
from PySide2 import QtWidgets
import sys
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, Qt, QDir, QTime
from PySide2.QtGui import *  # QIcon,QImage,QPixmap
from threading import Thread
from multiprocessing import Process
import PySide2
import cv2

import play_music
import ftp
import send_mysql
import pymysql

# import A17_face_register,A17_face_train,A17_face_predict
from A17_send_email_remind import Send_Email
'''
pyinstaller -F --add-data="D:\Miniconda\envs\opencv\Lib\site-packages\mediapipe\modules;mediapipe/modules" Qt_main.py

'''
'''
python==3.8
pip install opencv-python==4.5.5.62 opencv-contrib-python==4.5.5.62 cvzone==1.5.0 rembg==2.0.28 pillow imutils scipy Cmake dlib pymysql pygame mediapipe  -i https://pypi.douban.com/simple
'''
import os
import pandas as pd
import numpy as np
import dlib
import time
import logging
import csv

''' 人脸录入 '''
# 加载预训练模型，caffe
net = cv2.dnn.readNetFromCaffe('./data/caffemodel/deploy.prototxt',
                               './data/caffemodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# face = cv2.CascadeClassifier('caffemodel/haarcascade_frontalface_alt.xml')
''' 人脸训练 '''
# 设置人脸位置
faces_path = './data/face'
# #加载预训练模型，caffe
# net = cv2.dnn.readNetFromCaffe('./data/caffemodel/deploy.prototxt','./data/caffemodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# 人脸68关键点模型
predictor_path = "./data/predictor_68_face/shape_predictor_68_face_landmarks.dat"
# 使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)
# 加载人脸128D模型
face_128d_model = dlib.face_recognition_model_v1('./data/128d_model/dlib_face_recognition_resnet_model_v1.dat')
'''人脸预测'''
# import cvzone
# from cvzone import FaceMeshModule
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
from cvzone.FaceMeshModule import FaceMeshDetector  # 导入检测器
# import mediapipe
import datetime
import gc
from imutils import face_utils
from scipy.spatial import distance as dist

# 获取嘴巴坐标参数值
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
# 获取人眼的坐标  获取左右人眼的坐标参数值
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
# 设置嘴巴横纵比阈值
MAR_THRESH = 0.48
# 设置眼睛横纵比阈值
EYE_AR_THRESH = 0.3  # 根据人眼大小来调整
# 设置eye横纵比阈值差值
# 0.03~0.04
EYE_MIN = 0.165
EYE_MAX_MIN_DELTA = 0.28

MOUTH_MIN = 0.3
MOUTH_MAX_MIN_DELTA = 0.55  # 0.45
detector = FaceMeshDetector(maxFaces=1)  # 检测人脸
# segmentor = SelfiSegmentation()
# fpsReader = cvzone.FPS()
# 设至深度范围
DEEP_STRAT_RANGE = 35.0  # 35.0
DEEP_END_RANGE = 50.0

# 背景去除
from PIL import Image
from rembg import remove

# # 设置人脸位置
# faces_path = './data/face'
# #加载预训练模型，caffe
# net = cv2.dnn.readNetFromCaffe('./data/caffemodel/deploy.prototxt','./data/caffemodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# # 人脸68关键点模型
# predictor_path = "./data/predictor_68_face/shape_predictor_68_face_landmarks.dat"
# 使用官方提供的模型构建特征提取器
# predictor = dlib.shape_predictor(predictor_path)
# # 加载人脸128D模型
# face_128d_model = dlib.face_recognition_model_v1('./data/128d_model/dlib_face_recognition_resnet_model_v1.dat')

# 警报
import pygame


class mySQL:
    # def __init__(self):
    def __init__(self):
        self.host = '101.33.204.87'
        self.port = 3306
        self.user = 'A17'
        self.password = 'zhongpeng'
        self.databass = 'a17'
        self.charset = 'utf8'
        self.ui = QUiLoader().load('./data/qt/databaseConnect.ui')
        self.ui.setWindowIcon(QIcon('./data/qt/team.ico'))
        self.ui.setWindowTitle('人脸数据库配置')
        # self.ui.save.setEnabled(False)#初始化禁用保存按钮
        self.ui.test.clicked.connect(self.connectTest)  # 点击连接测试
        # self.ui.revise.clicked.connect(self.revise)#点击修改
        # self.ui.save.clicked.connect(self.connect)#点击保存
        # 初始化设置参数
        self.ui.host.setText('101.33.204.87')
        self.ui.port.setText('3306')
        self.ui.user.setText('A17')
        self.ui.password.setText('zhongpeng')
        self.ui.database.setText('a17')
        self.ui.charset.setText('utf8')

    def connectTest(self):
        '''
        数据库连接测试
        :param host:ip
        :param port: 端口
        :param user: 用户
        :param password:密码




        :param databass: 数据库名字
        :param charset: 字符类型
        :return:
        '''
        self.get()
        try:
            conn = pymysql.connect(  # 赋值给 conn连接对象
                host=self.host,  # 本地回环地址
                port=self.port,  # 默认端口
                user=self.user,  # 用户名
                password=self.password,  # 密码
                database=self.databass,  # 连接数据库名称
                charset=self.charset  # 编码 不能写utf-8
            )
        except:
            # 连接失败
            QMessageBox.critical(
                self.ui,
                '错误',
                '数据库连接失败！')
            # self.ui.save.setEnabled(False)
            return
        # 连接成功
        QMessageBox.information(
            self.ui,
            '成功',
            '数据库连接成功！')
        # 启用保存按钮
        # self.ui.save.setEnabled(True)
        choice = QMessageBox.question(
            self.ui,
            '确认',
            '是否进行保存并更改')

        if choice == QMessageBox.Yes:
            self.host = self.ui.host.text()
            self.port = int(self.ui.port.text())
            self.user = self.ui.user.text()
            self.password = self.ui.password.text()
            self.databass = self.ui.database.text()
            self.charset = self.ui.charset.text()
        if choice == QMessageBox.No:
            return
    def get(self):
        self.host = self.ui.host.text()
        self.port = int(self.ui.port.text())
        self.user = self.ui.user.text()
        self.password = self.ui.password.text()
        self.databass = self.ui.database.text()
        self.charset = self.ui.charset.text()



class Stats(object):
    def __init__(self):
        ''' 人脸录入'''
        # FACE_PATH
        self.faces_path = './data/face'  # 设置人脸存储路径
        # FONT
        self.font = cv2.FONT_ITALIC  # 设置字体类型
        # FACE
        self.existing_faces_cnt = 0  # 现存人脸张数
        # self.camera_preson_face = 0           # 摄像头识别到的人脸个数
        self.person_face_cnt = 0  # 将该用户人脸计数器
        # FLAG
        self.press_n_flag = 0  # 是否按下'n'创建人脸， 0：否  1：是
        self.save_flag = 0  # 人脸是否符合界限       0：否  1：是
        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        ''' 人脸预测 '''
        # FACE_PATH
        self.faces_path = './data/face'  # 设置人脸存储路径
        # # FONT
        # self.font = cv2.FONT_ITALIC  # 设置字体类型
        # FACE
        self.face_name_know_list = []  # 现存人脸张数
        self.face_feature_know_list = []  # 存放所有录入人脸的特征组
        # self.camera_preson_face = 0           # 摄像头识别到的人脸个数
        # # FPS
        # self.frame_time = 0
        # self.frame_start_time = 0
        # self.fps = 0
        # self.fps_show = 0
        # self.start_time = time.time()

        self.use_count = 0  # 用户人脸连续出现次数
        self.Unuse_count = 0  # 未知人脸连续出现次数

        # biopsy testing
        self.average_eye_EAR = 0  # 眼睛的纵横比值
        self.mouth_MAR = 0  # 嘴巴纵横比值
        self.eye_flag = 0  # 判定眼睛是否为活体   0：否  1：是
        self.mouth_flag = 0  # 判定嘴巴是否为活体   0：否  1：是
        self.eye_max = 0

        self.eye_ear_max = 0.0
        self.eye_ear_min = 1.0
        self.mouth_mar_max = 0.0
        self.mouth_mar_min = 1.0
        '''ui'''
        # self.em = Send_Email()
        self.video_flag = 0                     # 判断是否选择打开视频流
        self.video_off_flag = 0                 # 判断是否选择关闭视频流
        self.video_register_flag = 0            # 判断是否选择人脸录入
        self.video_predict_flag = 0             # 判断是否选择人脸检测
        self.video_off_flag = 0                 # 判断是否选择关闭视频流
        self.RANGE = 0                          # 初始化设置进度
        self.VALUE = 0                          # 初始化进度值
        self.cap = cv2.VideoCapture(0)
        self.ui = QUiLoader().load('data/qt/main.ui')                       # 从文件加载UI定义
        self.ui.setWindowIcon(QIcon('./data/qt/team.ico'))                         # 加载团队图片
        self.ui.setWindowTitle('人脸识别')                                 # 对windows窗口命名
        self.init_video()                                                   # 初始化显示图片
        self.ui.radioButton.clicked.connect(self.video_open_flag)           # 按下打开视频流动
        self.ui.radioButton_off.clicked.connect(self.video_off)             # 按下关闭视频流动
        self.ui.developer.triggered.connect(self.about)                     # 点击关于->开发者信息
        self.ui.safe.triggered.connect(self.send)                           # 点击设置->安全提醒
        self.ui.face_register.clicked.connect(self.face_register)           # 点击人脸录入
        self.ui.face_predict.clicked.connect(self.face_predict)             # 点击人脸识别
        self.ui.face_train.clicked.connect(self.face_train)                 # 点击人脸训练
        self.ui.face_new.clicked.connect(self.new_face_zero_to_one)         # 点击新建
        self.ui.face_save.clicked.connect(self.save_face_zero_to_one)       # 点击人脸抓取并保存
        self.ui.face_new.setEnabled(False)                                  # 新建人脸按钮不可用
        self.ui.face_save.setEnabled(False)                                 # 人脸保存按钮不可用
        self.save_face_flag = 0                                             # 是否可以保存人脸
        self.new_face_flag = 0                                              # 是否进行过创建人脸文件
        self.ui.progressBar.setRange(0,self.RANGE)                          # 初始化设置进度条
        self.ui.progressBar.setValue(self.VALUE)                            # 初始化进度条值
        self.email_send_flag = 1                                            # 安全提醒是否开启
        # self.key = cv2.waitKey(1)
        self.init_edit()                                                    # 初始化清楚文本框
        self.ui.output.setReadOnly(True)                                    # 设置文本框禁止编辑(设置为只读)
        self.init_face()                                                    # 人脸初始化
        self.csv_flag = 0                                                   # 是否有csv文件
        self.remove = remove                                                # 创建remove对象
        self.sql = send_mysql.mysql()                                       # 初始化数据库对象
        self.sql.connect_mysql()                                            # 连接数据库
        self.ui.sql.triggered.connect(self.sql_init)                        # 点击数据库配置
        self.date_init()                                                    # 初始化日期
        self.ui.sure.clicked.connect(self.select_mysql)                     # 识别信息->确定
        self.ui.flush.clicked.connect(self.flush)                           # 识别信息->刷新
        self.music_flag = 0                                                 # 报警标记 0->1 打开 1->0 关闭
    ''' 人脸录入 '''

    # 新建人脸存储路径文件夹
    def check_face_folder(self):
        # 判断是否存在人脸存储路径
        if os.path.isdir(self.faces_path):
            pass
        else:
            os.mkdir(self.faces_path)
        # 检查现存人脸数
    def check_existing_faces_cnt(self):
        if os.listdir(self.faces_path):  # 存在用户人脸数据
            person_list = os.listdir(self.faces_path)
            map(int, person_list)
            self.existing_faces_cnt = int(max(person_list))
            logging.info("%-60s / %-10s",'用户人脸个数 / Number of existing users :' + f"{self.existing_faces_cnt}",datetime.datetime.now())
            self.append_edit("现存用户人脸:" + "{}".format(self.existing_faces_cnt))
        else:
            self.existing_faces_cnt = 0

    # 更新 FPS
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time  # 开始时间减现在时间为时间间隔
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img):
        # 添加说明 / Add some notes
        # cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "FPS: " + str(self.fps_show.__round__(2)), (10, 30), self.font, 1.2, (0, 255, 0), 3,
                    cv2.LINE_AA)  # 调用时返回2位小数
        # cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)  # cv2.LINE_AA:抗锯齿线
        # cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 获取人脸
    def process(self, cap, camera_X, camera_Y):
        self.check_face_folder()  # 检查是否存在人脸文件根
        self.check_existing_faces_cnt()
        key = cv2.waitKey(1)
        while self.video_register_flag:
            ret, img = cap.read()
            img_copy = img.copy()
            w = img.shape[1]
            h = img.shape[0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.new_face_flag == 1:  # 如果按下n，创建人脸文件  self.press_n_flag 0->1
                self.existing_faces_cnt += 1  # 现存人脸数+1
                current_face_dir = '{}/{}'.format(self.faces_path, self.existing_faces_cnt)
                os.mkdir(current_face_dir)
                logging.info("\n%-30s%-10s / %-10s", "新建的人脸文件夹 / Create folders:", current_face_dir,datetime.datetime.now())
                self.append_edit("{} ----- {}".format("新建的人脸文件夹:", current_face_dir) )
                self.person_face_cnt = 0  # 将人脸计数器清零
                self.press_n_flag = 1  # 已经按下 'n'
                # print(self.person_face_cnt)
                self.new_face_one_to_zero()
                self.ui.face_save.setEnabled(True)
            # 为了获得最佳精度，必须分别对蓝色、绿色和红色通道执行 (104, 177, 123) 通道均值减法，并将图像调整为 300 x 300 的 BGR 图像，在 OpenCV 中可以通过使用 cv2.dnn.blobFromImage() 函数进行此预处理
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)
            # 将 blob 设置为输入以获得结果，对整个网络执行前向计算以计算输出
            net.setInput(blob)
            detections = net.forward()
            camera_preson_face = []  # (startX, startY), (endX, endY)
            # 最后一步是迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化：
            # 迭代所有检测结果
            for i in range(0, detections.shape[2]):
                self.camera_preson_face = 0
                # 获取当前检测结果的置信度
                confidence = detections[0, 0, i, 2]
                # 如果置信大于最小置信度，则将其可视化
                if confidence > 0.7:
                    # detected_faces += 1
                    # 获取当前检测结果的坐标
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')
                    # 绘制检测结果和置信度
                    text = "{:.3f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 3)
                    # print([startX,startY,endX,endY])
                    camera_preson_face.append((startX, startY, endX, endY))
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # 获取人脸个数
            num = len(camera_preson_face)
            # print(camera_preson_face)
            # print(num)
            # 检测到无人脸
            if num == 0:
                cv2.putText(img, 'NO Face!', (225, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 0, 255), 3)  # 提示窗口无人脸
            # 检测到多人脸
            elif num > 1:
                cv2.putText(img, 'Don'+"'"+'t two faces', (190 ,150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255),
                            3)  # 提示不要出现两张人脸
            else:
                # 检测到符合要求一张人脸
                # 判断头像是否符合规范（未出界）
                (startX, startY, endX, endY) in camera_preson_face
                face_width = endX - startX
                face_hight = endY - startY
                face_width_c2 = int(face_width / 2)
                face_hight_c2 = int(face_hight / 2)
                if (endX + face_width_c2) > camera_X or (endY + face_hight_c2) > camera_Y or (
                        startX - face_width_c2) < 0 or (startY - face_hight_c2) < 0:
                    # 超出界限
                    cv2.putText(img, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    self.save_flag = 0
                else:
                    self.save_flag = 1

                if self.save_flag:  # 人脸符合界限
                    if self.save_face_flag == 1:
                        # 根据人脸大小生成空的图像
                        img_blank = np.zeros((int(face_hight * 2), face_width * 2, 3), np.uint8)
                        if self.press_n_flag:  # 之前是否按下n
                            self.person_face_cnt += 1
                            for i in range(face_hight * 2):  # 上下个取半个face_hight
                                for j in range(face_width * 2):  # 左右个取半个face_hight
                                    img_blank[i][j] = img_copy[startY - face_hight_c2 + i][startX - face_width_c2 + j]
                            cv2.imwrite(
                                '{}/{}/{}.jpg'.format(self.faces_path, self.existing_faces_cnt, self.person_face_cnt),
                                img_blank)
                            # cv2.imwrite('{}/{}/{}111.jpg'.format(self.faces_path, self.existing_faces_cnt, self.person_face_cnt),img_copy[startY:startY+face_hight,startX:startX+face_width])
                            logging.info("%-60s  / %-10s", "写入本地 / Save into："+str(current_face_dir)+'/img_face_'+str(self.person_face_cnt)+'.jpg',
                                         datetime.datetime.now())
                            self.append_edit("{} --- {}/img_face_{}.jpg".format("写入本地 ：", str(current_face_dir),str(self.person_face_cnt)) )
                            self.save_face_one_to_zero()
                    else:
                        self.save_face_flag = 0
                else:
                    self.save_face_flag = 0

            # 按下q退出
            # if key == ord('q'):
            #     break
            #  更新 FPS
            self.update_fps()
            # 添加文字
            self.draw_note(img)

            # 可视化
            # cv2.imshow('output', img)
            self.img_to_label_show(img)
            key = cv2.waitKey(1)
            # cv2.namedWindow("camera", 1)
            # cv2.imshow("camera", img)
        # self.ui.
        print("人脸录入已关闭")
        logging.info('%-60s / %-10s','人脸录入已关闭 / Face entry is turned off',datetime.datetime.now())
        self.append_edit("人脸录入已关闭")

    def register_run(self):
        if os.path.isdir('./data'):
            pass
        else:
            os.mkdir('./data')
        ret, image = self.cap.read()
        camera_X, camera_Y = int(image.shape[1]), int(image.shape[0])
        if ret:
            self.process(self.cap, camera_X, camera_Y)  # 调用人脸识别
        else:
            print('摄像头打开失败')
            self.append_edit('摄像头打开失败')
            
        # cap.release()

    ''' 人脸训练'''

    def return_128d_features(self, photo_path):
        img = cv2.imread(photo_path)
        logging.info("%-60s %-20s / %-10s", "检测到人脸的图像 / Image with faces detected:", photo_path,datetime.datetime.now())
        w = img.shape[1]
        h = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 为了获得最佳精度，必须分别对蓝色、绿色和红色通道执行 (104, 177, 123) 通道均值减法，并将图像调整为 300 x 300 的 BGR 图像，在 OpenCV 中可以通过使用 cv2.dnn.blobFromImage() 函数进行此预处理
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)
        # 将 blob 设置为输入以获得结果，对整个网络执行前向计算以计算输出
        net.setInput(blob)
        detections = net.forward()
        # 最后一步是迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化：
        # 迭代所有检测结果
        camera_preson_face = []  # (startX, startY), (endX, endY)
        for i in range(0, detections.shape[2]):
            # print(i)
            # 获取当前检测结果的置信度
            confidence = detections[0, 0, i, 2]
            # 如果置信大于最小置信度，则将其可视化
            if confidence > 0.7:
                # detected_faces += 1
                # 获取当前检测结果的坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # print(box)
                (startX, startY, endX, endY) = box.astype('int')
                camera_preson_face.append((startX, startY, endX, endY))
                # 绘制检测结果和置信度
                # text = "{:.3f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 3)
                # cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        if len(camera_preson_face) == 1:
            (startX, startY, endX, endY) in camera_preson_face
            face_width_c5 = int((endX - startX) / 5)
            face_hight_c5 = int((endY - startY) / 5)
            # rectangle = dlib.rectangle(startX-face_width_c5, startY-face_hight_c5, endX+face_width_c5, endY+face_hight_c5)
            rectangle = dlib.rectangle(startX, startY, endX, endY)
            shape = predictor(img, rectangle)
            # 查看68point
            # for p in shape.parts():
            #     cv2.circle(img, (p.x, p.y), 3, (0,0,255), -1)
            # cv2.imshow('video', img)
            # cv2.waitKey(0)
            face_descriptor = face_128d_model.compute_face_descriptor(img, shape)
        else:
            face_descriptor = 0
            # logging.warning("No Face")
        self.VALUE += 1
        self.progress_bar(self.VALUE)
        return face_descriptor

    def return_features_mean_personX(self, personX_face_path):
        features_list_personX = []  # 存储特征值
        face_photos_list = os.listdir(personX_face_path)  # 获取personX对应的人脸图片路径
        if face_photos_list:
            for i in range(len(face_photos_list)):
                # 调用 return_128d_features 得到图片128D特征矢量值
                logging.info("%-60s %-20s / %-10s", "正在读的人脸图像 / Reading image:", personX_face_path + "/" + face_photos_list[i],datetime.datetime.now())
                photo_path = os.path.join(personX_face_path, face_photos_list[i])
                features_128d = self.return_128d_features(photo_path)  # 返回0或128d矢量  0：无人脸
                # 遇到没有检查到的人脸图片就跳过
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            logging.warning("文件夹内图像文件为空 / Warning: No images in%s / %-10s", personX_face_path,datetime.datetime.now())
        # 计算 128D 特征的均值 / Compute the mean
        # personX 的 N 张图像 x 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        return features_mean_personX

    def train_run(self):
        if os.path.exists(faces_path):
            person_list = os.listdir(faces_path)
            face_name = set()


            person_list.sort()  # 将人顺序从低到高排序
            if not os.path.exists("./data/face_csv"):
                os.makedirs("./data/face_csv")
            if not os.path.exists("./data/face_csv/features_all.csv"):
                f = open("./data/face_csv/features_all.csv", "w", newline="")
                print("新建features_all.csv")
                logging.info('%-60s / %-10s','新建features_all.csv / New File features_all.csv',datetime.datetime.now())
                self.append_edit('新建features_all.csv')
                f.close()
            with open("./data/face_csv/features_all.csv", "r+", newline="") as csvfile:
                write = csv.writer(csvfile)
                # 获取csv中存在的人脸
                for line in csvfile.readlines():
                    # print(type(line[0]))
                    face_name.add(line[0])
                print(face_name)
                self.RANGE = 0
                self.VALUE = 0
                for person_path in person_list:
                    if str(person_path) not in face_name:
                        person_len = len(os.listdir(os.path.join(faces_path,person_path)))
                        self.RANGE += person_len
                # 初始化进度
                self.init_progress_bar(self.RANGE)
                for person_path in person_list:
                    if str(person_path) not in face_name:
                        # 获取person对应的128D
                        # logging.info("%s_person_%s", faces_path, person_path)
                        personX_face_path = os.path.join(faces_path, person_path)
                        features_mean_personX = self.return_features_mean_personX(personX_face_path)
                        # 获取对应person编号
                        person_name = person_path
                        features_mean_personX = np.insert(features_mean_personX, 0, person_name,
                                                          axis=0)  # a=np.insert(arr, obj, values, axis)#arr原始数组，可一可多，obj插入元素位置，values是插入内容，axis是按行按列插入（0：行、1：列）
                        write.writerow(features_mean_personX)
                        # logging.info('\n')
                self.RANGE = 0
                self.VALUE = 0
                self.init_progress_bar(self.RANGE)
                self.progress_bar(self.VALUE)
                logging.info(
                    "所有录入人脸数据存入 / Save all the features of faces registered into: data/face_csv/shape_predictor_68_face_landmarks.dat %-10s",datetime.datetime.now())
                self.csv_flag = 1
                self.ui.face_predict.setEnabled(True)
        else:
            logging.error('%-60s / %-10s', '不存在人脸图片 / There is no face picture', datetime.datetime.now())
    ''' 人脸预测 '''

    # 从features_all.csv 读取人脸特征
    def get_faces_features(self):
        if os.path.exists("./data/face_csv/features_all.csv"):
            faces_features_path = './data/face_csv/features_all.csv'
            csv = pd.read_csv(faces_features_path, header=None)
            for person_name in range(csv.shape[0]):
                features_someone_arr = []
                self.face_name_know_list.append(csv.iloc[person_name][0])
                for featires_128d in range(1, 129):
                    if csv.iloc[person_name][featires_128d] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv.iloc[person_name][featires_128d])
                self.face_feature_know_list.append(features_someone_arr)
            # logging.info("Face in load:%d", len(self.face_feature_know_list))
            return 1
        else:
            logging.warning('%-60s / %-10s','未找到"features_all.csv" / "features_all.csv" not found!',datetime.datetime.now())
            # logging.warning('Please run "A17_face_register"->"A17_face_train" last runing "A17_face)predict"')
            return 0

    # 计算两个128D向量间的欧式距离
    def return_enclidean_distance(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def face_deep(self, img):
        '''
        计算人脸距离摄像头距离
        :param img: image
        :return: image,deep
        '''
        # img = img.copy()
        img, faces = detector.findFaceMesh(img, draw=False
                                           )  # 做检测器，查找面部网格，返回图像（img）和我们的面部（faces）  ,draw = False有了这句话后就看不到网格了
        if faces:  # 如果有面部（faces）可用
            ##通过以下语句找到左眼和右眼两个点
            face = faces[0]  # 先进行第一阶段
            pointLeft = face[145]  # 左边的值基本上是值145
            pointRight = face[374]  # 右边的值基本上是值374
            ##下面是找眼睛两个点之间的距离
            # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)  # 在眼睛两个点之间画一条线，起点是pointLeft,终点是pointRight,线条颜色为绿色，线宽为3
            # cv2.circle(img,pointLeft,5,(255,0,255),cv2.FILLED)     #在img图像上画圆，中心点为pointLeft,半径为5，颜色为紫色，最后的运行结果能够在成像人的左眼标出紫色的圆点
            # cv2.circle(img,pointRight,5,(255,0,255),cv2.FILLED)    #在img图像上画圆，中心点为pointRighe，半径为5，颜色为紫色，最后的运行结果能够在成像人的右眼标出紫色的圆点

            w, _ = detector.findDistance(pointLeft, pointRight)  # 将左眼点的位置到右眼点位置的距离赋值给w        w后面的下划线是忽略其他的值
            W = 6.3  # 这是人眼的左眼与右眼之间的距离为63mm，取了中间值，男人的为64mm，女人的为62mm

            ###查找距离
            ##通过上面f = (w * d) / W公式，可以大致的测出，当人眼距离相机50cm时，相机的焦距为300左右
            ##再将找到的焦距代入计算距离的公式，就可以算出距离
            f = 300
            d = (W * f) / w
            self.deep = d
            ##下面是将距离的文本字样跟随人脸移动输出在额头位置
            # cvzone.putTextRect(img, f'Depth:{int(d)}cm', (face[10][0] - 95, face[10][1] - 5),
            #                    scale=1.8)  # 将距离文本显示在图像上，以字符串形式显示，单位为cm,文本值显示的位置跟随人面部移动显示在额头上（额头上的id是10，也就是face[10],后面的face[10][0]表示第一个元素，face[10][1]表示第二个元素），
            cv2.putText(img, "Deep: " + '{:.2f}'.format(d), (10, 80), self.font, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)

        return mar

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])  # 计算两组 垂直 眼界标之间的距离
        B = dist.euclidean(eye[2], eye[4])  # 计算两组 垂直 眼界标之间的距离
        C = dist.euclidean(eye[0], eye[3])  # 计算水平眼界标之间的距离
        ear = (A + B) / (2.0 * C)
        return ear

    # 处理获取的视频流，进行人脸识别
    def predict(self, cap, camera_X, camera_Y):
        if self.get_faces_features():
            while self.video_predict_flag:
                self.frame_time += 1
                _, img = cap.read()
                key = cv2.waitKey(1)
                w = img.shape[1]
                h = img.shape[0]
                img_copy = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if key == ord('q'):  # 如果按下n，创建人脸文件  self.press_n_flag 0->1
                    break
                # 为了获得最佳精度，必须分别对蓝色、绿色和红色通道执行 (104, 177, 123) 通道均值减法，并将图像调整为 300 x 300 的 BGR 图像，在 OpenCV 中可以通过使用 cv2.dnn.blobFromImage() 函数进行此预处理
                try:
                    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)
                except:
                    continue
                # 将 blob 设置为输入以获得结果，对整个网络执行前向计算以计算输出
                net.setInput(blob)
                detections = net.forward()
                camera_preson_face = []  # (startX, startY), (endX, endY)
                # 最后一步是迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化：
                # 迭代所有检测结果
                for i in range(0, detections.shape[2]):
                    self.camera_preson_face = 0
                    # 获取当前检测结果的置信度
                    confidence = detections[0, 0, i, 2]
                    # 如果置信大于最小置信度，则将其可视化
                    if confidence > 0.7:
                        # detected_faces += 1
                        # 获取当前检测结果的坐标
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype('int')
                        # 绘制检测结果和置信度
                        # text = "{:.3f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 3)
                        # print([startX,startY,endX,endY])
                        camera_preson_face.append((startX, startY, endX, endY))
                        # cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # 获取人脸个数
                num = len(camera_preson_face)
                # print(camera_preson_face)
                # print(num)
                # 检测到无人脸
                if num == 0:
                    cv2.putText(img, 'NO Face!', (225, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                (0, 0, 255), 3)  # 提示窗口无人脸
                    # 重置连续用户\非用户连续人脸计数器
                    self.use_count = 0
                    self.Unuse_count = 0
                    # 重置纵横比值
                    self.average_eye_EAR = 0
                    self.mouth_MAR = 0
                    # 重置眼睛、嘴巴flag记录
                    self.eye_flag = 0
                    self.mouth_flag = 0
                    # 重置眼睛、嘴巴 max、min
                    self.eye_ear_min = 1.0
                    self.eye_ear_max = 0.0
                    self.mouth_mar_min = 1.0
                    self.mouth_mar_max = 0.0
                    # 重置人脸图像深度
                    self.deep = 0
                # 检测到多人脸
                elif num > 1:
                    cv2.putText(img, 'Don' + "'" + 't two faces', (190, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                (0, 0, 255),3)
                    # 重置连续用户\非用户连续人脸计数器
                    self.use_count = 0
                    self.Unuse_count = 0
                    # 重置纵横比值
                    self.average_eye_EAR = 0
                    self.mouth_MAR = 0
                    # 重置眼睛、嘴巴flag记录
                    self.eye_flag = 0
                    self.mouth_flag = 0
                    # 重置眼睛、嘴巴 max、min
                    self.eye_ear_min = 1.0
                    self.eye_ear_max = 0.0
                    self.mouth_mar_min = 1.0
                    self.mouth_mar_max = 0.0
                    # 重置人脸图像深度
                    self.deep = 0
                else:
                    # 检测到符合要求一张人脸
                    # 判断头像是否符合规范（未出界）
                    (startX, startY, endX, endY) in camera_preson_face
                    face_width = endX - startX
                    face_hight = endY - startY
                    face_width_c2 = int(face_width / 2)
                    face_hight_c2 = int(face_hight / 2)
                    if (endX + face_width_c2) > camera_X or (endY + face_hight_c2) > camera_Y or (
                            startX - face_width_c2) < 0 or (startY - face_hight_c2) < 0:
                        # 超出界限
                        cv2.putText(img, "OUT OF RANGE", (20, 300), self.font, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        # 进行人脸测距
                        self.face_deep(img)
                        if not (DEEP_STRAT_RANGE <= self.deep <= DEEP_END_RANGE):
                            cv2.putText(img, "KEEP DEEP in 35~50", (20, 300), self.font, 0.8, (0, 0, 255), 1,
                                        cv2.LINE_AA)
                        else:
                            rectangle = dlib.rectangle(startX, startY, endX, endY)
                            shape = predictor(img, rectangle)  # 获取68点值 shape值

                            shape_np = face_utils.shape_to_np(shape)  # 68点值 shape值 array

                            # 提取人眼坐标。来计算人眼纵横比
                            leftEye = shape_np[lStart:lEnd]
                            rightEye = shape_np[rStart:rEnd]
                            leftEAR = self.eye_aspect_ratio(leftEye)
                            rightEAR = self.eye_aspect_ratio(rightEye)
                            # 平均左右眼的纵横比
                            self.average_eye_EAR = (leftEAR + rightEAR) / 2.0

                            mouth = shape_np[mStart:mEnd]
                            self.mouth_MAR = self.mouth_aspect_ratio(mouth)

                            # print('ACERAGE_EYE_EAR:{} --------- MOUTH:{}'.format(
                            #     self.average_eye_EAR,self.mouth_EAR))
                            if self.eye_ear_max < self.average_eye_EAR:
                                self.eye_ear_max = self.average_eye_EAR
                            if self.eye_ear_min > self.average_eye_EAR:
                                self.eye_ear_min = self.average_eye_EAR
                            if self.mouth_mar_max < self.mouth_MAR:
                                self.mouth_mar_max = self.mouth_MAR
                            if self.mouth_mar_min > self.mouth_MAR:
                                self.mouth_mar_min = self.mouth_MAR
                            if EYE_MIN < (self.eye_ear_max - self.eye_ear_min) < EYE_MAX_MIN_DELTA:
                                self.eye_flag = 1
                            else:
                                self.eye_flag = 0
                                cv2.putText(img, "Please open your eye", (20, 330), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                            if MOUTH_MIN < (self.mouth_mar_max - self.mouth_mar_min) < MOUTH_MAX_MIN_DELTA:
                                self.mouth_flag = 1
                            else:
                                self.mouth_flag = 0
                                cv2.putText(img, "Open Mouth", (20, 360), self.font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            print('eye:{} ---- mouth:{}'.format(self.eye_ear_max - self.eye_ear_min,
                                                                self.mouth_mar_max - self.mouth_mar_min))

                            if self.eye_flag == 1 and self.mouth_flag == 1:

                                # 人脸模型128D特征数据提取
                                face_descriptor = face_128d_model.compute_face_descriptor(img, shape)
                                current_frame_e_distance_list = []
                                # 对于某张人脸，遍历所有存储的人脸特征
                                for i in range(len(self.face_feature_know_list)):
                                    # 如果 person_X not None
                                    if str(self.face_feature_know_list[i][0]) != '0.0':
                                        e_distance_tmp = self.return_enclidean_distance(face_descriptor,
                                                                                        self.face_feature_know_list[
                                                                                            i])  # 调用 return_enclidean_distance 计算两个128D向量间的欧式距离
                                        current_frame_e_distance_list.append(e_distance_tmp)
                                    else:
                                        # 空数据
                                        current_frame_e_distance_list.append(999999999)
                                # 寻找出最小的欧式距离匹配
                                # similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))  # 获取最小的下标
                                similar_person_num = np.argmin(current_frame_e_distance_list)
                                print(current_frame_e_distance_list)
                                if min(current_frame_e_distance_list) < 0.34:
                                    cv2.putText(img, "person_{}".format(similar_person_num), (startX, startY - 10),
                                                self.font, 0.8, (0, 0, 255), 2)
                                    self.use_count += 1
                                    self.Unuse_count = 0
                                else:
                                    cv2.putText(img, "Unknow", (startX, startY - 10), self.font, 0.8, (0, 0, 255), 2)
                                    self.Unuse_count += 1
                                    self.use_count = 0
                                if self.use_count == 10 or self.Unuse_count == 10:
                                    # 对图像进行背景替换
                                    '''imgOut = segmentor.removeBG(img_copy, cv2.imread('./data/white/480_640.jpg'),
                                                                threshold=0.8)

                                    imgStack = cvzone.stackImages([img_copy, imgOut], 2, 1)
                                    _, imgStack = fpsReader.update(imgStack)'''

                                    # 检测是否存在用户和非用户人脸抓拍存放记录
                                    if not os.path.isdir('./data/recording/use'):
                                        os.makedirs('./data/recording/use')
                                    if not os.path.isdir('./data/recording/Unknow'):
                                        os.makedirs('./data/recording/Unknow')
                                    now = datetime.datetime.today()
                                    year = now.year
                                    month = now.month
                                    day = now.day
                                    time_hour = now.hour
                                    time_minute = now.minute
                                    if self.use_count == 10:
                                        if not os.path.isdir('./data/recording/use/{}/{}/{}'.format(year, month, day)):
                                            os.makedirs('./data/recording/use/{}/{}/{}'.format(year, month, day))
                                        cv2.imwrite(
                                            './data/recording/use/{}/{}/{}/use{}_{}_{}.jpg'.format(year, month, day,
                                                                                                   similar_person_num,
                                                                                                   time_hour,
                                                                                                   time_minute),
                                            img_copy)
                                        cv2.imwrite(
                                            './data/recording/use/{}/{}/{}/use{}_ROI_{}_{}.jpg'.format(year, month, day,
                                                                                                       similar_person_num,
                                                                                                       time_hour,
                                                                                                       time_minute),
                                            img_copy[startY:endY, startX:endX])

                                        # unet模型 人体分割
                                        imgMask = self.to_mask('./data/recording/use/{}/{}/{}/use{}_{}_{}.jpg'.format(year, month, day,
                                                                                                   similar_person_num,
                                                                                                   time_hour,
                                                                                                   time_minute))

                                        imgMask = np.array(imgMask)

                                        cv2.imwrite('./data/recording/use/{}/{}/{}/use{}_mask_{}_{}.jpg'.format(year, month,day,similar_person_num,time_hour,time_minute),imgMask)


                                        # ftp将数据发送到云
                                        path = [
                                            './data/recording/use/{}/{}/{}/use{}_{}_{}.jpg'.format(year, month,
                                                                                                         day,
                                                                                                         similar_person_num,
                                                                                                         time_hour,
                                                                                                         time_minute),
                                            './data/recording/use/{}/{}/{}/use{}_mask_{}_{}.jpg'.format(year,
                                                                                                              month,
                                                                                                              day,
                                                                                                              similar_person_num,
                                                                                                              time_hour,
                                                                                                              time_minute),
                                            './data/recording/use/{}/{}/{}/use{}_ROI_{}_{}.jpg'.format(year,
                                                                                                             month, day,
                                                                                                             similar_person_num,
                                                                                                             time_hour,
                                                                                                             time_minute)]
                                        name = ['use{}_{}_{}.jpg'.format(similar_person_num, time_hour, time_minute),'use{}_mask_{}_{}.jpg'.format(similar_person_num, time_hour,time_minute),'use{}_ROI_{}_{}.jpg'.format(similar_person_num, time_hour,time_minute)]
                                        t_f = Thread(target=self.ftp, args=(path, name))
                                        t_f.start()
                                        # 发送给数据库
                                        t_sql = Thread(target=self.send_mysql, args=(similar_person_num,path[0], path[1], path[2]))
                                        t_sql.start()
                                        self.music_flag = 0


                                    else:
                                        # 非用户记录
                                        if not os.path.isdir(
                                                './data/recording/Unknow/{}/{}/{}'.format(year, month, day)):
                                            os.makedirs('./data/recording/Unknow/{}/{}/{}'.format(year, month, day))
                                        cv2.imwrite(
                                            './data/recording/Unknow/{}/{}/{}/Unknow_{}_{}.jpg'.format(year, month,
                                                                                                         day,
                                                                                                         time_hour,
                                                                                                         time_minute),
                                            img_copy)
                                        cv2.imwrite(
                                            './data/recording/Unknow/{}/{}/{}/Unknow_ROI_{}_{}.jpg'.format(year,
                                                                                                             month, day,
                                                                                                             time_hour,
                                                                                                             time_minute),
                                            img_copy[startY:endY, startX:endX])

                                        # unet模型 人体分割
                                        imgMask = self.to_mask(
                                             './data/recording/Unknow/{}/{}/{}/Unknow_{}_{}.jpg'.format(year, month, day,
                                                                                                   time_hour,
                                                                                                   time_minute))

                                        imgMask = np.array(imgMask)

                                        cv2.imwrite(
                                            './data/recording/Unknow/{}/{}/{}/Unknow_mask_{}_{}.jpg'.format(year,
                                                                                                              month,
                                                                                                              day,
                                                                                                              time_hour,
                                                                                                              time_minute),
                                            imgMask)
                                        # 构造邮件配置并发送邮件
                                        if self.email_send_flag:
                                            t = Thread(target=self.email_send_fun,args=(year, month,day,time_hour,time_minute))
                                            t.start()
                                        # ftp将数据发送到云
                                        path = ['./data/recording/Unknow/{}/{}/{}/Unknow_{}_{}.jpg'.format(year, month,day,time_hour,time_minute),'./data/recording/Unknow/{}/{}/{}/Unknow_mask_{}_{}.jpg'.format(year, month,day,time_hour,time_minute),'./data/recording/Unknow/{}/{}/{}/Unknow_ROI_{}_{}.jpg'.format(year, month,day,time_hour,time_minute)]
                                        name = ['Unknow_{}_{}.jpg'.format(time_hour,time_minute),'Unknow_mask_{}_{}.jpg'.format(time_hour,time_minute),'Unknow_ROI_{}_{}.jpg'.format(time_hour,time_minute)]
                                        t_a = Thread(target=self.ftp,args=(path,name))
                                        t_a.start()
                                        # 发送给数据库
                                        t2_sql = Thread(target=self.send_mysql, args=("Unknow",path[0], path[1], path[2]))
                                        t2_sql.start()
                                        # 报警
                                        # 打开报警循环开关
                                        if self.music_flag != 1:
                                            self.music_flag = 1
                                            t_w = Thread(target=self.music)
                                            t_w.start()
                            # print(gc.garbage)
                            # print()
                            # print(gc.collect())
                # 更新FPS
                self.update_fps()
                # 添加文字
                self.draw_note(img)
                self.img_to_label_show(img)
                # cv2.imshow('img', img)
                # cv2.waitKey(1)
            # print('max:' + str(self.test_max) + 'min:' + str(self.test_min))
            print("人脸识别已关闭")
            self.append_edit('人脸识别已关闭')
    def init_face(self):
        '''
        人脸初始化
        :return:
        '''
        if os.path.exists('./data/face_csv/features_all.csv'):
            self.csv_flag = 1
            self.ui.face_predict.setEnabled(True)
        else:
            self.csv_flag = 0
            self.ui.face_predict.setEnabled(False)

    def email_send_fun(self,year, month,day,time_hour,time_minute):
        print("准备发送")
        self.append_edit('准备发送')
        em = Send_Email()
        em.run()
        em.read_image_add('./data/recording/Unknow/{}/{}/{}/Unknow_{}_{}.jpg'.format(year, month,
                                                                                       day,
                                                                                       time_hour,
                                                                                       time_minute), 'Unknow.jpg')
        em.read_image_add('./data/recording/Unknow/{}/{}/{}/Unknow_ROI_{}_{}.jpg'.format(year,
                                                                                            month,
                                                                                            day,
                                                                                            time_hour,
                                                                                            time_minute), 'ROI.jpg')
        em.read_image_add('./data/recording/Unknow/{}/{}/{}/Unknow_mask_{}_{}.jpg'.format(year,
                                                                                            month,
                                                                                            day,
                                                                                            time_hour,
                                                                                            time_minute), 'mask.jpg')
        em.send_email()
        self.append_edit('发送成功')

    def play_music(self):
        file = r'data/mp3/warring.mp3'  # 要播放的歌曲本地地址
        pygame.mixer.init()  # mixer的初始化
        print("警报开启")  # 输出提示要播放的歌曲
        music = pygame.mixer.music.load(file)  # 载入一个音乐文件用于播放

        while self.music_flag:
            # 检查音乐流播放，有返回True，没有返回False
            # 如果没有音乐流则选择播放
            if pygame.mixer.music.get_busy() == False:  # 检查是否正在播放音乐
                pygame.mixer.music.play()  # 开始播放音乐流
                print("重复播放")
                logging.warning('%-60s / %-10s','重复播放 / repeat play',datetime.datetime.now())
    def predict_run(self):
        ret, image = self.cap.read()
        camera_X, camera_Y = int(image.shape[1]), int(image.shape[0])
        if ret:
            # gc.set_debug(gc.DEBUG_LEAK)
            self.predict(self.cap, camera_X, camera_Y)  # 调用人脸检测
        else:
            print('摄像头打开失败')
            self.append_edit('摄像头打开失败')
    def face_train_about(self):
        QMessageBox.about(
            self.ui,
            '操作成功',
            '人脸训练已完成')
    def about(self):
        '''
        弹出开发者相关信息窗口
        :return:
        '''
        QMessageBox.about(self.ui,
                          '关于',
                          f'''计算机设计大赛\n\n深海大菠萝团队作品\n\n成员：钟鹏、黄宇滔、陈勇杰
                            ''')
    def init_progress_bar(self,RANGE):
        ''' 初始化进度条 '''
        self.ui.progressBar.setRange(0,RANGE)
    def progress_bar(self,val):
        ''' 设置进度条进度 '''
        self.ui.progressBar.setValue(val)
    def flag_to_zero(self):
        self.video_flag = 0                     # 判断是否选择打开视频流
        self.video_off_flag = 0                 # 判断是否选择关闭视频流
        self.video_register_flag = 0            # 判断是否选择人脸录入
        self.video_predict_flag = 0             # 判断是否选择人脸检测
        self.video_off_flag = 0                 # 判断是否选择关闭视频流
        self.ui.face_new.setEnabled(False)      # 新建人脸按钮不可用
        self.ui.face_save.setEnabled(False)     # 人脸保存按钮不可用
        self.save_face_flag = 0                 # 重置人脸保存
        self.new_face_flag = 0                  # 重置新建人脸文件夹人脸
        # 重置连续用户\非用户连续人脸计数器
        self.use_count = 0
        self.Unuse_count = 0
        # 重置纵横比值
        self.average_eye_EAR = 0
        self.mouth_MAR = 0
        # 重置眼睛、嘴巴flag记录
        self.eye_flag = 0
        self.mouth_flag = 0
        # 重置眼睛、嘴巴 max、min
        self.eye_ear_min = 1.0
        self.eye_ear_max = 0.0
        self.mouth_mar_min = 1.0
        self.mouth_mar_max = 0.0
        # 重置人脸图像深度
        self.deep = 0


    def save_face_one_to_zero(self):
        self.save_face_flag = 0
    def new_face_one_to_zero(self):
        self.new_face_flag = 0

    def save_face_zero_to_one(self):
        self.save_face_flag = 1

    def new_face_zero_to_one(self):
        self.new_face_flag = 1
    def send(self):
        '''
        发送邮件
        '''
        if self.email_send_flag == 0:
            self.email_send_flag = 1
            self.ui.safe.setText("安全提醒(已开启)")
            QMessageBox.about(
                self.ui,
                '安全提醒更改',
                '安全提醒已开启')
            logging.info('%-60s/ %-10s','安全提醒已开启 / Security reminder is on',datetime.datetime.now())
        else:
            self.email_send_flag = 0
            self.ui.safe.setText("安全提醒(已关闭)")
            QMessageBox.warning(
                self.ui,
                '安全提醒更改',
                '安全提醒已关闭')
            logging.warning('%-60s / %-10s', '安全提醒已关闭 / Security reminder is off', datetime.datetime.now())

    def init_edit(self):
        self.ui.output.clear()
    def move_point(self):
        cursor = self.ui.output.textCursor()
        cursor.movePosition(QTextCursor.End)        # 还可以有别的位置
        self.ui.output.setTextCursor(cursor)

    def append_edit(self,txt):
        '''
        将txt参数打印到textEdit上
        :param txt:文本
        '''

        date_str ='[' + time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime()) + ']  '
        self.ui.output.append(date_str+txt)
        self.move_point()

    def init_video(self):
        '''
        初始化video,将没打开摄像头提示图片弄到label上
        txt : string
        :return:
        '''
        img = cv2.imread('./data/qt/videoOffImg.jpg')
        self.img_to_label_show(img)

    def img_to_label_show(self, img):
        '''
        将图片放到标签上
        :param img: BGR图像
        '''
        '''传入图片显示到label上'''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.ui.video.setPixmap(QPixmap.fromImage(showimg))
        # 按比例填充
        self.ui.video.setScaledContents(True)

    def video_open(self):
        '''
        将视频流放在label上
        '''
        while self.video_flag:
            _, img = self.cap.read()
            self.img_to_label_show(img)
        print("视频流已关闭")
        self.append_edit('视频流已关闭')
        self.init_video()
    def video_open_flag(self):
        '''
        判断是否需要摄像头打开
        '''
        if self.video_flag == 0:
            self.flag_to_zero()
            self.video_flag = 1
            # 创建多线程
            t = Thread(target=self.video_open)
            t.start()
    def video_off(self):
        '''
        关闭视频流
        '''
        if self.video_off_flag == 0:
            self.flag_to_zero()
            time.sleep(0.1)
            self.video_off_flag = 1
            self.init_video()
            print("视频流关闭已打开")
            self.append_edit('视频流关闭已打开')
    def face_register(self):
        '''
        人脸录入
        '''
        if self.video_register_flag == 0:
            self.flag_to_zero()
            self.video_register_flag = 1
            print("人脸录入已打开")
            self.append_edit('人脸录入已打开')
            # 创建多线程
            self.ui.face_new.setEnabled(True)  # 新建人脸按钮可用
            t = Thread(target=self.register_run)
            t.start()

    def face_predict(self):
        '''
        人脸预测
        '''
        if self.video_predict_flag == 0:
            self.flag_to_zero()
            self.video_predict_flag = 1
            print("人脸识别已打开")
            self.append_edit('人脸识别已打开')
            # 创建多线程
            t = Thread(target=self.predict_run)
            t.start()

    def face_train(self):
        t = Thread(target=self.train_run)
        t.start()

    def music(self):
        '''
        报警函数
        '''
        self.play_music()
    def ftp(self,path_a,name_a):
        '''
        ftp发送图片文件到云端进行备份
        :param path: 路径 str
        :param name: 图片名称.jpg  str
        '''
        for path_,name_ in zip(path_a,name_a):
            ftp1 = ftp.ftp()
            ftp1.ftp_connect()
            ftp1.print()
            # ftp1.down_data('Double_loss_vloss.png','Double_loss_vloss.png')
            # ftp1.up_data('Web.rar','Web.rar')
            path = ftp1.double_file_path(path_, name_)
            print(path_, path + '/' + name_)
            ftp1.up_data(path_, path + '/' + name_)
            ftp1.quit()
    def to_mask(self,img_path):
        '''
        将图片进行人体分割并保存
        :param img_path: 图片路径
        :return:Image对象
        '''
        inputs = cv2.imread(img_path)
        # inputs = cv2.cvtColor(inputs,cv2.COLOR_RGB2BGR)
        Image.fromarray(np.uint8(inputs))
        output = remove(inputs)
        return output
    def send_mysql(self,name,img_path,img_mask_path,img_roi_path):
        '''
        Thread
        发送数据到mysql
        :param img_path:img路径
        :param img_mask_path: mask路径
        :param img_roi_path: ROI路径
        '''
        self.sql.load_path(name,img_path,img_mask_path,img_roi_path)
        self.sql.post_mysql_data()
    def select_mysql(self):
        '''
        查询指定日期
        Returns:
        '''
        # 初始化列表
        self.data_init()
        # 获取选择日期
        year, month, day = self.get_date()
        choose_date = datetime.date(year=year,month=month,day=day)
        print(choose_date)
        res = self.sql.select_date(choose_date)
        print(res)
        self.add_tableWidget(res)
    def sql_init(self):
        '''
        初始化数据库ui
        '''
        self.SQL = mySQL()
        self.SQL.ui.show()

    def flush(self):
        '''
        刷新打印所有记录
        :return:
        '''
        # 初始化列表
        self.data_init()
        res = self.sql.select_all()
        self.add_tableWidget(res)
    def add_tableWidget(self,res):
        '''
        将查询数据打印到单元格中
        Parameters
        ----------
        res     tuple ((time, user, photo_all_path, photo_mask_path, photo_roi_path),....)

        Returns
        -------

        '''

        if res != ():
            self.loacl_data_check(res)
            # 将数据放在ui列表上
            # 插入的行号
            lineNo = 0
            # self.ui.tableWidget.resizeColumnToContents(0)
            # self.ui.tableWidget.resizeColumnToContents(1)
            # self.ui.tableWidget.resizeRowToContents(0)
            # self.ui.tableWidget.resizeRowToContents(1)
            for time, user, photo_all_path, photo_mask_path, photo_roi_path in res:
                time = str(time)
                print(photo_all_path, photo_mask_path, photo_roi_path)
                self.ui.tableWidget.insertRow(lineNo)
                item1 = QTableWidgetItem()
                pixmap1 = QPixmap(photo_all_path)
                item1.setData(Qt.DecorationRole, pixmap1)
                width1 = pixmap1.width()
                height1 = pixmap1.height()

                item2 = QTableWidgetItem()
                pixmap2 = QPixmap(photo_mask_path)
                item2.setData(Qt.DecorationRole,pixmap2)
                width2 = pixmap2.width()
                height2 = pixmap2.height()

                item3 = QTableWidgetItem()
                pixmap3 = QPixmap(photo_roi_path)
                item3.setData(Qt.DecorationRole,pixmap3)
                width3 = pixmap3.width()
                height3 = pixmap3.height()
                # size = pixmap1.sizeHint()
                # print(width1,width2,width3,height1,height2,height3)
                self.ui.tableWidget.setItem(lineNo, 0, QTableWidgetItem(time))
                self.ui.tableWidget.setItem(lineNo, 1, QTableWidgetItem(user))
                self.ui.tableWidget.setItem(lineNo, 2, item1)
                self.ui.tableWidget.setItem(lineNo, 3, item2)
                self.ui.tableWidget.setItem(lineNo, 4, item3)

                # 调整单元格大小以适应图片
                # self.ui.tableWidget.setRowHeight(0, height1)
                # self.ui.tableWidget.setColumnWidth(0, width1)

                self.ui.tableWidget.setRowHeight(lineNo, height1)
                self.ui.tableWidget.setColumnWidth(2,width1)
                self.ui.tableWidget.setColumnWidth(3, width2)
                self.ui.tableWidget.setColumnWidth(4, width3)
                lineNo += 1

    def date_init(self):
        '''
        初始化日期
        '''
        a = datetime.datetime.now()
        date = QDate(a.year, a.month, a.day)
        self.ui.dateEdit.setDate(date)
    def get_date(self):
        '''
        获取日期
        Returns : year, month, day
        '''
        date = self.ui.dateEdit.date()
        year = date.year()
        month = date.month()
        day = date.day()
        return year, month, day
    def put_data(self):
        '''
        放置日期，将日期默认设置为当天
        :return:
        '''
        self.date_init()
    def data_init(self):
        '''
        初始化列表数据
        '''
        # 清除列表数据
        self.ui.tableWidget.clearContents()
        self.ui.tableWidget.setRowCount(0)
    def cap_exit(self):
        '''
        退出程序
        释放多线程与摄像资源
        '''
        self.flag_to_zero()
        self.cap.release()
    def loacl_data_check(self,res):
        '''
        检测本地是否存在对应文件,不存在则云端下载
        Parameters
        ----------
        res : time,user,photo_all,photo_mask,photo_roi

        -------

        '''
        ftp_d = ftp.ftp()
        for time,user,photo_all_path,photo_mask_path,photo_roi_path in res:

            ftp_d.ftp_connect()
            path = str('/'.join(photo_all_path.split('/')[0:7]))
            # 判断是否存在父路径
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(photo_all_path):
                # 下载
                ftp_d.down_data(photo_all_path,photo_all_path)
            if not os.path.exists(photo_mask_path):
                # 下载
                ftp_d.down_data(photo_mask_path, photo_mask_path)
            if not os.path.exists(photo_roi_path):
                # 下载
                ftp_d.down_data(photo_roi_path, photo_roi_path)
        ftp_d.quit()
        print('查询数据本地与云端已同步!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if not os.path.exists('./data/logging'):
        try:
            os.makedirs('./data/logging')
        except:
            pass
    if os.path.exists('./data/logging/face.log'):
        logging.basicConfig(level=logging.INFO,filename='./data/logging/face.log',filemode='a+')
    else:
        logging.basicConfig(level=logging.INFO, filename='./data/logging/face.log', filemode='w')
    window = Stats()
    window.ui.show()
    # window.cap_exit()
    # print(app.exec_())
    a = app.exec_()
    if a == 0:
        print(a)
        window.cap_exit()
    logging.info('%-60s / %-10s\n\n\n','正常退出程序 / The program has exited normally',datetime.datetime.now())
    sys.exit()

