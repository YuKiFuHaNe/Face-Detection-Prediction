'''
运行main(参数1:cap)
'''
import cv2
import os
import pandas as pd
import numpy as np
import dlib
import time
import logging

#加载预训练模型，caffe
net = cv2.dnn.readNetFromCaffe('./data/caffemodel/deploy.prototxt','./data/caffemodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')
# face = cv2.CascadeClassifier('caffemodel/haarcascade_frontalface_alt.xml')
class Face_Register:
    def __init__(self):
        # FACE_PATH
        self.faces_path = './data/face'        # 设置人脸存储路径
        # FONT
        self.font = cv2.FONT_ITALIC             # 设置字体类型
        # FACE
        self.existing_faces_cnt = 0             # 现存人脸张数
        # self.camera_preson_face = 0           # 摄像头识别到的人脸个数
        self.person_face_cnt = 0                # 将该用户人脸计数器
        # FLAG
        self.press_n_flag = 0                   # 是否按下'n'创建人脸， 0：否  1：是
        self.save_flag = 0                      #人脸是否符合界限       0：否  1：是
        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

    # 新建人脸存储路径文件夹
    def check_face_folder(self):
        #判断是否存在人脸存储路径
        if os.path.isdir(self.faces_path):
            pass
        else:
            os.mkdir(self.faces_path)
    # 检查现存人脸数
    def check_existing_faces_cnt(self):
        if os.listdir(self.faces_path):             # 存在用户人脸数据
            person_list = os.listdir(self.faces_path)
            map(int,person_list)
            self.existing_faces_cnt = int(max(person_list))
            print("face:"+"{}".format(self.existing_faces_cnt))
        else:
            self.existing_faces_cnt = 0

    # 更新 FPS
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time   #开始时间减现在时间为时间间隔
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img):
        # 添加说明 / Add some notes
        # cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "FPS: " + str(self.fps_show.__round__(2)), (10, 20), self.font, 0.8, (0, 255, 0), 1,cv2.LINE_AA)                    # 调用时返回2位小数
        # cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)  # cv2.LINE_AA:抗锯齿线
        # cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 获取人脸
    def process(self, cap,camera_X,camera_Y):
        self.check_face_folder()              # 检查是否存在人脸文件根
        self.check_existing_faces_cnt()
        key = cv2.waitKey(1)
        while cap.isOpened():
            ret, img = cap.read()
            img_copy = img.copy()
            w = img.shape[1]
            h = img.shape[0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if key == ord('n'):                         # 如果按下n，创建人脸文件  self.press_n_flag 0->1
                self.existing_faces_cnt += 1            # 现存人脸数+1
                current_face_dir = '{}/{}'.format(self.faces_path,self.existing_faces_cnt)
                os.mkdir(current_face_dir)
                logging.info("\n%-40s %s", "新建的人脸文件夹 / Create folders:", current_face_dir)
                self.person_face_cnt = 0                # 将人脸计数器清零
                self.press_n_flag = 1                   # 已经按下 'n'
                # print(self.person_face_cnt)
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
                    camera_preson_face.append((startX,startY,endX,endY))
                    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # 获取人脸个数
            num = len(camera_preson_face)
            # print(camera_preson_face)
            # print(num)
            #检测到无人脸
            if num == 0:
                cv2.putText(img, 'Place the face in the middle of the camera', (60,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)     # 提示窗口无人脸
            #检测到多人脸
            elif num >1:
                cv2.putText(img, 'Do not have two faces', (200, 150), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2)                          # 提示不要出现两张人脸
            else:
                # 检测到符合要求一张人脸
                # 判断头像是否符合规范（未出界）
                (startX,startY,endX,endY) in camera_preson_face
                face_width = endX - startX
                face_hight = endY - startY
                face_width_c2 = int(face_width/2)
                face_hight_c2 = int(face_hight/2)
                if (endX + face_width_c2)>camera_X or (endY + face_hight_c2)>camera_Y or (startX - face_width_c2)<0 or (startY - face_hight_c2)<0:
                    #超出界限
                    cv2.putText(img, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    self.save_flag = 0
                else:
                    self.save_flag = 1

                if self.save_flag:                                      # 人脸符合界限
                    if key == ord('s'):
                        # 根据人脸大小生成空的图像
                        img_blank = np.zeros((int(face_hight * 2), face_width * 2, 3), np.uint8)
                        if self.press_n_flag:                           # 之前是否按下n
                            self.person_face_cnt += 1
                            for i in range(face_hight * 2):             # 上下个取半个face_hight
                                for j in range(face_width * 2):         # 左右个取半个face_hight
                                    img_blank[i][j] = img_copy[startY - face_hight_c2 + i][startX - face_width_c2 + j]
                            cv2.imwrite('{}/{}/{}.jpg'.format(self.faces_path,self.existing_faces_cnt,self.person_face_cnt), img_blank)
                            # cv2.imwrite('{}/{}/{}111.jpg'.format(self.faces_path, self.existing_faces_cnt, self.person_face_cnt),img_copy[startY:startY+face_hight,startX:startX+face_width])
                            logging.info("%-40s %s/img_face_%s.jpg", "写入本地 / Save into：",str(current_face_dir), str(self.person_face_cnt))
                else:
                    pass

            # 按下q退出
            if key == ord('q'):
                break
            #  更新 FPS
            self.update_fps()
            # 添加文字
            self.draw_note(img)

            # 可视化
            cv2.imshow('output', img)
            key = cv2.waitKey(1)
            # cv2.namedWindow("camera", 1)
            # cv2.imshow("camera", img)
    def run(self,cap):
        ret,image = cap.read()
        camera_X,camera_Y = int(image.shape[1]),int(image.shape[0])
        if ret:
            self.process(cap,camera_X,camera_Y)               # 调用人脸识别
        else:
            print('摄像头打开失败')
        # cap.release()
        cv2.destroyAllWindows()
def main(index):
    if os.path.isdir('./data'):
        pass
    else:
        os.mkdir('./data')
    # logging.basicConfig(filename='log.txt',filemode='a+',level=logging.INFO)
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    cap = cv2.VideoCapture(index)  # 获取摄像头
    Face_Register_con.run(cap)
if __name__ == '__main__':
    main(0)