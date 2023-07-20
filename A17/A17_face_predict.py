'''
运行main(参数1:摄像头cap)
'''

import cv2
import pandas as pd
import numpy as np
import dlib
import os
import logging
import time
import datetime
import gc
from imutils import face_utils
from scipy.spatial import distance as dist
# 获取嘴巴坐标参数值
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
# 获取人眼的坐标  获取左右人眼的坐标参数值
(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart,rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# 设置嘴巴横纵比阈值
MAR_THRESH = 0.48
# 设置眼睛横纵比阈值
EYE_AR_THRESH = 0.3                         # 根据人眼大小来调整
# 设置eye横纵比阈值差值
# 0.03~0.04
EYE_MIN = 0.165
EYE_MAX_MIN_DELTA = 0.28                    #
MOUTH_MIN = 0.3
MOUTH_MAX_MIN_DELTA = 0.55                  # 0.45

import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from cvzone.FaceMeshModule import FaceMeshDetector   #导入检测器
detector = FaceMeshDetector(maxFaces=1)  # 检测人脸

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
# 设至深度范围
DEEP_STRAT_RANGE = 35.0             # 35.0
DEEP_END_RANGE = 50.0

# 设置人脸位置
faces_path = './data/face'

#加载预训练模型，caffe
net = cv2.dnn.readNetFromCaffe('./data/caffemodel/deploy.prototxt','./data/caffemodel/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# 人脸68关键点模型
predictor_path = "./data/predictor_68_face/shape_predictor_68_face_landmarks.dat"
#使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)


# 加载人脸128D模型
face_128d_model = dlib.face_recognition_model_v1('./data/128d_model/dlib_face_recognition_resnet_model_v1.dat')
class Face_Predict:
    def __init__(self):
        # FACE_PATH
        self.faces_path = './data/face'         # 设置人脸存储路径
        # FONT
        self.font = cv2.FONT_ITALIC             # 设置字体类型
        # FACE
        self.face_name_know_list = []           # 现存人脸张数
        self.face_feature_know_list = []        # 存放所有录入人脸的特征组
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

        self.use_count = 0                      # 用户人脸连续出现次数
        self.Unuse_count = 0                    # 未知人脸连续出现次数

        # biopsy testing
        self.average_eye_EAR = 0                # 眼睛的纵横比值
        self.mouth_MAR = 0                      # 嘴巴纵横比值
        self.eye_flag = 0                       # 判定眼睛是否为活体   0：否  1：是
        self.mouth_flag = 0                     # 判定嘴巴是否为活体   0：否  1：是
        self.eye_max = 0

        self.eye_ear_max = 0.0
        self.eye_ear_min = 1.0
        self.mouth_mar_max = 0.0
        self.mouth_mar_min = 1.

        self.deep = 0                           # 设置人脸图像深度
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
        cv2.putText(img, "FPS: " + str(self.fps_show.__round__(2)), (10, 20), self.font, 0.8, (0, 255, 0), 1,cv2.LINE_AA)                    # 调用时返回2位小数
        # cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)  # cv2.LINE_AA:抗锯齿线
        # cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 从features_all.csv 读取人脸特征
    def get_faces_features(self):
        if os.path.exists("./data/face_csv/features_all.csv"):
            faces_features_path = './data/face_csv/features_all.csv'
            csv = pd.read_csv(faces_features_path,header=None)
            for person_name in range(csv.shape[0]):
                features_someone_arr = []
                self.face_name_know_list.append(csv.iloc[person_name][0])
                for featires_128d in range(1,129):
                    if csv.iloc[person_name][featires_128d] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv.iloc[person_name][featires_128d])
                self.face_feature_know_list.append(features_someone_arr)
            logging.info("Face in load:%d",len(self.face_feature_know_list))
            return 1
        else:
            logging.warning('"features_all.csv" not found!')
            logging.warning('Please run "A17_face_register"->"A17_face_train" last runing "A17_face)predict"')
            return 0
    # 计算两个128D向量间的欧式距离
    def return_enclidean_distance(self,feature_1,feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def face_deep(self,img):
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
            cv2.putText(img, "Deep: " + '{:.2f}'.format(d), (10, 100), self.font, 0.8, (0, 255, 0), 1,
                        cv2.LINE_AA)


    def mouth_aspect_ratio(self,mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)

        return mar

    def eye_aspect_ratio(self,eye):
        A = dist.euclidean(eye[1], eye[5])  # 计算两组 垂直 眼界标之间的距离
        B = dist.euclidean(eye[2], eye[4])  # 计算两组 垂直 眼界标之间的距离
        C = dist.euclidean(eye[0], eye[3])  # 计算水平眼界标之间的距离
        ear = (A + B) / (2.0 * C)
        return ear




    # 处理获取的视频流，进行人脸识别
    def predict(self,cap, camera_X, camera_Y):
        if self.get_faces_features():
            while cap.isOpened():
                self.frame_time +=1
                _,img = cap.read()
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
                    cv2.putText(img, 'Place the face in the middle of the camera', (60, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)  # 提示窗口无人脸
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
                    cv2.putText(img, 'Do not have two faces', (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                                2)  # 提示不要出现两张人脸
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
                        cv2.putText(img, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        # 进行人脸测距
                        self.face_deep(img)
                        if not (DEEP_STRAT_RANGE <= self.deep <= DEEP_END_RANGE):
                            cv2.putText(img, "KEEP DEEP in 35~50", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        else:
                            rectangle = dlib.rectangle(startX, startY, endX, endY)
                            shape = predictor(img, rectangle)                               # 获取68点值 shape值

                            shape_np = face_utils.shape_to_np(shape)                        # 68点值 shape值 array

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
                            if MOUTH_MIN < (self.mouth_mar_max - self.mouth_mar_min) < MOUTH_MAX_MIN_DELTA:
                                self.mouth_flag = 1
                            else:
                                self.mouth_flag = 0
                            print('eye:{} ---- mouth:{}'.format(self.eye_ear_max-self.eye_ear_min,self.mouth_mar_max-self.mouth_mar_min))

                            if self.eye_flag == 1 and self.mouth_flag == 1:

                                # 人脸模型128D特征数据提取
                                face_descriptor = face_128d_model.compute_face_descriptor(img, shape)
                                current_frame_e_distance_list = []
                                # 对于某张人脸，遍历所有存储的人脸特征
                                for i in range(len(self.face_feature_know_list)):
                                    # 如果 person_X not None
                                    if str(self.face_feature_know_list[i][0]) != '0.0':
                                        e_distance_tmp = self.return_enclidean_distance(face_descriptor,self.face_feature_know_list[i])               # 调用 return_enclidean_distance 计算两个128D向量间的欧式距离
                                        current_frame_e_distance_list.append(e_distance_tmp)
                                    else:
                                        # 空数据
                                        current_frame_e_distance_list.append(999999999)
                                # 寻找出最小的欧式距离匹配
                                similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))                          # 获取最小的下标
                                # print(current_frame_e_distance_list)
                                if min(current_frame_e_distance_list) < 0.34:
                                    cv2.putText(img,"person_{}".format(similar_person_num),(startX,startY-10),self.font,0.8,(0,0,255),2)
                                    self.use_count += 1
                                else:
                                    cv2.putText(img, "Unknow", (startX, startY - 10), self.font,0.8, (0, 0, 255), 2)
                                    self.Unuse_count += 1
                                if self.use_count == 10 or self.Unuse_count == 10:
                                    # 对图像进行背景替换
                                    imgOut = segmentor.removeBG(img_copy, cv2.imread('./data/white/480_640.jpg'), threshold=0.8)

                                    imgStack = cvzone.stackImages([img_copy, imgOut], 2, 1)
                                    _, imgStack = fpsReader.update(imgStack)
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
                                        if not os.path.isdir('./data/recording/use/{}/{}/{}'.format(year,month,day)):
                                            os.makedirs('./data/recording/use/{}/{}/{}'.format(year,month,day))
                                        cv2.imwrite('./data/recording/use/{}/{}/{}/use{}_{}_{}.jpg'.format(year,month,day,similar_person_num,time_hour,time_minute),img_copy)
                                        cv2.imwrite('./data/recording/use/{}/{}/{}/use{}_ROI_{}_{}.jpg'.format(year, month, day,similar_person_num,time_hour,time_minute),img_copy[startY:endY,startX:endX])
                                        cv2.imwrite('./data/recording/use/{}/{}/{}/use{}_mask_{}_{}.jpg'.format(year, month, day,similar_person_num,time_hour,time_minute),imgStack)

                                    else:
                                        if not os.path.isdir('./data/recording/Unknow/{}/{}/{}'.format(year,month,day)):
                                             os.makedirs('./data/recording/Unknow/{}/{}/{}'.format(year,month,day))
                                        cv2.imwrite('./data/recording/Unknow/{}/{}/{}/Unknow{}_{}_{}.jpg'.format(year, month, day,similar_person_num,time_hour,time_minute),img_copy)
                                        cv2.imwrite('./data/recording/Unknow/{}/{}/{}/Unknow{}_ROI_{}_{}.jpg'.format(year, month, day,similar_person_num,time_hour,time_minute),img_copy[startY:endY, startX:endX])
                                        cv2.imwrite('./data/recording/Unknow/{}/{}/{}/Unknow{}_mask_{}_{}.jpg'.format(year, month, day,similar_person_num,time_hour, time_minute),imgStack)
                            # print(gc.garbage)
                            # print()
                            # print(gc.collect())
                # 更新FPS
                self.update_fps()
                # 添加文字
                self.draw_note(img)
                cv2.imshow('img',img)
                cv2.waitKey(1)
            print('max:' + str(self.test_max) +'min:'+str(self.test_min))

    def run(self,cap):
        ret, image = cap.read()
        camera_X, camera_Y = int(image.shape[1]), int(image.shape[0])
        if ret:
            # gc.set_debug(gc.DEBUG_LEAK)
            self.predict(cap, camera_X, camera_Y)  # 调用人脸检测
        else:
            print('摄像头打开失败')
        cap.release()
        cv2.destroyAllWindows()
def main(cap):
    logging.basicConfig(level=logging.INFO)
    face_predict = Face_Predict()
    face_predict.run(cap)
if __name__ == '__main__':
    main(0)