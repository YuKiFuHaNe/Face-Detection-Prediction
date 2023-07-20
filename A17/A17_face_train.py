'''
 运行run()
'''
import cv2
import dlib
import numpy as np
import pandas as pd
import logging
import csv
import os



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



def return_128d_features(photo_path):
    img = cv2.imread(photo_path)
    logging.info("%-40s %-20s", "检测到人脸的图像 / Image with faces detected:", photo_path)
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
            camera_preson_face.append((startX,startY,endX,endY))
            # 绘制检测结果和置信度
            # text = "{:.3f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 3)
            # cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    if len(camera_preson_face) == 1:
        (startX, startY, endX, endY) in camera_preson_face
        face_width_c5 = int((endX - startX)/5)
        face_hight_c5 = int((endY - startY)/5)
        # rectangle = dlib.rectangle(startX-face_width_c5, startY-face_hight_c5, endX+face_width_c5, endY+face_hight_c5)
        rectangle = dlib.rectangle(startX, startY, endX,endY)
        shape = predictor(img, rectangle)
        # 查看68point
        # for p in shape.parts():
        #     cv2.circle(img, (p.x, p.y), 3, (0,0,255), -1)
        # cv2.imshow('video', img)
        # cv2.waitKey(0)
        face_descriptor = face_128d_model.compute_face_descriptor(img,shape)
    else:
        face_descriptor = 0
        logging.warning("No Face")
    return face_descriptor


def return_features_mean_personX(personX_face_path):
    features_list_personX = []                                  # 存储特征值
    face_photos_list = os.listdir(personX_face_path)            # 获取personX对应的人脸图片路径
    if face_photos_list:
        for i in range(len(face_photos_list)):
            # 调用 return_128d_features 得到图片128D特征矢量值
            logging.info("%-40s %-20s", "正在读的人脸图像 / Reading image:", personX_face_path + "/" + face_photos_list[i])
            photo_path = os.path.join(personX_face_path,face_photos_list[i])
            features_128d = return_128d_features(photo_path)    # 返回0或128d矢量  0：无人脸
            # 遇到没有检查到的人脸图片就跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning("文件夹内图像文件为空 / Warning: No images in%s/", personX_face_path)
    # 计算 128D 特征的均值 / Compute the mean
    # personX 的 N 张图像 x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX

def run():
    logging.basicConfig(level=logging.INFO)
    person_list = os.listdir(faces_path)
    face_name = set()
    person_list.sort()                     # 将人顺序从低到高排序
    with open("./data/face_csv/features_all.csv", "r+", newline="") as csvfile:
        write = csv.writer(csvfile)
        # 获取csv中存在的人脸
        for line in csvfile.readlines():
            # print(type(line[0]))
            face_name.add(line[0])
        print(face_name)

        for person_path in person_list:
            if str(person_path) not in face_name:
                # 获取person对应的128D
                logging.info("%s_person_%s",faces_path,person_path)
                personX_face_path = os.path.join(faces_path,person_path)
                features_mean_personX = return_features_mean_personX(personX_face_path)
                # 获取对应person编号
                person_name = person_path
                features_mean_personX = np.insert(features_mean_personX,0,person_name,axis=0)          #a=np.insert(arr, obj, values, axis)#arr原始数组，可一可多，obj插入元素位置，values是插入内容，axis是按行按列插入（0：行、1：列）
                write.writerow(features_mean_personX)
                logging.info('\n')
        logging.info("所有录入人脸数据存入 / Save all the features of faces registered into: data/face_csv/shape_predictor_68_face_landmarks.dat")
if __name__ == '__main__':
    run()