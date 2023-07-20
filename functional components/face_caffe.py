import cv2
import numpy as np
#加载预训练模型，caffe
net = cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000_fp16.caffemodel')

cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while(1):
    ret,img = cap.read()
    w = img.shape[1]
    h = img.shape[0]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #为了获得最佳精度，必须分别对蓝色、绿色和红色通道执行 (104, 177, 123) 通道均值减法，并将图像调整为 300 x 300 的 BGR 图像，在 OpenCV 中可以通过使用 cv2.dnn.blobFromImage() 函数进行此预处理
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)
    #将 blob 设置为输入以获得结果，对整个网络执行前向计算以计算输出
    net.setInput(blob)
    detections = net.forward()
    #最后一步是迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化：
    # 迭代所有检测结果
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
            # 绘制检测结果和置信度
            text = "{:.3f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 3)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # 可视化
    cv2.imshow('output',img)
    cv2.waitKey(1)