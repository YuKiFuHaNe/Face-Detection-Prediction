import cv2
import dlib

#读入摄像头数据
cap=cv2.VideoCapture(0)


predictor_path = "shape_predictor_68_face_landmarks.dat"

#使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)

#使用dlib自带的frontal_face_detector作为人脸检测器
detector = dlib.get_frontal_face_detector()

while True:
    _,frame=cap.read()
    dets = detector(frame, 1)
    print(dets)
    '''
    功能：对图像画人脸框
    参数：img_gray：输入的图片
    返回值：人脸检测矩形框4点坐标
    '''
    if len(dets) != 0:
        shape = predictor(frame, dets[0])
        for p in shape.parts():
            print(p)
            cv2.circle(frame, (p.x, p.y), 3, (0,0,255), -1)
    cv2.imshow('video',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()