from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
'''
· 口可以访问 [48，68] 。
· 右眉可以访问 [17，22]。
· 左眉可以访问 [22，27] 。
· 右眼可以访问 [36，42]。
· 左眼 可以访问 [42，48]。
· 鼻可以访问 [27，35]。
· 下巴边框可以访问 [0，17]
'''
# 获取人眼坐标,计算人眼纵横比
# 当检测到人眼时，通过实时计算人眼的纵横比，当人眼纵横比突然变小时，便可以通过此值来判断人眼眨眼了，也可以证明人脸不是照片

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])       # 计算两组 垂直 眼界标之间的距离
    B = dist.euclidean(eye[2],eye[4])       # 计算两组 垂直 眼界标之间的距离
    C = dist.euclidean(eye[0],eye[3])       # 计算水平眼界标之间的距离
    ear = (A + B) / (2.0 * C)
    return ear

# 人眼纵横比参数
EYE_AR_THRESH = 0.3                         # 根据人眼大小来调整
EYE_AR_CONSEC_FRAMES = 3                    # 定义检测到眨眼的帧数（设置为3，当3次检测到阈值小于设置值才算是一次正常的眨眼，也跟计算速度有关）

# 眨眼次数及帧次数
COUNTER = 0
TOTAL = 0

# 加载正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载人脸68点数据模型
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 获取人眼的坐标  获取左右人眼的坐标参数值
(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart,rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cap = cv2.VideoCapture(0)
while True:
    if cap.isOpened():
        _,frame = cap.read()
        # frame = imutils.resize(frame,width=450)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rects = detector(gray,0)
        for rect in rects:
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            # 提取人眼坐标。来计算人眼纵横比
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # 平均左右眼的纵横比
            ear = (leftEAR + rightEAR) / 2.0
            # 显示左右眼
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
            cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
            # 计算ear是否小于设置的值
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            #当大于时,判断是否超过3个帧
            else:
                # 当大于3个帧都要小于设置的值时,才定义一次眨眼
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0
            # 在视频上显示眨眼次数以及横纵比
            cv2.putText(frame,'blinks:{}'.format(TOTAL),(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame,'EAR:{:.2f}'.format(ear),(300,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            cv2.imshow('Frame',frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
cv2.destroyAllWindows()
cap.release()