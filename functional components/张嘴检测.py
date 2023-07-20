from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2


# 加载正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载人脸68点数据模型
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

MAR_THRESH = 0.5

cap = cv2.VideoCapture(0)
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)

    return mar

while True:
    if cap.isOpened():
        _,frame = cap.read()
        # frame = imutils.resize(frame,width=450)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rects = detector(gray,0)

        for rect in rects:
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame,[mouthHull],-1,(0,255,0),1)

            if mar > MAR_THRESH:
                cv2.putText(frame,'Mouth is open!',(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame,'MAR:{:.2f}'.format(mar),(300,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            key = cv2.waitKey(1)
            cv2.imshow('Frame',frame)
            if key == ord('q'):
                break
cv2.destroyAllWindows()
cap.release()
