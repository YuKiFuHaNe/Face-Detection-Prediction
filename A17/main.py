import cv2
import A17_face_register,A17_face_train,A17_face_predict


index = 0
cap = cv2.VideoCapture(index)
while True:
    keyb = cv2.waitKey(1)
    temp = int(input())
    if temp == 1:
        A17_face_register.main(cap)
    elif temp == 2:
        A17_face_train.run()
    elif temp == 3:
        A17_face_predict.main(cap)