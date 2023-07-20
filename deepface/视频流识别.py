from deepface import DeepFace
import cv2
cap = cv2.VideoCapture(0)
ret,img = cap.read()
a=DeepFace.stream()
print(a)
