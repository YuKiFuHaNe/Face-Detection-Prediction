'''
python>=3.9
'''
from rembg import remove
import cv2
from PIL import Image
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    img = Image.fromarray(np.uint8(img))
    out = remove(img)
    out = np.array(out)
    cv2.imshow('remove',out)
    cv2.imwrite('a.jpg',out)
    cv2.waitKey(1)
