import cv2
import json
import os
import matplotlib.pyplot as plt
def pred(pred_face_path):
    """
    导入训练好的训练集进行人脸识别匹配
    :param camera_id: 摄像头id
    :return:
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.xml')
    print("模型加载 OK")
    with open('index_to_label.json','r') as f:
        index_to_label = json.load(f)
    img = cv2.imread(pred_face_path,0)
    face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face.detectMultiScale(img)
    img_resize = 0
    try:
        if len(faces) > 1:
            area = 0
            x1=0
            w1=0
            y1=0
            h1=0
            for x, y, w, h in faces:
                if (w*h)>area:
                    x1,y1,w1,h1 = x, y, w, h
                    area = w1*h1
            img_resize =  cv2.resize(img[y1-20:y1+h1+30,x1-15:x1+w1+15],(150,150))
        elif len(faces)==1:
            for x, y, w, h in faces:
                img_resize = cv2.resize(img[y-20:y+h+30,x-15:x+w+15],(150,150))
        else:
            if img_resize == 0:
                print("未识别到人脸")
                print('图片路径为:{}'.format(pred_face_path))
                print('----------------------------------------------------------------------------------------------')
                return
    except:
        print('出错啦！')
    id, picture = recognizer.predict(img_resize)
    print('图片-->', id, '相似度评分：', picture)
    id = str(id)
    name = index_to_label.get(id)
    print('识别ID：{}为{}'.format(id,name))
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    pred('C:/Users/91522/Desktop/python_Project/A17/Aaron_Eckhart_0001.jpg')