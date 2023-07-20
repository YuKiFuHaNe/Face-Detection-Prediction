import cv2
import json
import os
import matplotlib.pyplot as plt
def pred(pred_face_path,label):
    """
    导入训练好的训练集进行人脸识别匹配
    :param camera_id: 摄像头id
    :return:
    """
    with open('index_to_label.json','r') as f:
        index_to_label = json.load(f)
    img = cv2.imread(pred_face_path,0)
    cv2.imshow('input={}'.format(label),img)
    cv2.waitKey(0)
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
                print('图片人物:{}'.format(label))
                print('----------------------------------------------------------------------------------------------')
                return
    except:
        print('出错啦！')
    id, picture = recognizer.predict(img_resize)
    print('图片-->', id, '相似度评分：', picture)
    id = str(id)
    name = index_to_label.get(id)
    print('识别ID：{}为{}'.format(id,name))
    print('########这图片人物的的真实名称{}########'.format(label))
    # cv2.imshow(name,img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.xml')
    print("模型加载 OK")
    path = os.listdir('./pred')
    all_img_path = []
    for dir_item in path:
        # full_path = os.path.abspath(os.path.join( dir_item))
        full_path = './' + 'pred/' + os.path.join(dir_item)
        all_img_path.append(full_path)
    print(all_img_path)
    all_img_label = [label.replace('\\','/').split('/')[-1].split('_0')[0] for label in all_img_path]
    p = zip(all_img_path,all_img_label)
    for pred_face_path,label in p:
        pred_face_path = pred_face_path.replace('\\','/')
        print('###  {}  --  {}  ###'.format(pred_face_path,label))
        pred(pred_face_path,label)