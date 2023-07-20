import os
import cv2
import numpy as np
from PIL import Image
import json
def train_img(path):
    """
    进行训练
    :param path:
    :return:
    """
    all_image_path = []
    #调用训练集
    face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    #图片集
    picture = []
    #标签集
    ids = []
    #图片集合
    # Imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(Imagepaths)
    #循环每一张图片
    for dir_item in os.listdir(path):
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path, dir_item))
        all_image_path.append(full_path)
    # print(all_image_path)
    all_labels = [label.replace('\\', '/').split('/')[3].split('.')[0].split('_0')[0] for label in all_image_path]
    # print(all_labels)
    # print(len(all_labels))
    label_names = np.unique(all_labels)
    # print(len(label_names))
    label_to_index = dict((name, i) for i, name in enumerate(label_names))
    index_to_label = dict((i, name) for name, i in label_to_index.items())
    all_index = [label_to_index.get(index) for index in all_labels]
    ids = all_index
    with open('index_to_label.json','w') as f:
        json.dump(index_to_label,f)
    for image_path in all_image_path:
        img = cv2.imread(image_path,0)
        # img = np.cast(img,np.uint8)
        picture.append(img)

    faces = picture
    ids = ids
    #创建训练
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #进行训练
    recognizer.train(faces, np.array(ids))
    #写出训练后的数据
    recognizer.write('train.xml')
    np.save('features.npy', faces)
    np.save('labels.npy', ids)
    print("Model Save")

if __name__ == '__main__':
    train_img('E:/A17_P/FACE_RESIZE')