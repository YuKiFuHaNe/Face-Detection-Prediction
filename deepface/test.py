from deepface import DeepFace
import glob
import cv2
face_path = glob.glob("E:/A17_P/FACE/*.jpg")
name = [picture_name.replace("\\","/").split("FACE/")[1].split("_0")[0] for picture_name in face_path]
count = len(face_path)
print(len(face_path))
print(name)
