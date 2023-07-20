from deepface import DeepFace
import cv2

models = ["VGG-Face","Facenet","Facenet512","OpenFace","DeepFace","DeepID","ArcFace","Dlib","SFace"]
verfification = DeepFace.verify(img1_path="./test/Aaron_Peirsol_0004.jpg",img2_path="./test/Aaron_Peirsol_0002.jpg",model_name=models[4])
print(verfification)