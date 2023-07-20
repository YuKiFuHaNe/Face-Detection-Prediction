from deepface import DeepFace

img_path1 = r'img.png'

img_path2 = r'img_1.png'

img_path1 = img_path1.replace('\\', '/')

img_path2 = img_path2.replace('\\', '/')

models=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

db_path = r'data'

db_path = db_path.replace('\\', '/')

recognition = DeepFace.find(img_path = img_path2, db_path = db_path, model_name = models[0], enforce_detection=False)

print(recognition)
