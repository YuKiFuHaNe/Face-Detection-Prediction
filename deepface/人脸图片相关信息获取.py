from deepface import DeepFace

analysis = DeepFace.analyze(img_path="./test/Aaron_Eckhart_0001.jpg", actions=["age", "gender", "emotion", "race"])
print(analysis)
