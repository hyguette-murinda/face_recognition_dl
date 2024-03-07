import cv2
import os
import numpy as np
from PIL import Image

# Create LBPH Face Recognizer
#pip install opencv-contrib-python
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade Classifier for face detection
detector = cv2.CascadeClassifier("../models/haar.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faceSamples = []
    Ids = []
    names = []
    ages = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            #print(Id)
            name = str(os.path.split(imagePath)[-1].split(".")[2])
            #print(name)
            age = int(os.path.split(imagePath)[-1].split(".")[3])
            # print (age)
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
                #print(Ids)
                names.append(name)
                #print(names)
                ages.append(age)
                #print(ages)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    return faceSamples, Ids

faces, Ids = getImagesAndLabels('../dataset')
recognizer.train(faces, np.array(Ids))
recognizer.save("../models/trained_lbph_face_recognizer_model.yml")
