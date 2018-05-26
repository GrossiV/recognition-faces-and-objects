import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPoints = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
recognitionFace = dlib.face_recognition_model_v1("assets/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descriptionFace = None

for arquive in glob.glob(os.path.join("photos/training","*.jpg")):
    image = cv2.imread(arquive)
    facesDetected = detectorFace(image, 1)
    numberFacesDetected = len(facesDetected)
    #print(numberFacesDetected)

    if numberFacesDetected > 1:
        print("There's more than one face in the image {}".format(arquive))
        exit()
    elif numberFacesDetected < 1:
        print("No face detected in archives {}".format(arquive))
        exit()

    cv2.imshow("Training", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

