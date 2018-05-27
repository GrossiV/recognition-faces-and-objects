import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPoints = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
recognitionFacial = dlib.face_recognition_model_v1("assets/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("assets/indices_rn.pickle")
descritorsFaciais = np.load("assets/descritors_rn.npy")
limiar = 0.5

for arquive in glob.glob(os.path.join("photos", "*.jpg")):
    image = cv2.imread(arquive)
    facesDetected = detectorFace(image, 2)
    for face in facesDetected:
        l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pointsFacial = detectorPoints(image, face)
        descritorFacial = recognitionFacial.compute_face_descriptor(image, pointsFacial)
        listDescritorFacial = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.asarray(listDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        distances = np.linalg.norm(npArrayDescritorFacial - descritorsFaciais, axis=1)
        print("Distâncias: {}".format(distances))
        minimum = np.argmin(distances)
        print(minimum)
        distanceMinimum = distances[minimum]
        print(distanceMinimum)

        if distanceMinimum <= limiar:
            name = os.path.split(indices[minimum])[1].split(".")[0]
        else:
            name = ' '

        cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
        text = "{} {:.4f}".format(name, distanceMinimum)
        cv2.putText(image, text, (r, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

    cv2.imshow("Detector hog", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()