import dlib
import cv2
import numpy as np

def imprimePoints(image, facialPoints):
    for p in facialPoints.parts():
        cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)
detectorFace = dlib.get_frontal_face_detector()
detectorPoints = dlib.shape_predictor("assets/shape_predictor_5_face_landmarks.dat")
image = cv2.imread("photos/training/ronald.0.1.jpg")
imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
facesDetected = detectorFace(imageRGB, 0)
pointsFaces = dlib.full_object_detections()
for face in facesDetected:
    points = detectorPoints(imageRGB, face)
    pointsFaces.append(points)
    imprimePoints(image, points)

images = dlib.get_face_chips(imageRGB, pointsFaces)
for img in images:
    imageBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Imagem original: ", image)
    cv2.waitKey(0)
    cv2.imshow("Imagem alinhada: ", imageBGR)
    cv2.waitKey(0)

#cv2.imshow("5 Pontos: ", image)
#cv2.waitKey(0)
cv2.destroyAllWindows()