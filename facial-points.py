import dlib
import cv2
import numpy as np

def imprimePontos(image, facePoints):
    for p in facePoints.parts():
        cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)

def imprimeNumeros(image, facePoints):
    for i, p in enumerate(facePoints.parts()):
        cv2.putText(image, str(i), (p.x, p.y), font, .55, (0, 0, 255), 1)

def imprimeLinhas(image, facePoints):
    p68 = [[0, 16, False], # linha do queixo
           [17, 21, False], # sombrancelha direita
           [22, 26, False], # sombancelha esquerda
           [27, 30, False], # ponte nasal
           [30, 35, True], # nariz inferior
           [36, 41, True], # olho esquerdo
           [42, 47, True], # olho direito
           [48, 59, True], # labio externo
           [60, 67, True]] # labio interno
    for k in range(0, len(p68)):
        points = []
        for i in range(p68[k][0], p68[k][1] + 1):
            point = [facePoints.part(i).x, facePoints.part(i).y]
            points.append(point)
        point = np.array(points, dtype=np.int32)
        cv2.polylines(image, [point], p68[k][2], (255, 0, 0), 2)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#image = cv2.imread("photos/training/ronald.0.1.jpg")
#image = cv2.imread("photos/group.0.jpg")
#image = cv2.imread("photos/group.1.jpg")
#image = cv2.imread("photos/group.2.jpg")
#image = cv2.imread("photos/group.3.jpg")
#image = cv2.imread("photos/group.4.jpg")
image = cv2.imread("photos/group.5.jpg")
#image = cv2.imread("fotos/grupo.6.jpg")
#image = cv2.imread("fotos/grupo.7.jpg")

detectorFace = dlib.get_frontal_face_detector()
detectorPoints = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
facesDetected = detectorFace(image, 2)
for face in facesDetected:
    points = detectorPoints(image, face)
    print(points.parts())
    print(len(points.parts()))
    #imprimePontos(image, points)
    #imprimeNumeros(image, points)
    imprimeLinhas(image, points)

cv2.imshow("Pontos faciais", image)
cv2.waitKey(0)
cv2.destroyAllWindows()