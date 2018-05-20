import cv2
import dlib

def imprimePoints(image, facePoints):
    for p in facePoints.parts():
        cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
image = cv2.imread("photos/training/ronald.0.0.jpg")

detectorFace = dlib.get_frontal_face_detector()
detectorFacePoints = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
facesDetected = detectorFace(image, 2)

for face in facesDetected:
    points = detectorFacePoints(image, face)
    print(points.parts())
    print(len(points.parts()))
    imprimePoints(image, points)

cv2.imshow("Facial Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()