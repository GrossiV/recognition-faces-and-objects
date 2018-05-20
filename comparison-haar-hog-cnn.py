import cv2
import dlib

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#image = cv2.imread("photos/group.0.jpg")
#image = cv2.imread("photos/group.1.jpg")
#image = cv2.imread("photos/group.2.jpg")
#image = cv2.imread("photos/group.3.jpg")
#image = cv2.imread("photos/group.4.jpg")
#image = cv2.imread("photos/group.5.jpg")
image = cv2.imread("fotos/grupo.7.jpg")

# Haar
detectorHaar = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
facesDetectedHaar = detectorHaar.detectMultiScale(imageGray, scaleFactor=1.1, minSize=(10,10))

# Hog
detectorHog = dlib.get_frontal_face_detector()
facesDetectedHog = detectorHog(image, 2)

# CNN
detectorCNN = dlib.cnn_face_detection_model_v1("assets/mmod_human_face_detector.dat")
facesDetectedCNN = detectorCNN(image, 2)

for (x, y, l, a) in facesDetectedHaar:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)
    cv2.putText(image, "Haar", (x, y - 5), font, 0.5, (0, 255, 0))

for face in facesDetectedHog:
    l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)
    cv2.putText(image, "Hog", (l, t), font, 0.5, (0, 255, 255))

for face in facesDetectedCNN:
    l, t, r, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
    cv2.rectangle(image, (l, t), (r, b), (255, 255, 0), 2)
    cv2.putText(image, "CNN", (l, t), font, 0.5, (255, 255, 0))

cv2.imshow("Comparison  between detectors", image)
cv2.waitKey(0)
cv2.destroyAllWindows()