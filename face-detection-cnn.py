import cv2
import dlib

image = cv2.imread("photos/group.0.jpg")
detector = dlib.cnn_face_detection_model_v1("assets/mmod_human_face_detector.dat")
facesDetected = detector(image, 2)
print(facesDetected)
print("Faces detected: ", len(facesDetected))
for face in facesDetected:
    l, t, r, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
print(c)
cv2.rectangle(image, (l, t), (r, b), (255, 255, 0), 2)

cv2.imshow("Detector CNN", image)
cv2.waitKey(0)
cv2.destroyAllWindows()