import cv2
import dlib

image = cv2.imread("photos/group.0.jpg")
detector = dlib.get_frontal_face_detector()
faceDetected = detector(image, 1)

print(faceDetected)
print("Faces Detected: ", len(faceDetected))

for face in faceDetected:
    #print(face)
    #print(face.left())
    #print(face.top())
    #print(face.right())
    #print(face.bottom())
    l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)

cv2.imshow("Detector HOG", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
