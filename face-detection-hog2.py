import cv2
import dlib

subdetector = ["Looking Forward", "Left view", "Right view",
        "The front turning the left", "The front turning right"]

image = cv2.imread("photos/group.0.jpg")
detector = dlib.get_frontal_face_detector()

faceDetected, score, idx = detector.run(image)

print(faceDetected)
print(score)
print(idx)

for i, face in enumerate(faceDetected):
    print(i)
    print(face)
    print("Detection: {}, score: {:.4f}, Sub-detector: {}".format(i, score[i], subdetector[int(idx[i])]))
    l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)

cv2.imshow("Detector HOG", image)
cv2.waitKey(0)
cv2.destroyAllWindows()