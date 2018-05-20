import cv2
import dlib

#image = cv2.imread("photos/group.0.jpg")
#image = cv2.imread("photos/group.1.jpg")
#image = cv2.imread("photos/group.2.jpg")
#image = cv2.imread("photos/group.3.jpg")
#image = cv2.imread("photos/group.4.jpg")
#image = cv2.imread("photos/group.5.jpg")
#image = cv2.imread("photos/group.6.jpg")
image = cv2.imread("photos/group.7.jpg")

detectorHog = dlib.get_frontal_face_detector()
facesDetectedHog, score, idx = detectorHog.run(image, 2)

detectorCNN = dlib.cnn_face_detection_model_v1("assets/mmod_human_face_detector.dat")
facesDetectedCNN = detectorCNN(image, 2)

for i, d in enumerate(facesDetectedHog):
    print(score[i])
print("")
for face in facesDetectedCNN:
    print(face.confidence)

