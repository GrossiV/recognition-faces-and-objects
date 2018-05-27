import dlib
import glob
import cv2
import os

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5

#dlib.train_simple_object_detector("assets/treinamento_delirium.xml", "assets/detector_delirium.svm", options)

detector = dlib.simple_object_detector("assets/detector_delirium.svm")
for image in glob.glob(os.path.join("delirium", "*.jpg")):
    img = cv2.imread(image)
    objectDetected = detector(img, 2)
    for d in objectDetected:
        l, t, r, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
    cv2.imshow("Detector of Logos: ", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()