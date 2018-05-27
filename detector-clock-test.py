import os
import dlib
import cv2
import glob

print(dlib.test_simple_object_detector("assets/teste_relogios.xml", "assets/detector_relogios.svm"))

detectorClock = dlib.simple_object_detector("assets/detector_relogios.svm")
for image in glob.glob(os.path.join("relogios_teste", "*.jpg")):
    img = cv2.imread(image)
    objectDetected = detectorClock(img, 2)
    for d in objectDetected:
        l, t, r, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
    cv2.imshow("Detector of Clocks: ", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()