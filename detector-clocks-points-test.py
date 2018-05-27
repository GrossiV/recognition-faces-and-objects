import dlib
import cv2
import glob
import os

detectorClock = dlib.simple_object_detector("assets/detector_relogios.svm")
clockPointsDetector = dlib.shape_predictor("assets/detector_relogios_pontos.dat")

print(dlib.test_shape_predictor("assets/teste_relogios_pontos.xml", "assets/detector_relogios_pontos.dat"))

def imprimirPoints(image, points):
    for p in points.parts():
        cv2.circle(image, (p.x, p.y), 2, (0, 255, 0))

for arquive in glob.glob(os.path.join("relogios_teste", "*.jpg")):
    image = cv2.imread(arquive)
    objectDetected = detectorClock(image, 2)
    for clock in objectDetected:
        l, t, r, b = (int(clock.left()), int(clock.top()), int(clock.right()), int(clock.bottom()))
        points = clockPointsDetector(image, clock)
        imprimirPoints(image, points)
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)

    cv2.imshow("Points Detector :", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()