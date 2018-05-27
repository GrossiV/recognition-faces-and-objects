import dlib
import cv2
import glob
import os

def imprimirPoints(image, points):
    for p in points.parts():
        cv2.circle(image, (p.x, p.y), 2, (0, 255, 0))

for arquive in glob.glob(os.path.join("relogios_teste2", "*.jpg")):
    image = cv2.imread(arquive)

    cv2.imshow("Points Detector :", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()