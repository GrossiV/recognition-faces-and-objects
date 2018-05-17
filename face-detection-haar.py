import cv2

image = cv2.imread("photos/group.0.jpg")

classificator = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faceDetected = classificator.detectMultiScale(imageGray, scaleFactor=1.2, minSize=(50,50))
print(faceDetected)
print("Faces Detected: ", len(faceDetected))
for (x, y, l, a) in faceDetected:
    cv2.rectangle(image,(x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Detector Haar", image)
cv2.waitKey(0)
cv2.destroyAllWindows()