import cv2
import numpy as np

img = np.zeros((512, 512))
print(img.shape)
img = np.zeros((512, 512, 3), np.uint8)
img[:] = 0, 0, 0
cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
cv2.rectangle(img, (0, 0), (300, 300), (0, 200, 0), 2)
cv2.circle(img, (450, 50), 30, (250, 0, 0), 3)
cv2.circle(img, (450, 50), 20, (0, 250, 0), 3)
cv2.circle(img, (450, 50), 10, (0, 0, 250), 3)
cv2.putText(img, "cad", (420, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (250, 0, 0), 3)
cv2.imshow("Window", img)
cv2.waitKey(0)
