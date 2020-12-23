import cv2
import numpy as np

img = cv2.imread("C:/Users/HP/Desktop/trudea.jpg")
imgresize = cv2.resize(img, (500, 300))
imgcrop = img[0:150, 150:500]
cv2.imshow("window", imgresize)
cv2.imshow("crop", imgcrop)
cv2.waitKey(0)
