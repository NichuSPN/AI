import cv2
import numpy as np

width, height = [500, 500]
img = cv2.imread("C:/Users/HP/Desktop/trudea.jpg")
pts1 = np.float32([[0, 117], [0, 177], [302, 117], [302, 177]])
pts2 = np.float32([[0, 0], [0, height], [width, 0], [height, width]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgout = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("warping", imgout)
cv2.waitKey(0)
