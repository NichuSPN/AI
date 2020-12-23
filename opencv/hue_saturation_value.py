import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_purple = np.array([110, 100, 100])
upper_purple = np.array([130, 255, 255])

while True:
    success, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            print(area)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("hsv", img)
    cv2.imshow("mask", mask)
    cv2.waitKey(1)
