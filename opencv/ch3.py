import cv2

img = cv2.imread("C:/Users/HP/Desktop/trudea.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (7, 7), 9)
img_canny = cv2.Canny(img_gray, 100, 100)
cv2.imshow("canny", img_canny)
cv2.imshow("blur", img_blur)
cv2.imshow("gray", img_gray)
cv2.waitKey(0)
