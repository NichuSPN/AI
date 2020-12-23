import cv2

cap = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(r"E:\pythonProject\opencv\xml\haarcascade_frontalface_alt.xml")
eyecascade = cv2.CascadeClassifier(r"E:\pythonProject\opencv\xml\haarcascade_eye.xml")
while True:
    _, img = cap.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(imggray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    eyes = eyecascade.detectMultiScale(imggray, 1.1, 4)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
