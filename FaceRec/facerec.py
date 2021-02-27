import cv2
import os
import numpy as np


class Recognizer:

    def faceDetection(self, test_img):
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        face_haar_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
        faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)
        return faces, gray_img

    def labels_for_train(self,directory):
        faces = []
        faceID = []

        for path,subdirnames,filenames in os.walk(directory):
            for filename in filenames:
                if filename.startswith("."):
                    print("Skipped system file")
                    continue
                id = os.path.basename(path)