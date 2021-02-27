import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import csvwriter as cw
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



class FaceRec:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.namelist = cw.read("names.csv").to_numpy().flatten()

    def face_cropped(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    def generate_dataset(self, name):
        cap = cv2.VideoCapture(0)
        img_id = 0
        while True:
            _, frame = cap.read()
            if self.face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(self.face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = "data/" + name + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped Face", face)
                if cv2.waitKey(1) == 13 or int(img_id) == 1000:
                    break
        cap.release()
        cv2.destroyAllWindows()
        self.namelist.append(name)
        cw.write("names.csv", [name])
        print("Collecting datasets completed")

    def my_label(self, image_name):
        arr = np.zeros(len(self.namelist))
        name = image_name.split('.')[-3]
        arr[np.where(self.namelist == name)] = 1
        return arr

    def my_data(self):
        data = []
        for img in tqdm(os.listdir("data")):
            path = os.path.join("data", img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (50, 50))
            data.append([np.array(img_data), self.my_label(img)])
        shuffle(data)
        return data

    def split_data(self):
        data = self.my_data()
        train = data[:len(self.namelist) * 800]
        test = data[len(self.namelist) * 800:]
        X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
        X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
        Y_train = np.array([i[1] for i in train])
        Y_test = np.array([i[1] for i in test])

        return X_train, X_test, Y_train, Y_test

    def model_face_rec(self):
        tf.compat.v1.reset_default_graph()
        convnet = input_data(shape=[50, 50, 1])
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)  # prevent overfitting
        convnet = fully_connected(convnet, len(self.namelist), activation='softmax')
        convnet = regression(convnet,
                             optimizer='adam',
                             learning_rate=0.001,
                             loss='categorical_crossentropy')
        model = tflearn.DNN(convnet, tensorboard_verbose=3)
        X_train, X_test, Y_train, Y_test = self.split_data()
        r = model.fit(X_train, Y_train,
                      n_epoch=12,
                      validation_set=(X_test, Y_test),
                      show_metric=True,
                      run_id="Face Recognition")
