import cv2
import numpy as np
import os
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random


class FaceRecognition:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

    def faceExtractor(self, img):
        faces = self.face_classifier.detectMultiScale(img, 1.3, 5)
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            x -= 10
            y -= 10
            cropped_face = img[y:y + h + 50, x:x + w + 50]
            return cropped_face

    def capture(self, name):
        if name not in os.listdir("./Datasets/Train/") and name not in os.listdir("./Datasets/Test/"):
            os.makedirs('./Datasets/Train/' + name)
            os.makedirs('./Datasets/Test/' + name)
        cap = cv2.VideoCapture(0)
        count = 0
        count1 = 0
        str1 = './Datasets/Train/'
        while True:
            ret, frame = cap.read()
            if self.faceExtractor(frame) is not None:
                if count1 < 150:
                    count1 += 1
                else:
                    str1 = './Datasets/Test/'
                    count1 = 0
                count += 1
                face = cv2.resize(self.faceExtractor(frame), (224, 224))
                file_name_path = str(str1) + str(name) + '/' + str(count1) + '.jpg'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print("Face not found")
                pass
            if cv2.waitKey(1) == 13 or count == 200:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Collecting Samples Complete")

    def train(self):
        IMAGE_SIZE = [224, 224]

        train_path = 'Datasets/Train'
        valid_path = 'Datasets/Test'

        # add preprocessing layer to the front of VGG
        vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

        # don't train existing weights
        for layer in vgg.layers[:-1]:
            layer.trainable = False

            # useful for getting number of classes
        folders = glob('Datasets/Train/*')

        # our layers - you can add more if you want
        vgg.layers.add(Dense(len(folders), activation='softmax'))
        # x = Dense(1000, activation='relu')(x)
        prediction = vgg(x)

        # create a model object
        model = Model(inputs=vgg.input, outputs=prediction)
        # model = load_model('facenet_keras.h5')
        # view the structure of the model
        model.summary()

        # tell the model what cost and optimization method to use
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        from keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')

        test_set = test_datagen.flow_from_directory('Datasets/Test',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')

        '''r=model.fit_generator(training_set,
                                 samples_per_epoch = 8000,
                                 nb_epoch = 5,
                                 validation_data = test_set,
                                 nb_val_samples = 2000)'''

        # fit the model
        r = model.fit_generator(
            training_set,
            validation_data=test_set,
            epochs=5,
            steps_per_epoch=len(training_set),
            validation_steps=len(test_set)
        )
        print(r.history)
        # loss

        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='test loss')
        plt.legend()
        plt.show()
        plt.savefig('LossVal_loss')

        plt.plot(r.history['accuracy'], label='train acc')
        plt.plot(r.history['val_accuracy'], label='test acc')
        plt.legend()
        plt.show()
        plt.savefig('AccVal_acc')

        model.save('facefeatures_new_model.h5')

    def recog(self):
        model = load_model("facefeatures_new_model.h5")
        video_capture = cv2.VideoCapture(0)
        while True:
            _, frame = video_capture.read()
            # canvas = detect(gray, frame)
            # image, face =face_detector(frame)

            face = self.faceExtractor(frame)
            if type(face) is np.ndarray:
                face = cv2.resize(face, (224, 224))
                im = Image.fromarray(face, 'RGB')
                # Resizing into 128x128 because we trained the model with this image size.
                img_array = np.array(im)
                # Our keras model used a 4D tensor, (images x height x width x channel)
                # So changing dimension 128x128x3 into 1x128x128x3
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                print(pred)

                name = "None matching"

                if (pred[0][0] > 0.5):
                    name = 'Nichu'
                elif (pred[0][1] > 0.5):
                    name = 'Panneer Sir'
                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
