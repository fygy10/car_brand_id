import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers, models
from keras.optimizers import Adam
from keras.layers import Dense, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator


#CLASS TO TRAIN MODEL AND PREDICT NEW INSTANCES
class CarIDModel:

    #set parameters for model training
    def __init__(self, learning_rate=0.0002, epochs=30, classes=8):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.classes = classes
        self.model = self.build_model()


    #use MobileNEt as the pre-trained model and fine-tune it to my databased
    def build_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3),
            include_top=False,
            weights='imagenet'
        )
        for layer in base_model.layers[0:12]:
            layer.trainable = True

        model = models.Sequential([
            base_model,
            layers.Conv2D(32, (5,5), activation='relu', padding="same"),
            layers.MaxPool2D(2,2),
            layers.Conv2D(32, (5,5), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(640, activation='relu'),
            layers.LeakyReLU(alpha = 0.01),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.classes, activation='softmax')
        ])
        return model


    #complile model
    def compile_model(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']

        )


    #loading the data iand normalize it
    #used for loops since subfolder names were the labels and the images inside corresponded to them
    def load_data(self, folder_path):
        images = []
        labels = []
        class_labels = os.listdir(folder_path)

        for class_label in class_labels:
            class_path = os.path.join(folder_path, class_label)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    if img_path.endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (96, 96))  #the size commonly used with MobileNet
                        images.append(img)
                        labels.append(class_labels.index(class_label))

        images = np.array(images)
        labels = to_categorical(labels, num_classes=self.classes)

        return images, labels



    #training the data and performing train-test split
    def train(self, train_path):
        images, labels = self.load_data(train_path)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test))



    #this function evaluates image each passed into flask and categorizes it
    def evaluate(self, imgdata):
        model = self.model

        img = cv2.resize(imgdata, (96, 96))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        prediction = model.predict(img)
        predicited_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicited_class_index]

        return predicted_class_name




run_model = CarIDModel()
class_names = ['Hyundai', 'Lexus',  'Mazda', 'Volswagen', 'Mercedes', 'Opel', 'Skoda', 'Toyota']
#train_path = ''
#run_model.train(train_path)
# print(run_model.evaluate(imgfile))

