"""
Original file is located at
    https://colab.research.google.com/drive/1Q_trcjQNWiweeegcWfth8AvFfdGiOCJe
"""

!git clone https://github.com/adhammm/test.git

cd test

import tensorflow as tf
import os
import numpy as np
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import random
from keras.utils.np_utils import to_categorical

def read_data():
    # Load training data set from CSV file
    #training_data_df = pd.read_csv("data_file.csv", sep=',')

    # Pull out columns for X (data to train with) and Y (value to predict)
    with open("data_file.csv", 'r') as read_obj:
        csv_reader = csv.DictReader(read_obj, delimiter=',')

        X_training = []
        Y_training = []
        for line in csv_reader:
            X_training.append(line['Images'])

            if ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [0, 1, 1] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [0, 0, 1]):
                Y_training.append(0)
            elif ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 0, 1] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 1, 1]):
                Y_training.append(1)
            elif ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 1, 0] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 0, 0]):
                Y_training.append(2)

        #Y_training.append([int(line['Right']), int(line['Forword']), int(line['Left'])])
    X_training = np.array(X_training)
    Y_training = np.array(Y_training)



    return (X_training, Y_training)

data_train = read_data()

import matplotlib.pyplot as plt
X_train =[]
print(len(data_train[0]))
for i in range(len(data_train[0])):
  x=cv2.imread(data_train[0][i])
  X_train.append(x)

Y_train = data_train[1]
print(Y_train.shape)
X_train =np.asarray(X_train)
X_train = X_train.reshape(998,288, 352, 3)

def preprocess(img):
    img = img/255
    return img
  
X_train = np.array(list(map(preprocess, X_train)))

"""
datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.)

datagen.fit(X_train)
# for X_batch, y_batch in

batches = datagen.flow(X_train, Y_train, batch_size = 15)
X_batch, y_batch = next(batches)
"""

y_train = to_categorical(Y_train, 3)

def modified_model():
  model = Sequential()
  model.add(Conv2D(32, (5, 5), input_shape=(288,352, 3), activation='relu',strides=(2,2)))
  model.add(Conv2D(36, (3, 3), activation='relu'))
  model.add(Conv2D(48, (3, 3), activation='relu',strides=(2,2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu',strides=(2,2))) 
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(Conv2D(256, (3, 3), activation='relu'))

  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(512, activation='relu'))

  model.add(Dropout(0.5))
  
  model.add(Dense(3, activation='softmax'))
  
  model.compile(Adam(lr = 0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = modified_model()
print(model.summary())

history = model.fit(X_train, y_train,batch_size=50,epochs=20,shuffle = 1,verbose=1)