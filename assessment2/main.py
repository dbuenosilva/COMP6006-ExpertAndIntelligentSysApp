# -*- coding: utf-8 -*-
"""
File: main.py
Project: Object Recognition System
Date: 29/08/2021
Author: Diego Bueno da Silva
e-mail: d.bueno.da.silva.10@student.scu.edu.au
ID: 23567850

Dataset source: 
Yuval Netzer, Tao Wang, 
Adam Coates, Alessandro Bissacco,
Bo Wu, Andrew Y. Ng 
Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.
http://ufldl.stanford.edu/housenumbers/

Issue with MacOS:
    https://github.com/mluerig/phenopype/issues/5
    https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv

"""

# baseline cnn model for the mnist problem
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

import numpy as np
import os
import cv2
import sys
import pathlib

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

pathImages = path + "dataset/test/colour/"
pathGrayImages = path + "dataset/test/gray/"

IMG_WIDTH=32
IMG_HEIGHT=32

def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for file in os.listdir(img_folder):
       
       if len(file) > 4 and ( file[len(file)-3:len(file)] == "png" or file[len(file)-3:len(file)] == "jpg"):
           
            image= cv2.imread( img_folder + file, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)        
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(file)

            
    return img_data_array, class_name

# extract the image array and class name
#img_data, class_name =create_dataset(pathImages)



# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()


"""
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# normalize pixel values
trainX = trainX.astype('float32') / 255
testX = testX.astype('float32') / 255
# one hot encode target values


trainY = to_categorical(trainY)
testY = to_categorical(testY)

# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model
model.fit(trainX, trainY, epochs=5, batch_size=128)
# evaluate model
_, acc = model.evaluate(testX, testY, verbose=0)
print(acc)

""" 