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
import csv

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)


##########################################################################
# Function: loadDataset 
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 07/09/2021
# Description: Load images dataset to a numpy array
#              
# 
# Parameters: pathImages - a string with the path where images are in.
#             imageWidth - the imageWidth to resize the image.
#             imageHeight - the height to resize the image.
# 
# Return:     X - a numpy array with a colection of each image in array
#
##########################################################################

def getFileOriginalName(elem):
    if len(elem) > 4 and ( elem[len(elem)-3:len(elem)] == "png" or elem[len(elem)-3:len(elem)] == "jpg"):
        return ( int(elem[0:len(elem)-6]) )
    return 0

def loadDataset(pathImages, imageWidth, imageHeight ):
   
    X = np.array([])

    print("Reading images...")
    
    listOfFiles = os.listdir(pathImages)

    listOfFiles.sort(key=getFileOriginalName) # Order the list according to files name to match with label file

    """ 
    for file in listOfFiles: 
       print(file)
       if len(file) > 4 and ( file[len(file)-3:len(file)] == "png" or file[len(file)-3:len(file)] == "jpg"):
           
            image = cv2.imread( pathImages + file, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (imageWidth, imageHeight),interpolation = cv2.INTER_AREA)        
            image = np.array(image)
            image = image.astype('float32')
            image /= 255 
            X = np.append(X,image)
    """
    #print("Done! Loaded all images from " + pathImages + " to X")
    
    return(listOfFiles)


##########################################################################
# Function: loadLabels
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 08/09/2021
# Description: Load labes from a CSV file
# 
# Parameters: pathImages - a string with the path where images are in.
# 
# Return:     Y - a numpy array with a colection labels
#
##########################################################################

def loadLabels(pathFile):
    
    with open(pathFile, newline='') as f:
        reader = csv.reader(f)
        Y = np.array( list(reader) )

    return(Y)


"""
Training original colour images resizing 32 x 32
"""
#trainX = loadDataset(path + "dataset/train/colour-original/", 32, 32)
#trainY = loadLabels("dataset/train/labels.csv")

mytestX = loadDataset(path + "dataset/test/colour-original/", 32, 32)
mytestY = loadLabels(path + "dataset/test/labels.csv")




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