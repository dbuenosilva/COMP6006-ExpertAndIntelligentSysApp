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

#import tensorflow as tf # using Tensorflow 2.4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical

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
                     # original file name  # sequencial per digit (ROI)
        return ( int(elem[0:len(elem)-6] + elem[len(elem)-5:len(elem)-4]) )
    return 0

def loadDataset(pathImages, imageWidth, imageHeight ):
   
    X = []

    print("Reading images...")
    
    listOfFiles = os.listdir(pathImages)

    listOfFiles.sort(key=getFileOriginalName) # Order the list according to files name to match with label file

    for file in listOfFiles: 
       print(file)
       if len(file) > 4 and ( file[len(file)-3:len(file)] == "png" or file[len(file)-3:len(file)] == "jpg"):
           
            image = cv2.imread( pathImages + file, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (imageWidth, imageHeight),interpolation = cv2.INTER_AREA)        
            image = np.array(image)
            image = image.astype('uint8')
            X.append(image)
    
    print("Done! Loaded all images from " + pathImages + " to X")
    
    return( np.array(X, np.uint8) )


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
    
    Y = np.array([])
    
    with open(pathFile, newline='') as f:
        reader = csv.reader(f)
        #Y = np.array( list(reader) )
        rows =  list(reader) 
        
        # converting the list to numpy array
        for label in rows :
            Y = np.append(Y,label[0])
        
    return(Y.astype('uint8'))


##########################################################################
# Function: getAcuracy
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 08/09/2021
# Description: Get acuracy of image recognition using ANN.
# 
# Parameters: trainX,trainY,testX,testY
# 
# Return:     acc - Accuracy achieved 
#
##########################################################################

def getAcuracy(trainX,trainY,testX,testY):

    if len(trainX.shape) >= 4 : # it has thrind dimension with channels
        instances, width, height, channels = trainX.shape
    else:   
        # reshape dataset to have a single channel
        channels = 0 #gray scale
        instances, width, height = trainX.shape    
    
    print("Loading to the model: instances = " + str(instances) + ", width = " + str(width) + "  height = " + str(height) + ", channels = " + str(channels) + " ")
    
    # normalize pixel values
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255
    
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    print("Training the model with ANN...")
    
    
    if channels > 0:
        shape = (width, height, channels) 
    else:
        shape = (width, height) 
    # define model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(shape)))
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

    print("Achivied accuracy: " + str(acc))

    return(acc)




results = []

"""
Training original colour images resizing 32 x 32
"""
trainX = loadDataset(path + "dataset/train/colour-original/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "dataset/test/colour-original/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "original colour images", str(getAcuracy(trainX,trainY,testX,testY)) ] )


"""
Training colour images applied Gaussian blur feature and resizing 32 x 32
"""
trainX = loadDataset(path + "dataset/train/colour-plus-gaussian-blur/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "dataset/test/colour-plus-gaussian-blur/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "colour images applied Gaussian blur feature", str(getAcuracy(trainX,trainY,testX,testY)) ] )


"""
Training gray scale imageges resizing 32 x 32
"""
trainX = loadDataset(path + "dataset/train/gray-scale-only/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "dataset/test/gray-scale-only/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "gray scale images", str(getAcuracy(trainX,trainY,testX,testY)) ] )


"""
Training gray scale images applying histograms equalisation and otsu thresholding features
"""
trainX = loadDataset(path + "dataset/train/gray-scale-plus-histograms-equalisation-and-otsu-thresholding/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "dataset/test/gray-scale-plus-histograms-equalisation-and-otsu-thresholding/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "gray scale images applying histograms equalisation and otsu thresholding features", str(getAcuracy(trainX,trainY,testX,testY)) ] )


"""
Training gray scale images applying Laplacian operator feature
"""
trainX = loadDataset(path + "dataset/train/gray-scale-plus-laplacian-operator/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "dataset/test/gray-scale-plus-laplacian-operator/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "gray scale images applying Laplacian operator feature", str(getAcuracy(trainX,trainY,testX,testY)) ] )


"""
Training gray scale images applying only otsu thresholding feature
"""
trainX = loadDataset(path + "dataset/train/gray-scale-plus-otsu-thresholding/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "dataset/test/ggray-scale-plus-otsu-thresholding/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "gray scale images applying only otsu thresholding feature", str(getAcuracy(trainX,trainY,testX,testY)) ] )


"""
Training gray scale images applying only otsu thresholding feature and inverting the background to white
"""
trainX = loadDataset(path + "gray-scale-plus-otsu-thresholding-inverting-background/", 32, 32)
trainY = loadLabels(path + "dataset/train/labels.csv")

testX = loadDataset(path + "gray-scale-plus-otsu-thresholding-inverting-background/", 32, 32)
testY = loadLabels(path + "dataset/test/labels.csv")

results.append( [ "gray scale images applying only otsu thresholding feature and inverting the background to white", str(getAcuracy(trainX,trainY,testX,testY)) ] )

    
print(results)




