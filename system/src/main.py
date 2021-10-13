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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.model_selection import train_test_split
from datetime import datetime

import numpy as np
import pandas as pd
import os
import cv2
import sys
import pathlib
import csv
import random

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

def loadDataset(pathImages, imageWidth, imageHeight, orderby = False, verbose = False ):
   
    X = []

    print("\nReading images...")
    
    listOfFiles = os.listdir(pathImages)

    if orderby:
        listOfFiles.sort(key=getFileOriginalName) # Order the list according to files name to match with label file

    for file in listOfFiles: 

       if len(file) > 4 and ( file[len(file)-3:len(file)] == "png" or file[len(file)-3:len(file)] == "jpg"):
            if verbose:
                print(file)
            image = cv2.imread( pathImages + file, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (imageWidth, imageHeight),interpolation = cv2.INTER_AREA)        
            image = np.array(image)
            image = image.astype('uint8')
            X.append(image)
    
    print("\nDone! Loaded all images from " + pathImages + " to X \n")
    
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
        print("\nLoaded labels from " + pathFile + "\n")   
    return(Y.astype('uint8'))


##########################################################################
# Function: getModel
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 08/09/2021
# Description: Get acuracy of image recognition using ANN.
# 
# Parameters: trainX,trainY,testX,testY,modelFileName
# 
# Return:     acc - Accuracy achieved, loss
#
##########################################################################

def getModel(trainX,trainY,testX,testY,modelFileName = "myModel.h5"):

    modelFileName = path + modelFileName

    if len(trainX.shape) >= 4 : # it has thrid dimension with channels (RGB)
        instances, width, height, channels = trainX.shape
    else: #Gray scale, reshape dataset to have a single channel
        instances, width, height, channels = (trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
        instancesTest = testX.shape[0]        
        trainX = trainX.reshape((instances,width, height, channels))
        testX = testX.reshape((instancesTest,width, height, channels))    

    print("\nLoading to the model: instances = " + str(instances) + ", width = " + str(width) + "  height = " + str(height) + ", channels = " + str(channels) + " ")
    
    # normalize pixel values
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255
    
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    print("\nTraining the model with FFNN...")
       
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

    # Stopping early according to myMinDelta to avoid overfitting. Trained model saved at myModelFile
    myMinDelta  = 0.05 # minimum improvement rate for do not early stop
    myPatience  = 2   # how many epochs run with improvement lower than myMinDelta     
    myCallbacks = [EarlyStopping(monitor='accuracy', min_delta=myMinDelta , patience=myPatience, mode='auto'),
                 ModelCheckpoint(filepath=modelFileName, monitor='accuracy', save_best_only=True, verbose=1)]
     
    ## Training the model according to the labels and chose hyperparameters
    model.fit(trainX, trainY, epochs=50, batch_size=128, verbose=1, callbacks = myCallbacks)

    # evaluate model
    loss, accuracy = model.evaluate(testX, testY, verbose=1)

    print("\nAchivied accuracy: " + str(accuracy) + ("      :)" if accuracy > 0.75 else "       :(") )
    print("Loss: " + str(loss))    

    # Saving the designed NN
    if modelFileName:
        model.save(modelFileName)

    return(loss, accuracy)


##########################################################################
# 
# main program
# 
# Training the model to different image features and plotting the results
#
##########################################################################

results  = []
loss     = 0
accuracy = 0

"""
### Training original colour images resizing 32 x 32
"""
try:    
    trainX = loadDataset(path + "../dataset/train-preprocessed/colour-original/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/colour-original/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "colour-original.h5")
    results.append( [ "original colour images", loss, accuracy ] )
except:
    print("Error trying to train colour-original!\n")
    results.append( [ "original colour images", 0, 0 ] )

"""
### Training colour images applied Gaussian blur feature and resizing 32 x 32
"""
try:   
    trainX = loadDataset(path + "../dataset/train-preprocessed/colour-plus-gaussian-blur/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/colour-plus-gaussian-blur/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "colour-plus-gaussian-blur.h5")
    results.append( [ "colour images applied Gaussian blur feature", loss, accuracy ] )
except:
    print("Error trying to train colour images applied Gaussian blur feature!\n")
    results.append( [ "colour images applied Gaussian blur feature", 0, 0 ] )

"""
### Training gray scale imageges resizing 32 x 32
"""
try:   
    trainX = loadDataset(path + "../dataset/train-preprocessed/gray-scale-only/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/gray-scale-only/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "gray-scale-only.h5")
    results.append( [ "gray scale images", loss, accuracy ] )
except:
    print("Error trying to train gray scale images!\n")
    results.append( [ "gray scale images", 0, 0 ] )

"""
### Training gray scale images applying histograms equalisation and otsu thresholding features
"""
try:   
    trainX = loadDataset(path + "../dataset/train-preprocessed/gray-scale-plus-histograms-equalisation-and-otsu-thresholding/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/gray-scale-plus-histograms-equalisation-and-otsu-thresholding/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "gray-scale-plus-histograms-equalisation-and-otsu-thresholding.h5")
    results.append( [ "gray scale images applying histograms equalisation and otsu thresholding features",  loss, accuracy ] )
except:
    print("Error trying to train gray scale images applying histograms equalisation and otsu thresholding features!\n")
    results.append( [ "gray scale images applying histograms equalisation and otsu thresholding features",  0, 0 ] )

"""
### Training gray scale images applying Laplacian operator feature
"""
try:   
    trainX = loadDataset(path + "../dataset/train-preprocessed/gray-scale-plus-laplacian-operator/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/gray-scale-plus-laplacian-operator/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "gray-scale-plus-laplacian-operator.h5")
    results.append( [ "gray scale images applying Laplacian operator feature",  loss, accuracy ] )
except:
    print("Error trying to train gray scale images applying Laplacian operator feature!\n")
    results.append( [ "gray scale images applying Laplacian operator feature",  0, 0 ] )

"""
### Training gray scale images applying only otsu thresholding feature
"""
try:   
    trainX = loadDataset(path + "../dataset/train-preprocessed/gray-scale-plus-otsu-thresholding/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/gray-scale-plus-otsu-thresholding/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "gray-scale-plus-otsu-thresholding.h5")
    results.append( [ "gray scale images applying only otsu thresholding feature",  loss, accuracy ] )
except:
    print("Error trying to train gray scale images applying only otsu thresholding feature!\n")
    results.append( [ "gray scale images applying only otsu thresholding feature",  0, 0 ] )

"""
### Training gray scale images applying only otsu thresholding feature and inverting the background to white
"""
try:   
    trainX = loadDataset(path + "../dataset/train-preprocessed/gray-scale-plus-otsu-thresholding-inverting-background/", 32, 32, True, True )
    trainY = loadLabels(path + "../dataset/train/labels.csv")

    testX = loadDataset(path + "../dataset/test-preprocessed/gray-scale-plus-otsu-thresholding-inverting-background/", 32, 32, True, True )
    testY = loadLabels(path + "../dataset/test/labels.csv")

    loss, accuracy = getModel(trainX,trainY,testX,testY, "gray-scale-plus-otsu-thresholding-inverting-background.h5")
    results.append( [ "gray scale images applying only otsu thresholding feature and inverting the background to white",  loss, accuracy ] )
except:
    print("Error trying to train gray scale images applying only otsu thresholding feature and inverting the background to white!\n")
    results.append( [ "gray scale images applying only otsu thresholding feature and inverting the background to white",  0, 0 ] )

"""
### Training crops colour images from Yolo5 resizing 32 x 32
"""
try:

    X0 = loadDataset(path + "yolov5/runs/detect/exp6/crops/0/", 32, 32, False, True )
    Y0 = loadLabels(path + "yolov5/runs/detect/exp6/crops/0/labels.csv")

    X1 = loadDataset(path + "yolov5/runs/detect/exp6/crops/1/", 32, 32, False, True )
    Y1 = loadLabels(path + "yolov5/runs/detect/exp6/crops/1/labels.csv")

    X2 = loadDataset(path + "yolov5/runs/detect/exp6/crops/2/", 32, 32, False, True )
    Y2 = loadLabels(path + "yolov5/runs/detect/exp6/crops/2/labels.csv")

    X3 = loadDataset(path + "yolov5/runs/detect/exp6/crops/3/", 32, 32, False, True )
    Y3 = loadLabels(path + "yolov5/runs/detect/exp6/crops/3/labels.csv")

    X4 = loadDataset(path + "yolov5/runs/detect/exp6/crops/4/", 32, 32, False, True )
    Y4 = loadLabels(path + "yolov5/runs/detect/exp6/crops/4/labels.csv")

    X5 = loadDataset(path + "yolov5/runs/detect/exp6/crops/5/", 32, 32, False, True )
    Y5 = loadLabels(path + "yolov5/runs/detect/exp6/crops/5/labels.csv")

    X6 = loadDataset(path + "yolov5/runs/detect/exp6/crops/6/", 32, 32, False, True )
    Y6 = loadLabels(path + "yolov5/runs/detect/exp6/crops/6/labels.csv")

    X7 = loadDataset(path + "yolov5/runs/detect/exp6/crops/7/", 32, 32, False, True )
    Y7 = loadLabels(path + "yolov5/runs/detect/exp6/crops/7/labels.csv")

    X8 = loadDataset(path + "yolov5/runs/detect/exp6/crops/8/", 32, 32, False, True )
    Y8 = loadLabels(path + "yolov5/runs/detect/exp6/crops/8/labels.csv")

    X9 = loadDataset(path + "yolov5/runs/detect/exp6/crops/9/", 32, 32, False, True )
    Y9 = loadLabels(path + "yolov5/runs/detect/exp6/crops/9/labels.csv")

    # Concatenating all crops into unique X and y arrays
    X = np.concatenate((X0, X1, X2, X3, X4, X5, X6, X7, X8, X9), axis=0)
    y = np.concatenate((Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9), axis=0)

    # Spliting into train and test
    [trainX, testX, trainY, testY] = train_test_split(X, y, test_size = 0.2, random_state= 0, shuffle=True )

    loss, accuracy = getModel(trainX,trainY,testX,testY, "yolo.h5")
    results.append( [ "crops colour images from Yolo5", loss, accuracy ] )
except:
    print("Error trying to train crops colour images from Yolo5!\n")
    results.append( [ "crops colour images from Yolo5", 0, 0 ] )


if len(results):
    # Create the pandas DataFrame with all results
    df = pd.DataFrame(results, columns = ['Image Feature and Preprocessing approch', 'Loss', 'Accuracy'] )

    # datetime object containing current date and time
    now = datetime.now()
    fileName = now.strftime("%Y-%m-%d_%H%M") # YYYY-MM-DD_HH-MM

    # Exporting results to a CSV file
    df.to_csv('system/results/' + fileName + '.csv',index=False)
    print("\nResults saved on " + path + "system/results/" + fileName + ".csv")    
    # Plotting the results and comparing

print("\nDone!")