# -*- coding: utf-8 -*-
"""
File: getlabels.py
Project: Object Recognition System
Date: 14/10/2021
Author: Diego Bueno da Silva
e-mail: d.bueno.da.silva.10@student.scu.edu.au
ID: 23567850

Dataset source: http://ufldl.stanford.edu/housenumbers/

Issue with MacOS:
    https://github.com/mluerig/phenopype/issues/5
    https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv

Required libraries:
    
    pip install mat73 # to read MatLat v7.3 file
    pip install opencv-python
    pip install opencv-contrib-python
    pip install Pillow
    
"""
import cv2
import sys
import mat73
import pathlib
import numpy as np
import csv

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)


##########################################################################
# Function: saveLabeltoCSV
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 08/09/2021
# Description: Save the labes to an SCV file
#              
# 
# Parameters: pathImages - a string with the path where the file is created.
#             l;abel - the label to add in the file
# 
# Return:     null
#
##########################################################################

def saveLabeltoCSV(pathToCsv, label = ''):

    with open(pathToCsv, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(label)
        
    return
# end of saveLabeltoCSV function


##########################################################################
# Function: ReadingAndPreProcessingImages 
# Author: Diego Bueno - d.bueno.da.silva.10@student.scu.edu.au 
# Date: 29/08/2021
# Description: Reading all the imagens from defined path, getting the ROI 
#              and labels them according to digitStruct.mat file.
# 
# Parameters: pathImages - a string with the path where images are in.
#             matLabFileDatDict - a string with the matlab file dict.
# 
# Return:     null
#
##########################################################################

def gettingLabels(pathImages, matLabFileDatDict):
    
    # mat73 lib takes long to read test_digitStruct.mat
    print( "\nReading the metadados digitStruct.mat\n" )
    print( "It may take time, be patience...\n" )
    
    data_dict = mat73.loadmat(matLabFileDatDict)    
    boxes = data_dict['digitStruct']['bbox']
    fileNames = data_dict['digitStruct']['name']
    
    #boxes[0] # 5 
    #{'height': array(30.),
    # 'label': array(5.),
    # 'left': array(43.),
    # 'top': array(7.),
    # 'width': array(19.)}
    
    #boxes[1] # 210
    #{'height': [array(23.), array(23.), array(23.)],
    # 'label': [array(2.), array(1.), array(10.)],
    # 'left': [array(99.), array(114.), array(121.)],
    # 'top': [array(5.), array(8.), array(6.)],
    # 'width': [array(14.), array(8.), array(12.)]}
    
    #	cv2.IMREAD_COLOR or 1: Read the image in colour mode.
    #	cv2.IMREAD_GRAYSCALE or 0: Read the image in grayscale mode.
    #	cv2.IMREAD_UNCHANGED or -1: Read the image with alpha channels.
       
    print( "Reading dataset " + pathImages )
    
    for i in range(0,len(fileNames)):
    
        print( "Reading file " +  fileNames[i] + "...")
    
        if type(boxes[i]['label']).__module__ == np.__name__:
            boxes[i]['label'] = [ boxes[i]['label'] ]
            boxes[i]['top'] = [ boxes[i]['top'] ]
            boxes[i]['height'] = [ boxes[i]['height'] ]
            boxes[i]['left'] = [ boxes[i]['left'] ]
            boxes[i]['width'] = [ boxes[i]['width'] ]
    
        # Creating a new file to each box ( number in the original image )    
        for number in range(0, len(boxes[i]['label']) ) :
                
            label = str(int(boxes[i]['label'][number]))

            try: 
                    print("Saving labels to csv file")
                    saveLabeltoCSV(pathImages + "labels.csv", label)
                
            except:
                  print("An exception occurred trying to save the new images.")
                                      
         
        print("\nFinished the step of getting labels from the dataset " + pathImages)

    return
# end of function


#                 Reading and gettin labels from train dataset                                    
gettingLabels(path + "../dataset/train/", path + '../dataset/train_digitStruct.mat')

#                 Reading and getting labels from test dataset                                    
gettingLabels(path + "../dataset/test/", path + '../dataset/test_digitStruct.mat')









