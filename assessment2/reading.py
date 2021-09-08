# -*- coding: utf-8 -*-
"""
File: reading.py
Project: Object Recognition System
Date: 29/08/2021
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

def readingAndPreProcessingImages(pathImages, matLabFileDatDict):
    
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
    
    
    """
        Defining diretories to save ROI applying chosen preprocessing methods
    """
    pathColour = pathImages + "colour-original/"
    pathColourPlusGaussian = pathImages + "colour-plus-gaussian-blur/"
    pathGrayScale = pathImages + "gray-scale-only/"
    pathGrayPlusHistEquPlusOtsuThr = pathImages + "gray-scale-plus-histograms-equalisation-and-otsu-thresholding/"
    pathGrayPlusLaplacianOperator = pathImages + "gray-scale-plus-laplacian-operator/"
    pathGrayPlusOtsuThr = pathImages + "gray-scale-plus-otsu-thresholding/"
    pathGrayPlusOtsuThrInvertingBgd = pathImages + "gray-scale-plus-otsu-thresholding-inverting-background/"
    
    print( "Reading dataset " + pathImages )
    
    for i in range(0,len(fileNames)):
    
        print( "Reading file " +  fileNames[i] + "...")
    
        readImg = cv2.imread(pathImages + fileNames[i], cv2.IMREAD_UNCHANGED )
        
        """
           Getting Region of Interest (ROI)  
           
           boxes[i] is an array with information of all numbers in the image
           The region of interest is given for each number in the image by 
           reading the digitStruct.mat file content.
           
        """    
    
        print( "Cutting off region of interest.\n" )
    
        original_height, original_width, channels = readImg.shape;
        print( "Original size of " + fileNames[i] + " image: " + str(original_height) + " x " + str(original_width) + " x " + str(channels) )
    
        # the metadados contains a single numpy array when there is only one number
        # in the image, and a list of numpy arrays when there are more than one number.
        # Converting to list to keep standard data type in the follow loop
        if type(boxes[i]['label']).__module__ == np.__name__:
            boxes[i]['label'] = [ boxes[i]['label'] ]
            boxes[i]['top'] = [ boxes[i]['top'] ]
            boxes[i]['height'] = [ boxes[i]['height'] ]
            boxes[i]['left'] = [ boxes[i]['left'] ]
            boxes[i]['width'] = [ boxes[i]['width'] ]
    
        # Creating a new file to each box ( number in the original image )    
        for number in range(0, len(boxes[i]['label']) ) :
                
            top    = int(boxes[i]['top'][number])
            height = int(boxes[i]['height'][number])
            left   = int(boxes[i]['left'][number])
            width  = int(boxes[i]['width'][number])
            
            roi = readImg[ top : top + height , left : left + width  ]
            label = str(int(boxes[i]['label'][number]))
            fileNameWithoutPNG = fileNames[i][0: (len(fileNames[i]) - 4) ]
    
            roi_height, roi_width, roi_channels = roi.shape;
            print( "ROI size of " + label + " in image " + fileNames[i] + ": " + str(roi_height) + " x " + str(roi_width) + " x " + str(roi_channels) )
    
            """ Pre-processing """
    
            if roi_height > 0 and roi_width > 0 and roi_channels > 0: # file 344.png is missing ROI dimensions in digitStruct.mat

                try: 
                
                    print("Pre-rocessing file " + pathImages + fileNames[i] + ". Cutting off image of the label " + label)        
            
                    print( "Saving the number " + label + " from the cut image " + fileNames[i] + " with the original colours\n\n")
                    cv2.imwrite(pathColour + fileNameWithoutPNG + "-" + str(number) + ".png", roi)
                    print( "New image file saved: " + pathColour + fileNameWithoutPNG + "-" + str(number) + ".png" )
            
            
                    print( "Applying Gaussian Filtering (Image Blurring) \n\n")
                    blur = cv2.GaussianBlur(roi,(5,5),0)
                    cv2.imwrite(pathColourPlusGaussian + fileNameWithoutPNG + "-" + str(number) + ".png", blur)
                    print( "New image file saved: " + pathColourPlusGaussian + fileNameWithoutPNG + "-" + str(number) + ".png" )
            
                    print( "Converting to gray scale \n\n")
                    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(pathGrayScale + fileNameWithoutPNG + "-" + str(number) + ".png", gray)
                    print( "New image file saved: " + pathGrayScale + fileNameWithoutPNG + "-" + str(number) + ".png" )
            
                    print( "Applying Otsu's thresholding in the gray scale image \n\n")
                    ret,otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cv2.imwrite(pathGrayPlusOtsuThr + fileNameWithoutPNG + "-" + str(number) + ".png", otsu)
                    print( "New image file saved: " + pathGrayPlusOtsuThr + fileNameWithoutPNG + "-" + str(number) + ".png" )
            
                    print( "Applying Histograms Equalization and Otsu's thresholding in the gray scale image \n\n")
                    equalisedImg = cv2.equalizeHist(gray)        
                    ret,equalisedImg_otsu = cv2.threshold(equalisedImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cv2.imwrite(pathGrayPlusHistEquPlusOtsuThr + fileNameWithoutPNG + "-" + str(number) + ".png", equalisedImg_otsu)
                    print( "New image file saved: " + pathGrayPlusHistEquPlusOtsuThr + fileNameWithoutPNG + "-" + str(number) + ".png" )
            
                    print( "Applying Laplacian Operator in the gray scale image \n\n")        
                    blur_img = cv2.GaussianBlur(gray, (3, 3), 0) # Apply gaussian blur
                    laplacian = cv2.Laplacian(blur_img, cv2.CV_64F) # Positive Laplacian Operator        
                    cv2.imwrite(pathGrayPlusLaplacianOperator + fileNameWithoutPNG + "-" + str(number) + ".png", laplacian)
                    print( "New image file saved: " + pathGrayPlusLaplacianOperator + fileNameWithoutPNG + "-" + str(number) + ".png" )
            
                    print( "Applying Otsu's thresholding in the gray scale image and converting backgroup to white \n\n")
                    # counting the number of pixels. Used 128 as thredshold to split into black and white pixels
                    number_of_white_pixels = np.sum(otsu > 128)
                    number_of_black_pixels = np.sum(otsu <= 128)
            
                    if ( number_of_white_pixels < number_of_black_pixels ): # assuming it has black background
                        otsu = (255 - otsu) # inverting the image
            
                    cv2.imwrite(pathGrayPlusOtsuThrInvertingBgd + fileNameWithoutPNG + "-" + str(number) + ".png", otsu)
                    print( "New image file saved: " + pathGrayPlusOtsuThrInvertingBgd + fileNameWithoutPNG + "-" + str(number) + ".png" )
                
                
                    print("Saving labels to csv file")
                    saveLabeltoCSV(pathImages + "labels.csv", label)
                
                except:
                  print("An exception occurred trying to save the new images.")
                                      
            else :
                print( "Error to pre-processing ROI of " + label + " in image " + fileNames[i] )            
        
        print("\nFinished the step of reading the dataset " + pathImages)

    return
# end of readingAndPreProcessingImages function


#                 Reading and pre-processing train dataset                                    
#readingAndPreProcessingImages(path + "dataset/train/", path + 'dataset/train_digitStruct.mat')

#                 Reading and pre-processing test dataset                                    
readingAndPreProcessingImages(path + "dataset/test/", path + 'dataset/test_digitStruct.mat')









