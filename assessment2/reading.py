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

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

# mat73 lib takes long to read test_digitStruct.mat

print( "\nReading the metadados digitStruct.mat\n" )
print( "It may take time, be patience...\n" )

data_dict = mat73.loadmat(path + 'dataset/test_digitStruct.mat')    
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

""""
Reading all the imagens, getting ROI and saving as GRAYSCALE mode

"""

#	cv2.IMREAD_COLOR or 1: Read the image in colour mode.
#	cv2.IMREAD_GRAYSCALE or 0: Read the image in grayscale mode.
#	cv2.IMREAD_UNCHANGED or -1: Read the image with alpha channels.

pathImages = path + "dataset/test/"
pathGrayImages = path + "dataset/test/gray/"
for i in range(0,len(fileNames)):
    print( "Reading file " +  fileNames[i])
    readImg = cv2.imread(pathImages + fileNames[i], cv2.IMREAD_GRAYSCALE )
    
    """
       Getting Region of Interest (ROI)  
       
       boxes[i] is an array with information of all numbers in the image
       The region of interest is given for each number in the image by 
       reading the digitStruct.mat file content.
       
    """    

    print( "Cutting off region of interest.\n" )

    original_height, original_width = readImg.shape;
    print( "Original size of " + fileNames[i] + " image: " + str(original_height) + " x " + str(original_width))

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

        print( "Saving the number " + str(int(boxes[i]['label'][number])) + " from the cut image " + fileNames[i] + " in gray scale mode \n\n")
        
        fileNameWithoutPNG = fileNames[i][0: (len(fileNames[i]) - 4) ]        
        
        """                 Histograms Equalization             """
#        equalisedImg = cv2.equalizeHist(roi)

        """                 Otsu's thresholding                 """            
        ret,threshold = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
        # counting the number of pixels
        number_of_white_pix = np.sum(threshold > 128)
        number_of_black_pix = np.sum(threshold <= 128)

        if ( number_of_white_pix < number_of_black_pix ): # black background
            threshold = (255 - threshold) # inverting the image
            
           

        """                 Laplacian Operator                  """        

        # Apply grey scale
        #grey_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply gaussian blur
        #blur_img = cv2.GaussianBlur(grey_img, (3, 3), 0)
        
        # Positive Laplacian Operator
        #laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)

        cv2.imwrite(pathGrayImages + fileNameWithoutPNG + "-" + str(number) + ".png", threshold)
        print( "New gray image file saved: " + fileNameWithoutPNG + "-" + str(number) + ".png" )
    
    print("Finished the step of reading the dataset " + pathImages)

