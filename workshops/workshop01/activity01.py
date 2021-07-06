# -*- coding: utf-8 -*-

"""
Activity 01 of workshop 01
Date: 06/07/2021
Author: Diego Bueno da Silva
e-mail: d.bueno.da.silva.10@student.scu.edu.au
ID: 23567850

 Practising OpenCV 

Credits: https://jspaint.app/

"""

import cv2
import sys
import pathlib

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

####################################################### 
# Activity 01 - Reading an image and showing the      #
# values of pixels in the first row of the image.     #
#######################################################

#	cv2.IMREAD_COLOR or 1: Read the image in colour mode.
#	cv2.IMREAD_GRAYSCALE or 0: Read the image in greyscale mode.
#	cv2.IMREAD_UNCHANGED or -1: Read the image with alpha channels.

testingImg = cv2.imread(path + "test.png", cv2.IMREAD_GRAYSCALE)

if len(testingImg) > 0:
    
    print("First row value prixels:")
    print(testingImg[0:1,:]) ## all white ( 255  )
    
    
####################################################### 
# Activity 02 - Displaying the test image in an       #
# window.                                             #
#######################################################

cv2.imshow("My test image:", testingImg)
cv2.waitKey(0) # some issues on MacOS to close the window
cv2.destroyAllWindows()

####################################################### 
# Activity 03 - Creating a window named               #
# 'My Test Image 2' and then display the image.       #
#######################################################

# WINDOW_FULLSCREEN => doesn't work on MacOS
cv2.namedWindow("My test image 2", cv2.WINDOW_NORMAL)
cv2.imshow("My first test image window", testingImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
 



    
    
    
