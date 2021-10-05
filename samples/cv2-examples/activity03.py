# -*- coding: utf-8 -*-
"""
Activity 03 of workshop 01
Date: 10/07/2021
Author: Diego Bueno da Silva
e-mail: d.bueno.da.silva.10@student.scu.edu.au
ID: 23567850

 Practising OpenCV 

Credits: https://jspaint.app/

Issue with MacOS:
    https://github.com/mluerig/phenopype/issues/5
    https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv

"""

import cv2
import sys
import pathlib

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

####################################################### 
# Activity 03 - Creating a window named               #
# 'My Test Image 2' and then display the image.       #
#######################################################

#	cv2.IMREAD_COLOR or 1: Read the image in colour mode.
#	cv2.IMREAD_GRAYSCALE or 0: Read the image in grayscale mode.
#	cv2.IMREAD_UNCHANGED or -1: Read the image with alpha channels.

testingImg = cv2.imread(path + "test.png", cv2.IMREAD_GRAYSCALE)

# WINDOW_FULLSCREEN => doesn't work on MacOS
cv2.namedWindow("My test image 2", cv2.WINDOW_NORMAL)
cv2.imshow("My first test image window", testingImg)
cv2.waitKey(5000)
cv2.destroyWindow("My test image 2")
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(4) #bug on MacOS. Need to add waitKey here