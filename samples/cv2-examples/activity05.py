# -*- coding: utf-8 -*-
ß
"""
Activity 05 of workshop 01
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
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

testimg = cv2.imread(path + "test.png")
b,g,r = cv2.split(testimg)
testimg2 = cv2.merge([r,g,b])
plt.subplot(121);plt.imshow(testimg) # expects distorted colour
plt.subplot(122);plt.imshow(testimg2) # expect true colour
plt.show()

cv2.startWindowThread()

cv2.imshow('bgr image',testimg) # expects true colour
#cv2.imshow('rgb image',testimg2) # expects distorted colour
cv2.waitKey(5000)
cv2.destroyWindow('bgr image')
cv2.waitKey(4) #bug
#cv2.destroyWindow('rgb image')
cv2.destroyAllWindows()
cv2.waitKey(4) #bug on MacOS. Need to add waitKey here