# -*- coding: utf-8 -*-
ß
"""
Activity 04 of workshop 01
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
import matplotlib.pyplot as plt
import sys
import pathlib

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

testimg = cv2.imread(path + "test.png")
 
####################################################### 
# Activity 04 - Reading an image ‘test.jpg' and       #
# convert it into grey. Using Matplotlib to display   #
#                                                     #
#######################################################


plt.pyplot.imshow(testingImg, cmap = 'gray', interpolation= 'bicubic')
#plt.show()
#plt.imshow(testingImg)
plt.pyplot.xticks([]), plt.pyplot.yticks([])
plt.pyplot.show()

cv2.waitKey(4) #bug on MacOS. Need to add waitKey here