# -*- coding: utf-8 -*-
"""
File: preprocessing.py
Project: Object Recognition System
Date: 29/08/2021
Author: Diego Bueno da Silva
e-mail: d.bueno.da.silva.10@student.scu.edu.au
ID: 23567850

Dataset source: http://ufldl.stanford.edu/housenumbers/

Issue with MacOS:
    https://github.com/mluerig/phenopype/issues/5
    https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv

"""
import cv2
import sys
import numpy as np
import pathlib
from matplotlib import pyplot as plt

path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)

pathGrayImages = path + "dataset/test/gray/"


"""	                Simple Thresholding attempt

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_GRAYSCALE )
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

#cv2.imwrite(pathGrayImages + fileNameWithoutPNG + "-" + str(number) + ".png", roi)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(0,6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

"""



"""	                Adaptive Thresholding attempt



img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_GRAYSCALE )
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(0,4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

"""




"""             CLAHE (Contrast Limited Adaptive Histogram Equalization)  attempt

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_GRAYSCALE )

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

#cv2.imwrite(pathGrayImages + '3-0-clahe.png',cl1)


"""

"""                 Otsu’s Binarization simulation

# global thresholding
ret1,th1 = cv2.threshold(equalisedImg,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(equalisedImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(equalisedImg,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [equalisedImg, 0, th1,
          equalisedImg, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(0,3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

"""


"""                 Histograms Equalization  + Otsu’s Binarization

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_GRAYSCALE )
equalisedImg = cv2.equalizeHist(img)
#cv2.imwrite(pathGrayImages + "3-0-res.png",equalisedImg)

# Otsu's thresholding
ret2,th2 = cv2.threshold(equalisedImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
"""



"""                	Image Filtering (2D Convolution) attempt


pathGrayImages = path + "dataset/test/colour/"

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
"""


"""                	Image Smoothing (Image Blurring) attempt


pathGrayImages = path + "dataset/test/colour/"

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )


blur = cv2.blur(img,(5,5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

"""

"""                	Gaussian Filtering (Image Blurring) attempt
"""

pathGrayImages = path + "dataset/test/colour/"

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )

blur = cv2.GaussianBlur(img,(5,5),0)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


"""                Median Filtering  (Image Blurring) attempt


pathGrayImages = path + "dataset/test/colour/"

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )

median = cv2.medianBlur(img,5)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median Filtering')
plt.xticks([]), plt.yticks([])
plt.show()

"""


"""                 Bilateral Filtering attempt

pathGrayImages = path + "dataset/test/colour/"

img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )

blur = cv2.bilateralFilter(img,9,75,75)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()

"""




"""                 Image Gradients attempt

pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

""" 


"""                 Image Gradients attempt Sobel CV 8U

pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )


# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()

"""



"""                 Image Gradients	Sobel Edge Detection attempt


from PIL import Image

# Open the image
pathGrayImages = path + "dataset/test/colour/"
img = np.array(Image.open(pathGrayImages + "3-0.png")).astype(np.uint8)

# Apply grey scale
gray_img = np.round(0.299 * img[:, :, 0] +0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

# Sobel Operator
h, w = gray_img.shape
# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

# define images with 0s
newhorizontalImage = np.zeros((h, w))
newverticalImage = np.zeros((h, w))
newgradientImage = np.zeros((h, w))

# offset by 1
for i in range(1, h - 1):
    for j in range(1, w - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        newverticalImage[i - 1, j - 1] = abs(verticalGrad)

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        newgradientImage[i - 1, j - 1] = mag

plt.figure()
plt.title('3-0.png')
plt.imsave('3-0-sobel.png', newgradientImage, cmap='gray', format='png')
plt.imshow(newgradientImage, cmap='gray')
plt.show()

"""


"""                 Image Gradients	Sobel Edge Detection RGB attempt


from PIL import Image

# Open the image
pathGrayImages = path + "dataset/test/colour/"
img = np.array(Image.open(pathGrayImages + "3-0.png")).astype(np.uint8)

# Sobel Operator
h, w, d = img.shape

# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

# define images with 0s
newgradientImage = np.zeros((h, w, d))

# offset by 1
for channel in range(d):
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1, channel]) + \
                             (horizontal[0, 1] * img[i - 1, j, channel]) + \
                             (horizontal[0, 2] * img[i - 1, j + 1, channel]) + \
                             (horizontal[1, 0] * img[i, j - 1, channel]) + \
                             (horizontal[1, 1] * img[i, j, channel]) + \
                             (horizontal[1, 2] * img[i, j + 1, channel]) + \
                             (horizontal[2, 0] * img[i + 1, j - 1, channel]) + \
                             (horizontal[2, 1] * img[i + 1, j, channel]) + \
                             (horizontal[2, 2] * img[i + 1, j + 1, channel])

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1, channel]) + \
                           (vertical[0, 1] * img[i - 1, j, channel]) + \
                           (vertical[0, 2] * img[i - 1, j + 1, channel]) + \
                           (vertical[1, 0] * img[i, j - 1, channel]) + \
                           (vertical[1, 1] * img[i, j, channel]) + \
                           (vertical[1, 2] * img[i, j + 1, channel]) + \
                           (vertical[2, 0] * img[i + 1, j - 1, channel]) + \
                           (vertical[2, 1] * img[i + 1, j, channel]) + \
                           (vertical[2, 2] * img[i + 1, j + 1, channel])

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            # Avoid underflow: clip result
            newgradientImage[i - 1, j - 1, channel] = mag

# now add the images r g and b
rgb_edge = newgradientImage[:,:,0] + newgradientImage[:,:,1] + newgradientImage[:,:,2]

plt.figure()
plt.title('3-0-sobel-rgb.png')
plt.imsave('3-0-sobel-rgb.png', rgb_edge, cmap='gray', format='png')
plt.imshow(rgb_edge, cmap='gray')
plt.show()

"""



"""                 Laplacian Operator


# Open the image
pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_UNCHANGED )

# Apply grey scale
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply gaussian blur
blur_img = cv2.GaussianBlur(grey_img, (3, 3), 0)

# Positive Laplacian Operator
laplacian = cv2.Laplacian(blur_img, cv2.CV_64F)

plt.figure()
plt.title('Shapes')
plt.imsave('shapes-lap.png', laplacian, cmap='gray', format='png')
plt.imshow(laplacian, cmap='gray')
plt.show()

"""



"""                 Homomorphic filter attempt


from math import exp, sqrt

# Open the image
pathGrayImages = path + "dataset/test/colour/"
image = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_GRAYSCALE )
height, width = image.shape
dft_M = cv2.getOptimalDFTSize(height)
dft_N = cv2.getOptimalDFTSize(width)

#Initialise the global parameters of the formula
yh, yl, c, d0, = 0, 0, 0, 0
# Initialization of global parameters set by the user
y_track, d0_track, c_track = 0, 0, 0
complex = 0 

def homomorphic():
    global yh, yl, c, d0, complex
    du = np.zeros(complex.shape, dtype = np.float32)
    #H(u, v)
    for u in range(dft_M):
        for v in range(dft_N):
            du[u,v] = sqrt((u - dft_M/2.0)*(u - dft_M/2.0) + (v - dft_N/2.0)*(v - dft_N/2.0))

    du2 = cv2.multiply(du,du) / (d0*d0)
    re = np.exp(- c * du2)
    H = (yh - yl) * (1 - re) + yl
    #S(u, v)
    filtered = cv2.mulSpectrums(complex, H, 0)
     #inverse DFT (does the shift back first)
    filtered = np.fft.ifftshift(filtered)
    filtered = cv2.idft(filtered)
    #normalization to be representable 
    filtered = cv2.magnitude(filtered[:, :, 0], filtered[:, :, 1])
    cv2.normalize(filtered, filtered, 0, 1, cv2.NORM_MINMAX)
    #g(x, y) = exp(s(x, y))
    filtered = np.exp(filtered)
    cv2.normalize(filtered, filtered,0, 1, cv2.NORM_MINMAX)
    
    cv2.namedWindow('homomorphic', cv2.WINDOW_NORMAL)
    cv2.imshow("homomorphic", filtered)
    cv2.resizeWindow("homomorphic", 600, 550)
    
def setyl(y_track):
    global yl
    yl = y_track
    if yl == 0:
        yl = 1
    if yl > yh:
        yl = yh - 1
    homomorphic()

def setyh(y_track):
    global yh
    yh = y_track
    if yh == 0:
        yh = 1
    if yl > yh:
        yh = yl + 1
    homomorphic()

def setc(c_track):
    global c
    c = c_track/1000.0
    if c == 0:
        c_track = 1    
    homomorphic()

def setd0(d0_track):
    global d0
    d0 = d0_track
    if d0 == 0:
        d0 = 1
    homomorphic()

def main():
    # copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) 
    # BORDER_CONSTANT = Pad the image with a constant value (i.e. black or 0)
    padded = cv2.copyMakeBorder(image, 0, dft_M - height, 0, dft_N - width, cv2.BORDER_CONSTANT, 0) 
    # +1 pra tratar log(0)
    padded = np.log(padded+1)
    global complex
    complex = cv2.dft(np.float32(padded)/255.0, flags = cv2.DFT_COMPLEX_OUTPUT)
    complex = np.fft.fftshift(complex)
    img = 20 * np.log(cv2.magnitude(complex[:,:,0], complex[:,:,1]))
    
#    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
#    cv2.imshow("Image", image)
    cv2.imwrite(pathGrayImages + "teste2.png", image)
#    cv2.resizeWindow("Image", 400, 400)


 #   cv2.namedWindow('DFT', cv2.WINDOW_NORMAL)
 #   cv2.imshow("DFT", np.uint8(img))
    cv2.imwrite( pathGrayImages + "teste3.png", np.uint8(img))
    #cv2.resizeWindow("DFT", 250, 250)

#    cv2.createTrackbar("YL", "Image", y_track, 100, setyl)
#    cv2.createTrackbar("YH", "Image", y_track, 100, setyh)
#    cv2.createTrackbar("C", "Image", c_track, 100, setc)
#    cv2.createTrackbar("D0", "Image", d0_track, 100, setd0)
 
#    cv2.waitKey(0)     
#    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

"""


"""                 Homomorphic filter (Discrete Fourier Transform) attempt


from math import exp, sqrt

#http://photoandtravels.blogspot.com/2011/04/against-light-photography.html
pathGrayImages = path + "dataset/test/colour/"
image = cv2.imread(pathGrayImages + "3-0.png", cv2.IMREAD_GRAYSCALE )
height, width = image.shape
dft_M = cv2.getOptimalDFTSize(height)
dft_N = cv2.getOptimalDFTSize(width)

#Initialise the global parameters of the formula
yh, yl, c, d0, = 0, 0, 0, 0
# Initialization of global parameters set by the user
y_track, d0_track, c_track = 0, 0, 0
complex = 0 

def homomorphic():
    global yh, yl, c, d0, complex
    du = np.zeros(complex.shape, dtype = np.float32)
    #H(u, v)
    for u in range(dft_M):
        for v in range(dft_N):
            du[u,v] = sqrt((u - dft_M/2.0)*(u - dft_M/2.0) + (v - dft_N/2.0)*(v - dft_N/2.0))

    du2 = cv2.multiply(du,du) / (d0*d0)
    re = np.exp(- c * du2)
    H = (yh - yl) * (1 - re) + yl
    #S(u, v)
    filtered = cv2.mulSpectrums(complex, H, 0)
     #inverse DFT (does the shift back first)
    filtered = np.fft.ifftshift(filtered)
    filtered = cv2.idft(filtered)
    #normalization to be representable 
    filtered = cv2.magnitude(filtered[:, :, 0], filtered[:, :, 1])
    cv2.normalize(filtered, filtered, 0, 1, cv2.NORM_MINMAX)
    #g(x, y) = exp(s(x, y))
    filtered = np.exp(filtered)
    cv2.normalize(filtered, filtered,0, 1, cv2.NORM_MINMAX)
    
    cv2.namedWindow('homomorphic', cv2.WINDOW_NORMAL)
    cv2.imshow("homomorphic", filtered)
    cv2.resizeWindow("homomorphic", 600, 550)
    
def setyl(y_track):
    global yl
    yl = y_track
    if yl == 0:
        yl = 1
    if yl > yh:
        yl = yh - 1
    homomorphic()

def setyh(y_track):
    global yh
    yh = y_track
    if yh == 0:
        yh = 1
    if yl > yh:
        yh = yl + 1
    homomorphic()

def setc(c_track):
    global c
    c = c_track/1000.0
    if c == 0:
        c_track = 1    
    homomorphic()

def setd0(d0_track):
    global d0
    d0 = d0_track
    if d0 == 0:
        d0 = 1
    homomorphic()

def main():
    # copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) 
    # BORDER_CONSTANT = Pad the image with a constant value (i.e. black or 0)
    padded = cv2.copyMakeBorder(image, 0, dft_M - height, 0, dft_N - width, cv2.BORDER_CONSTANT, 0) 
    # +1 pra tratar log(0)
    padded = np.log(padded+1)
    global complex
    complex = cv2.dft(np.float32(padded)/255.0, flags = cv2.DFT_COMPLEX_OUTPUT)
    complex = np.fft.fftshift(complex)
    img = 20 * np.log(cv2.magnitude(complex[:,:,0], complex[:,:,1]))
    
    cv2.imwrite( pathGrayImages + "3-0-dft.jpg", np.uint8(img))
 

if __name__ == '__main__':
    main()

"""


"""            Harris Corner Detection attempt


pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png" )
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[125,125,0]

cv2.imwrite( pathGrayImages + "3-0-dst.jpg", np.uint8(img))

""" 


"""            Finding corner with FAST function 


pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", 0 )

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp,None, color=(255,0,0))

#Print all default params
print("Threshold: ", fast.getThreshold());
print("nonmaxSuppression: ", fast.getNonmaxSuppression());
print("neighborhood: ", fast.getType());
print("Total Keypoints with nonmaxSuppression: ", len(kp));

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
#fast.setBool('nonmaxSuppression',0)
fast.setNonmaxSuppression(False);
kp = fast.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)

plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("with nonmaxSuppression")
plt.subplot(1,2,2)
plt.imshow(img3)
plt.title("without nonmaxSuppression")
plt.show()

"""




"""            Finding edges with Canny function


pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", 0 )
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

"""

""" erosion

pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", 1)

#Activity 12: The following program reads an image and applies the Erosion operation to the image using the 5x5 kernel.
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

cv2.imwrite( pathGrayImages + "3-0-erosion.jpg", erosion)

"""


"""  applies the Dilation operation to the image using the 5x5 kernel.

pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", 0)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 1)
cv2.imwrite( pathGrayImages + "3-0-5x5-kernel.jpg", dilation)

""" 

"""    applies the Opening operation to the image using the 5x5 kernel.


pathGrayImages = path + "dataset/test/colour/"
img = cv2.imread(pathGrayImages + "3-0.png", 0)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite( pathGrayImages + "3-0-5x5-kernel.jpg", closing)

""" 
