# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 13:18:26 2017

@author: rpicatoste
"""


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from functions import *

#%matplotlib inline

folder_files = "CarND-LaneLines-P1"

#reading in an image
image = mpimg.imread(folder_files + '/test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  
#plt.imshow(grayscale(image))
# if you wanted to show a single color channel image called 'gray', 
# for example, call as plt.imshow(image, cmap='gray')



import os
list_images = os.listdir(folder_files + "/test_images/") 

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#%%

%clear

ker_size = 5
low_threshold = 50
high_threshold = 150

image_name = list_images[0]
image = mpimg.imread(folder_files + '/test_images/' + image_name)
plt.imshow(image) 

print("image")
plt.show()


im_gray = grayscale( image )
plt.imshow( im_gray ) 

print("image gray")
plt.show()

im_blur = gaussian_blur( im_gray, ker_size )
plt.imshow( im_blur ) 

print("image gray and blur")
plt.show()

im_canny = canny( im_blur , low_threshold, high_threshold)
plt.imshow( im_canny ) 

print("canny")
plt.show()



print("done")
