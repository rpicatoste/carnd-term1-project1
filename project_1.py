#%%

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
import os

#%matplotlib inline

folder_files = "CarND-LaneLines-P1"

list_images = os.listdir(folder_files + "/test_images/") 

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
%clear
for image_name in list_images:
    
#image_name = list_images[0]
    image = mpimg.imread(folder_files + '/test_images/' + image_name)
    im_result = my_pipeline(image)
    plt.imshow( im_result ) 
    print("Final image: " + image_name)
    plt.show()
    


print("done")

#%%

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    return result
