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
#%clear
for image_name in list_images:
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
    result = my_pipeline(image)
    
    return result

##%%
## next video
#white_output = folder_files + '\white.mp4'
#clip1 = VideoFileClip(folder_files + "\solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
##%time 
#white_clip.write_videofile(white_output, audio=False)
#print("first video done")
#
##%%
## next video
#yellow_output = folder_files + '\yellow.mp4'
#clip2 = VideoFileClip(folder_files + '\solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image)
##%time
#yellow_clip.write_videofile(yellow_output, audio=False)
#print("second video done")

#%%
# next video
challenge_output = folder_files + '\chal.mp4'
clip2 = VideoFileClip(folder_files + '\challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
#%time
challenge_clip.write_videofile(challenge_output, audio=False)
print("extra video done")
