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
from functions import *
import os

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

directory = os.path.dirname("results/")
if not os.path.exists(directory):
    os.makedirs(directory)
        
#%%
# next video
white_output = 'results\white.mp4'
clip1 = VideoFileClip('input_videos\solidWhiteRight.mp4')
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#%time 
white_clip.write_videofile(white_output, audio=False)
print("first video done")


# next video
yellow_output = 'results\yellow.mp4'
clip2 = VideoFileClip('input_videos\solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
#%time
yellow_clip.write_videofile(yellow_output, audio=False)
print("second video done")

#%%
# next video
challenge_output = 'results\chal.mp4'
clip2 = VideoFileClip('input_videos\challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
#%time
challenge_clip.write_videofile(challenge_output, audio=False)
print("extra video done")
