# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 13:55:54 2017

@author: Pica
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



def my_print():
    print("hola")


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel but NOTE: to see 
    the returned image as grayscale (assuming your grayscaled image is 
    called 'gray') you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once
    you want to average/extrapolate the line segments you detect to map out 
    the full extent of the lane (going from the result shown in 
    raw-lines-example.mp4 to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, \
                            np.array([]), \
                            minLineLength = min_line_len, \
                            maxLineGap = max_line_gap )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def my_pipeline(image):
    plot_all = 0 

    # Parameters of the detection
    ker_size = 5
    low_threshold = 50
    high_threshold = 150
    
    fraction_top = 1/100*60     # distance to the the top as fraction
    fraction_left = 1/100*16
    fraction_right = 1/100*84
    
    # Define the Hough transform parameters
    rho = 1*0.5 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 *0.1# angular resolution in radians of the Hough grid
    threshold = 15#4     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20#8    # maximum gap in pixels between connectable line segments

    if(plot_all):
        plt.imshow(image) 
        print("image")
        plt.show()
    
    
    im_gray = grayscale( image )
    
    
    if(plot_all):
        plt.imshow( im_gray ) 
        print("image gray")
        plt.show()
    
    im_blur = gaussian_blur( im_gray, ker_size )
    
    if(plot_all):
        plt.imshow( im_blur ) 
        print("image gray and blur")
        plt.show()
    
    im_canny = canny( im_blur , low_threshold, high_threshold)
    
    if(plot_all):
        plt.imshow( im_canny ) 
        print("canny")
        plt.show()
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    
    bottom = imshape[0]
    top = imshape[0]*fraction_top
    left = imshape[1]*fraction_left
    right = imshape[1]*fraction_right 
                   
    vertices = np.array([[  (left,bottom),\
                            (left, top), \
                            (right, top), \
                            (right,bottom)]], \
                            dtype=np.int32)
    
    im_cropped = region_of_interest( im_canny, vertices )
    #cv2.fillPoly(mask, vertices, ignore_mask_color)
    #im_cropped = cv2.bitwise_and(edges, mask)
    
    if(plot_all):
        plt.imshow( im_cropped ) 
        print("cropped")
        plt.show()
    
    # Hough transform
    # Run Hough on edge detected image
    im_hough = hough_lines(im_cropped, rho, theta, threshold, min_line_length, max_line_gap)
    
    if(plot_all):
        plt.imshow( im_hough ) 
        print("hough")
        plt.show()
    
    
    im_final = weighted_img(im_hough, image, α=0.8, β=1., λ=0.)
    if(plot_all):
        plt.imshow( im_final ) 
        print("Final image")
        plt.show()

    return im_final