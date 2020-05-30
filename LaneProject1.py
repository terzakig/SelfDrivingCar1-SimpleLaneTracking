# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:37:36 2016

@author: george
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
import random

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image



def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size, sigma2):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma2)

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
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
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
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1.0, offset = 0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, offset)
    

# distance of point from line (defined by end -points)
def distP(line, x0, y0):
    for x1, y1, x2, y2 in line:
        B = x2-x1
        A = -(y2-y1)
        C = x1 * (y2-y1) - y1*(x2-x1)
    dist = math.fabs(A*x0+B*y0+C) / math.sqrt(A*A + B*B)

    return dist

# get the "average" line, dispose outliers and return 
# the two lines that make up the lane
def bestLineInGroup(group, height):
    # computing the average cross section 
    # coordinate in the x-axis with the lower part of the screen
    best = []
    bestScore = 999999
    n = 50
    while n > 0:
        s = random.choice(group)
        for sx1, sy1, sx2, sy2 in s:
           B = sx2-sx1
           A = -(sy2-sy1)
           C = sx1 * (sy2-sy1) - sy1*(sx2-sx1)
           score = 0
           for line in group:
               for x1, y1, x2, y2 in line:
                   score += ( math.fabs(A*x1+B*y1+C) + math.fabs(A*x2+B*y2+C) ) / math.sqrt(A*A+B*B)
        if score < bestScore:
            bestScore = score
            best = s
        n -= 1
        
    return best
    
# vcreate an elongated version of the segment for drawing purposes
def demoSegment(line, height):
    h = height
    l = height / 2 + h/10
    for x1, y1, x2, y2 in line:
        B = x2-x1
        A = -(y2-y1)
        C = x1 * (y2-y1) - y1*(x2-x1)
        
        xl = -(C + B * l) / A
        xh = -(C + B * h) / A

    return [[xl, l, xh,h]]
           
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    gray = grayscale(image)


    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, 7, 2)

    # Define our parameters for Canny and apply
    low_threshold = 40
    high_threshold = 100
    
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]
    margin_pixels_down = 50
    margin_pixels_up = 450
    vertices = np.array([[(margin_pixels_down,height),(margin_pixels_up, height/2+50), (width-margin_pixels_up, height/2+50), (width-margin_pixels_down,height)]], dtype=np.int32)
    #cv2.fillPoly(mask, vertices, ignore_mask_color)
    #masked_edges = cv2.bitwise_and(edges, mask)

    masked_edges = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 4     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # ditch the lines with slope not between AngleLow-AngleHigh degrees
    angleLow = 20 * np.pi / 180
    angleHigh = 60 * np.pi / 180
    lines1 = []
    group1 = [] # group #1 of lines with roughly the same orientation
    group2 = [] # group #2 of lines with roughly the same orientation
    for line in lines: 
        for x1, y1, x2, y2 in line:
            slope = math.atan2(y2-y1,x2-x1)
            if (slope >= angleLow) and (slope <= angleHigh): 
                lines1.append(line)
                group1.append(line)
                
            if (slope <= -angleLow) and (slope >= -angleHigh):
                lines1.append(line)
                group2.append(line)
    
                
    # trying RANSAC 
    if len(group1) > 0 and len(group2)>0:
        n = 100
        bestLine1 = group1[0]
        bestLine2 = group2[0]
        bestScore = 9999999
        bestInliers1 = []
        bestInliers2 = []
        cutoffScore = 20 # 10 pixels robust cutoff score
        while n > 0:
            # choosing two lines from the group
            s1 = random.choice(group1)
            s2 = random.choice(group2)
            # finding inttersection of these two segments
            p1 = s1[0][0] # x1
            p2 = s1[0][1] # y1
            u1 = s1[0][2] - s1[0][0] # x2-x1
            u2 = s1[0][3] - s1[0][1] # y2-y1
            q1 = s2[0][0] # x1
            q2 = s2[0][1] # y1
            v1 = s2[0][2] - s2[0][0] # x2-x1
            v2 = s2[0][3] - s2[0][1] # y2-y1
        
            d = v1*u2-u1*v2 # this is a determinant which should not be zero 
            lamda = ( u2 * (p1-q1) - u1*(p2-q2) ) / d
            # thus, the intersection is:
            x0 = q1 + lamda * v1
            y0 = q2 + lamda * v2
        
            # now find the distance of all lines in the group
            inliers1 = []
            score1 = 0
            for line in group1:
                d = distP(line, x0, y0)
                if d < cutoffScore:
                    inliers1.append(line)
                    score1 += d
                else:
                    score1 += cutoffScore
                inliers2 = []
            score2 = 0
            for line in group2:
                d = distP(line, x0, y0)
                if d < cutoffScore:
                    inliers2.append(line)
                    score2 += d
                else:
                    score2 += cutoffScore
            score = score1 + score2
            if bestScore > score:
                bestScore = score
                bestLine1 = s1
                bestLine2 = s2
                bestInliers1 = inliers1
                bestInliers2 = inliers2
            # concatenate the inliers for debugging
            allinliers = bestInliers1 + bestInliers2
            n = n - 1
    else:
        allinliers = lines1
    
    # now finding the best line in each group
    # and drawing it 
    if (len(group1) > 0):
        segment = bestLineInGroup(group1, height)
        # get the largest segment for illustration drawing
        drawsegment = demoSegment(segment, height)
        for x1, y1, x2, y2 in drawsegment:    
                cv2.line(line_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),13)
    
    if (len(group2) > 0):
        segment = bestLineInGroup(group2, height)
        drawsegment = demoSegment(segment, height)
        for x1, y1, x2, y2 in drawsegment:    
            cv2.line(line_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),13)
    
    # Iterate over the output "lines" and draw lines on a blank image
#    for line in allinliers:
#        for x1,y1,x2,y2 in line:
#            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    #lines_img = cv2.addWeighted(color_edges, 0.5, line_image, 1, 0)
    
    
    
    return line_image

######################## TEST ON IMAGES ######################

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
#imagenames = os.listdir("test_images/")
#for i in range(len(imagenames)):
#    img = (mpimg.imread("test_images/"+imagenames[i])*255).astype('uint8')
#    lineimg = process_image(img)
#    name = imagenames[i][0:len(img)-4]
#    ext = imagenames[i][len(img)-4:len(img)]
#    cv2.imwrite("test_images/"+name+'-lines'+".jpg", lineimg)
#    plt.imshow(lineimg)
#    print("test_images/"+name+'-lines'+".jpg")
    

################### TEST ON VIDEO ############################


# 1. White video
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

#Yellow video
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)



