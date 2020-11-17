#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:24:35 2020

@author: rastislav
"""

import cv2
import imutils
import matplotlib.pyplot as plt

# load the image, convert it to grayscale, and blur it slightly
imgPath="/home/rastislav/Desktop/bp/brain_tumor_vgg/dataset-bigger-spl/train/no/2.jpg"

image = cv2.imread(imgPath)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(grayImage, (5, 5), 0)


# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh1 = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.erode(thresh1, None, iterations=2)
thresh = cv2.dilate(thresh2, None, iterations=2)

# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)
# show the output image

default_img=cv2.imread(imgPath)
# crop
ADD_PIXELS = 0
new_img = default_img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

"""
    Print all the images and filters 
"""

my_dpi=200
fig=plt.figure(figsize=(12,8),dpi=my_dpi)

ax1=fig.add_subplot(2,3,1)
ax1.set_title("1: Default img.")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(default_img)

ax2=fig.add_subplot(2,3,2)
ax2.set_title("2: As grayscale.")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(grayImage)

ax3=fig.add_subplot(2,3,3)
ax3.set_title("3: Added blur.")
ax3.set_xticks([])
ax3.set_yticks([])
ax3.imshow(gray)

ax4=fig.add_subplot(2,3,4)
ax4.set_title("4: Added treshold")
ax4.set_xticks([])
ax4.set_yticks([])
ax4.imshow(thresh1)

ax5=fig.add_subplot(2,3,5)
ax5.set_title("5: Added erosion")
ax5.set_xticks([])
ax5.set_yticks([])
ax5.imshow(thresh2)

ax6=fig.add_subplot(2,3,6)
ax6.set_title("6: Added dilatition")
ax6.set_xticks([])
ax6.set_yticks([])
ax6.imshow(thresh)

plt.show()

plt.close('all')
plt.imshow(image,interpolation='nearest') 
plt.title("Find contour in the images with filters")


plt.show()

plt.close('all')
plt.imshow(new_img,interpolation='nearest') 
plt.title("New, cropped image")

plt.show()
