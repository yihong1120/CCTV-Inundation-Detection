#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
def sobel(input,output):
    image = cv2.imread(input)
    gradX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    cv2.imwrite(output, gradient)
    return
'''
input="/Users/YiHung/Downloads/0002_gcom.png"
output="/Users/YiHung/Downloads/0006_sobel.png"
sobel(input,output)
'''
