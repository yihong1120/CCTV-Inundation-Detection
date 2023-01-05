#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
def threshold(threshold,grayscale_degree,input,output):
    img = cv2.imread(input)
    ret,threshold = cv2.threshold(img,threshold,grayscale_degree,cv2.THRESH_BINARY)
    cv2.imwrite(output, threshold)
    return
'''
input="/Users/YiHung/Downloads/0002_sobel_blur.png"
output="/Users/YiHung/Downloads/0002_sobel_blur_threshold.png"
threshold(60,255,input,output)
'''
