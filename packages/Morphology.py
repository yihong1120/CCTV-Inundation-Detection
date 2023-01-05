#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
import numpy as np
def morphology(input,output):
    
    img = cv2.imread(input)
    '''
    # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opening', opening)
    '''
    # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('closing', closing)
    
    cv2.imwrite(output, closing)
    return
'''
input="/Users/YiHung/Downloads/0002_sobel_blur_threshold.png"
output="/Users/YiHung/Downloads/0002_sobel_blur_threshold_morphologCL.png"
morphology(input,output)
'''
