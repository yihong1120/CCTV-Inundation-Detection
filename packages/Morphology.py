#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
import numpy as np
def apply_morphological_operations(input_image_path, output_image_path):
    
    input_image = cv2.imread(input_image_path)
    '''
    
    
    
    #cv2.imshow('opening', opening)
    '''
    # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel)
    
    
    cv2.imwrite(output_image_path, closing)
    return
'''
input="/Users/YiHung/Downloads/0002_sobel_blur_threshold.png"
output="/Users/YiHung/Downloads/0002_sobel_blur_threshold_morphologCL.png"
morphology(input,output)
'''
