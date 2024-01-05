#!/usr/bin/env python
# coding: utf-8

# In[12]:
import numpy as np
import cv2
def apply_morphological_operations(input_path, output_path):
    """Applies morphological operations to an image using OpenCV.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path where the processed image should be saved.
    """
    
    img = cv2.imread(input)

    # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    
    cv2.imwrite(output_path, closing)
    return
'''

'''
