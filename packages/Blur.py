#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
# Removed unused import argparse
# numpy import not used, so it is also removed
def apply_gaussian_blur(br1, br2, blurry, input_path, output_path):
    """Applies a Gaussian blur to an image using OpenCV.

    Args:
        br1 (int): Width (in pixels) of the kernel.
        br2 (int): Height (in pixels) of the kernel.
        blurry (int): Standard deviation in the X and Y directions for the Gaussian kernel.
        input_path (str): Path to the input image.
        output_path (str): Path to save the blurred image.
    """
    image = cv2.imread(input_path)

    blurred = cv2.GaussianBlur(image, (br1, br2 ), blurry)
    
    cv2.imwrite(output_path, blurred)
    return

input_path="/path/to/input_image.png"
output_path="/path/to/output_blurred_image.png"
apply_gaussian_blur(9, 9, 100, input_path, output_path)
