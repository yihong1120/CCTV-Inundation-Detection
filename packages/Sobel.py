#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2

def edge_detection_sobel(input_path: str, output_path: str) -> None:
    image = cv2.imread(input_path)
    gradX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    cv2.imwrite(output_path, gradient)
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
 This function performs edge detection using the Sobel operator.

 Args:
     input_path (str): The file path of the input image.
     output_path (str): The file path to save the output gradient image.
 Returns:
     None
 This function performs edge detection using the Sobel operator.

 Args:
     input_path (str): The file path of the input image.
     output_path (str): The file path to save the output gradient image.
 Returns:
     None
input="/Users/YiHung/Downloads/0002_gcom.png"
output="/Users/YiHung/Downloads/0006_sobel.png"
sobel(input,output)
'''
