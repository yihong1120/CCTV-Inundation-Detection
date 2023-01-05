#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
def grayscale(input,output):
    #gray the photos
    image = cv2.imread(input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output, gray)
    return
'''
input="/Users/YiHung/Downloads/0002_com.png"
output="/Users/YiHung/Downloads/後來灰階.png"
grayscale(input,output)
'''
