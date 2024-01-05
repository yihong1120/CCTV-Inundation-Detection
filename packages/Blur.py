#!/usr/bin/env python
# coding: utf-8

# In[12]:
import cv2
import cv2
def blur(br1,br2,blurry,Input,Output):
    image = cv2.imread(Input)
    #result = np.hstack([canny])
    #ret,thresh1 = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(image, (br1, br2 ), blurry)
    
    cv2.imwrite(Output, blurred)
    return

In="/Users/YiHung/Downloads/0002_sobel.png"
Out="/Users/YiHung/Downloads/0002_sobel_blur.png"
blur(9,9,100,In,Out)
