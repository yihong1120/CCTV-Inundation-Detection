#!/usr/bin/env python
# coding: utf-8

# In[12]:
from PIL import Image
from PIL import ImageChops
def compare(fichier1,fichier2,out_f): #file route1, file route2, output route
    #comparer le photos
    image_one = Image.open(fichier1)
    image_two = Image.open(fichier2)
    img1 = image_one.convert("RGB")
    img2 = image_two.convert("RGB")
    diff = ImageChops.difference(img1, img2)
    diff.save(out_f)
    return
'''
input1="/Users/YiHung/Downloads/00.png"
input2="/Users/YiHung/Downloads/02.png"
output="/Users/YiHung/Downloads/0002_com.png"
compare(input1,input2,output)
'''
