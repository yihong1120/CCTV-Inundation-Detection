#!/usr/bin/env python
# coding: utf-8

# In[12]:
import numpy as np
from PIL import Image
import cv2
def InRange(hp,sp,vp,ha,sa,va,INPUT):
    img = cv2.imread(INPUT)#overlay

    # OpenCV的顏色預設是BGR格式，這邊將其轉換為HSV格式
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 以HSV格式決定要提取的顏色範圍，顏色格式的說明請參考後續內容
    lower = np.array([hp,sp,vp]) #0,0,0
    upper = np.array([ha,sa,va]) #0,0,26
    # 將HSV影像的閾值設定為想要提取的顏色
    mask = cv2.inRange(hsv, lower, upper)
    # 使用bitwise_and()合併掩膜(mask)和原來的影像
    #img_specific = cv2.bitwise_and(img,img, mask= mask)
    #result = cv2.bitwise_and(image, image, mask)
    # 展示原圖、掩膜、抽取顏色後的影像
    cv2.imwrite(INPUT, mask)
    #os.remove(base_path_overlay+str(min_in_file(base_path_overlay))+".png")
    '''
    # 输入文件
    img = Image.open(INPUT)

    width = img.size[0]#长度
    height = img.size[1]#宽度
    img = img.convert("RGB")
    for i in range(img.width):
        for j in range(img.height):
            r,g,b = img.getpixel((i,j))
            if(r == 0 and g == 0 and b== 0):
                r=65
                g=105
                b=225
            img.putpixel((i,j), (r,g,b))
    # 输出图片
    img.save(INPUT)
    #os.remove(base_path_InRange+str(min_in_file(base_path_InRange))+".png")
    '''
    src = cv2.imread(INPUT)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,254,255,cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV
    b, g, r = cv2.split(src)
    rgba = [b,g,r,alpha]
    #b,g,r,
    dst = cv2.merge(rgba,4)
    cv2.imwrite(INPUT, dst)
    #os.remove(base_path_couleurs_bleu+str(min_in_file(base_path_couleurs_bleu))+".png")
    return
if __name__=="__main__":
    input="./05_test.png"
    InRange(0,43,46,10,255,255,input)
