#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

def apply_color_range(hp, sp, vp, ha, sa, va, input_path):
    """Applies a color range to an image using OpenCV.

    Args:
        hp, sp, vp (int): Lower bound for the HSV color space.
        ha, sa, va (int): Upper bound for the HSV color space.
        input_path (str): Path to the input image.
    """
    img = cv2.imread(input_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([hp, sp, vp])
    upper = np.array([ha, sa, va])

    mask = cv2.inRange(hsv, lower, upper)

    cv2.imwrite(input_path, mask)
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
if __name__ == "__main__":
    input_path = "./05_test.png"
    apply_color_range(0, 43, 46, 10, 255, 255, input_path)
