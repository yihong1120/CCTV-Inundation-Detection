import cv2
from PIL import Image
import numpy as np
def convex_hull(input, input_type = 'image', output = None):
    if input_type == 'image':
        img = cv2.imread(input)#polygon
    else:
        img = input
    #cv2.imshow("img", img)
    # 转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #img.save('./convex____2.png')
    #img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary, 2, 1)
    #contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull=cv2.convexHull(contours[0])

    #print(hull)
    #cv2.polylines(img, [hull], True, (0, 255, 0), 2)
    img=cv2.fillPoly(img, [hull], (255,255,255))
    if input_type == 'image':
        cv2.imwrite(output, img)
    else:
        return img

if __name__=="__main__":
    input='./14_0.png'
    output='./14_0.png'
    convex_hull(input,output)
