import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.getcwd()))
import img_adjust_tous_points

def draw_polygon(img, matrice, input_type = 'image', output = None,):
    if input_type == 'image':
        img=cv2.imread(img)
        
    img=img.shape
    size = (img[0], img[1])
    # 全黑.可以用在屏保
    img = np.zeros(size)
    # Create a black image
    #####img = np.zeros((1080,1920,3), np.uint8) #y,x
    #####l1=img_adjust_tous_points.region_cal("./edge_enhanse.png")
    #pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    #pts = np.array([[1161,441],[954,713],[999,750],[1207,482],[1101,396],[893,667],[938,704],[1148,434],[1222,487],[1016,756],[1060,796],[1267,528]], np.int32)
    pts=matrice
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img, [pts], (255,255,255))
    cv2.polylines(img,[pts],True,(255,255,255))

    if input_type == 'image':
        cv2.imwrite(output,img)
    else:
        return img

if __name__=="__main__":
    input='./14.png'
    output='./14.png'
    l1=img_adjust_tous_points.region_cal(input)
    draw_polygon(input,output,l1)
