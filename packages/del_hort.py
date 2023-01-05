import cv2
import numpy as np

def del_hort(input,output):
    img = cv2.imread(input)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    linek = np.zeros((11,11),dtype=np.uint8)
    linek[5,...]=2
    x=cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek ,iterations=2)
    gray-=x
    cv2.imwrite(output,gray)
    return gray

if __name__=="__main__":
    input='./14_0.png'
    output='./14_0.png'
    del_hort(input,output)
