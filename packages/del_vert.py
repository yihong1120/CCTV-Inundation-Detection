import cv2
import numpy as np

def del_vert(input,output):
    img = cv2.imread(input)
    img=np.rot90(img,1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    linek = np.zeros((11,11),dtype=np.uint8)
    linek[5,...]=2
    x=cv2.morphologyEx(img, cv2.MORPH_OPEN, linek ,iterations=2)
    img-=x
    img=np.rot90(img,-1)
    if output!=1:
        cv2.imwrite(output,img)
    return img

if __name__=="__main__":
    input='./14_0.png'
    output='./14_0.png'
    del_vert(input,output)
