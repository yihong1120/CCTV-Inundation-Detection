import cv2
import numpy as np

def inrange_blanc(input,output):
    # read image and apply median blur to remove noise
    img = cv2.imread(input)
    img = cv2.medianBlur(img, 5)

    # convert image to hsv color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # set range for white color in hsv color space
    lower_red = np.array([0,0,221])
    upper_red = np.array([180, 30, 255])

    # create mask
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # write mask to output file if specified
    if output != 1:
        cv2.imwrite(output,mask)

    return mask

if __name__=="__main__":
    input='./14.png'
    output='./14.png'
    inrange_blanc(input,output)
