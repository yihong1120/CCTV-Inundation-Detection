import cv2
import numpy as np
import imutils

def Get_Outline(image):
    """
    Finds the outline of a document in the input image.
    """
    # Blur and detect edges
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return image, edged

def Get_cnt(edged):
    """
    Finds rectangles in the edge map that approximately enclose documents.
    """
    # Find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    
    # Process contours to find rectangles
    rectangles = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rectangles.append(approx)
    return rectangles

def region_cal(input, input_type='image'):
    """
    Detects documents in the input image or edge map.
    """
    if input_type == 'image':
        image = cv2.imread(input)
    else:
        image = input
    image, edged = Get_Outline(image)
    rectangles = Get_cnt(edged)
    return rectangles

if __name__ == "__main__":
    input = "./14_0.png"
    region_cal(input)
