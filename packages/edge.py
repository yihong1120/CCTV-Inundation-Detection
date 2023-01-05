import cv2
def edge(img, output = None, input_type = 'image'):
    if input_type == 'image':
        img = cv2.imread(img)
    #img = cv2.medianBlur(img,5)
    img = cv2.Canny(img,100,200)
    
    if input_type == 'image':
        cv2.imwrite(output,img)
    else:
        return img
    
if __name__=="__main__":
    input='./14.png'
    output='./14.png'
    edge(input,output)
