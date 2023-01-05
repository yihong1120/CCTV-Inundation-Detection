from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np

def Get_Outline(image):
    #image = cv2.imread(input_dir)
    #image = np.uint8(image)
    ###image = np.array(image, dtype=np.uint8)
    ###image = (image*255).astype(np.uint8)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (5,5),0)
    edged = cv2.Canny(blurred,75,200)
    return image,edged

def Get_cnt(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if  imutils.is_cv3()  else   cnts[0]
    docCnt =None
    l1=[]
    nombre=0
    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 輪廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 獲取近似的輪廓
            #if len(approx) ==4:                            # 近似輪廓有四個頂點
            #l1=[]
            while len(approx)==4:
                nombre+=1
                #print(len(approx))
                docCnt = approx
                '''
                docCnt = np.reshape(docCnt,(4,1,2))
                docCnt = np.reshape(docCnt,(-1,2))
                '''
                #print(docCnt)
                docCnt=docCnt.tolist()
                #print("-----------------")
                #docCnt = np.reshape(docCnt, (2, 4))
                #docCnt = docCnt.flatten()
                #print(docCnt)
                '''
                if nombre==1:
                    l1=docCnt
                else:
                    #print(docCnt)
                    print("-----------------")
                    #l1=np.array([l1,docCnt])
                '''
                l1.append(docCnt)
                
                break
    l1=np.array(l1)
    #print(l1)
    #print(nombre)
    #l1="sample"
    
    l1 = np.reshape(l1,(nombre,4,2))
    l1 = np.reshape(l1,(-1,2))
    #print(l1)
    '''
    print(l1[0][0])
    print(l1[0][1])
    print(max(l1[::3]))
    '''
    return l1,nombre

def region_cal(input, input_type = 'image'):
    if input_type == 'image':
        image = cv2.imread(input)
    else:
        image = input
    image,edged = Get_Outline(image)
    l1 = Get_cnt(edged)
    #print(l1)
    #print(docCnt)
    return l1
    
if __name__=="__main__":

    input = "./14_0.png"
    #region_cal(input_dir)
    region_cal(input)
    #region_cal("./detelet_vh.png")
#convex_hull
