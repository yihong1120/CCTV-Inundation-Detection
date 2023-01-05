from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np

def Get_Outline(image):
    #image = cv2.imread(input_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,75,200)
    return image,gray,edged

def Get_cnt(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    docCnt =None

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 輪廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 獲取近似的輪廓
            if len(approx) ==4:                            # 近似輪廓有四個頂點
                docCnt = approx
                break
    return docCnt

def matrice(docCntList, totale_length, totale_width,height_img,width_img):

    '''
    image = cv2.imread(input_dir)
    image,gray,edged = Get_Outline(image)
    docCnt = Get_cnt(edged)
    result_img = four_point_transform(image, docCnt.reshape(4,2)) # 對原始影象進行四點透視變換
    #####cv2.imwrite("./edge.png",edged)
    docCntList=docCnt.tolist()
    '''
    docCntList=docCntList.tolist()

    x0,y0=docCntList[0][0],docCntList[0][1]
    x1,y1=docCntList[1][0],docCntList[1][1]
    x2,y2=docCntList[2][0],docCntList[2][1]
    x3,y3=docCntList[3][0],docCntList[3][1]
    distance0=(x0**2+y0**2)**0.5
    distance1=(x1**2+y1**2)**0.5
    distance2=(x2**2+y2**2)**0.5
    distance3=(x3**2+y3**2)**0.5
    distanceList=[]
    distanceList.append(distance0)
    distanceList.append(distance1)
    distanceList.append(distance2)
    distanceList.append(distance3)

    if min(distanceList)==distance0:
        sx0,sy0=x0,y0
    elif min(distanceList)==distance1:
        sx0,sy0=x1,y1
    elif min(distanceList)==distance2:
        sx0,sy0=x2,y2
    elif min(distanceList)==distance3:
        sx0,sy0=x3,y3
    #sx0, sy0最靠近原點
    '''
    distance0=((x0-sx0)**2+(y0-sy0)**2)**0.5
    distance1=((x1-sx0)**2+(y1-sy0)**2)**0.5
    distance2=((x2-sx0)**2+(y2-sy0)**2)**0.5
    distance3=((x3-sx0)**2+(y3-sy0)**2)**0.5
    distanceList=[]
    distanceList.append(distance0)
    distanceList.append(distance1)
    distanceList.append(distance2)
    distanceList.append(distance3)
    '''
    distanceList.remove(min(distanceList))
    if min(distanceList)==distance0:
        sx1,sy1=x0,y0
    elif min(distanceList)==distance1:
        sx1,sy1=x1,y1
    elif min(distanceList)==distance2:
        sx1,sy1=x2,y2
    elif min(distanceList)==distance3:
        sx1,sy1=x3,y3
    #sx1, sy1最靠近sx0, sy0
        
    distanceList.remove(min(distanceList))
    if min(distanceList)==distance0:
        sx2,sy2=x0,y0
    elif min(distanceList)==distance1:
        sx2,sy2=x1,y1
    elif min(distanceList)==distance2:
        sx2,sy2=x2,y2
    elif min(distanceList)==distance3:
        sx2,sy2=x3,y3
    #sx2, sy2第二近sx0, sy0
        
    distanceList.remove(min(distanceList))
    if min(distanceList)==distance0:
        sx3,sy3=x0,y0
    elif min(distanceList)==distance1:
        sx3,sy3=x1,y1
    elif min(distanceList)==distance2:
        sx3,sy3=x2,y2
    elif min(distanceList)==distance3:
        sx3,sy3=x3,y3
    #sx3, sy3第三近sx0, sy0

    cX = width_img*2
    cY = height_img*2
    
    totale_length=totale_length/200
    totale_width=totale_width/200
    
    slope=abs((sy0-sy2)/(sx2-sx0))
    '''
    width=(nombres_crosswalks*2-1)*40/200 #totale_length
    length--totale_length/200
    width--totale_width/200
    '''
    if slope<=1:
        rotate=[[cX-totale_length,cY-totale_width],[cX+totale_length,cY-totale_width],[cX-totale_length,cY+totale_width],[cX+totale_length,cY+totale_width]]
    elif slope>1:
        rotate=[[cX-totale_width,cY-totale_length],[cX+totale_width,cY-totale_length],[cX-totale_width,cY+totale_length],[cX+totale_width,cY+totale_length]]
        #rotate=[[cX-150,cY-20],[cX+150,cY-20],[cX-150,cY+20],[cX+150,cY+20]]
    #img = cv2.imread("./05.png")

    #影像透視點轉換變形
    #rows,cols = image.shape[:2]
    #提供至少4個座標點來計算3x3的透視轉換矩陣
    pts1 = np.float32([[sx0,sy0],[sx2,sy2],[sx1,sy1],[sx3,sy3]])
    pts2 = np.float32(rotate)
    
    #計算3x3矩陣轉換,將場內線轉換成水平直線
    M = cv2.getPerspectiveTransform(pts1,pts2)
    #####print(M)
    #透視轉換

    return M

def rotate_img(imager, matric, totale_length, totale_width, input_type="image"):

    #route
    if input_type == "image":
        imager = cv2.imread(imager)
    #pillow to opencv
    else:
        imager = cv2.cvtColor(np.asarray(imager), cv2.COLOR_RGB2BGR)
    
    M=matrice(matric, totale_length, totale_width,imager.shape[0],imager.shape[1])
    imager = cv2.warpPerspective(imager,M,(imager.shape[1]*4,imager.shape[0]*4))
    
    h=imager.shape[0]
    w=imager.shape[1]
    
    inundation=0
    for x in range(w):
        for y in range(h):
            '''
            if imager[y,x][3]!=0:
                inundation+=1
            '''
            '''
            if imager[y,x][0]!=255 or imager[y,x][1]!=255 or imager[y,x][2]!=0:
                imager[y,x]=[0,0,0]
            '''
            
            if imager[y,x][0]==255 and imager[y,x][1]==255 and imager[y,x][2]==0:
                inundation+=1
            
    print('area computing...')
    cv2.imwrite('./Output_imager.png',imager)
    
    '''
    imager = cv2.cvtColor(imager, cv2.COLOR_BGR2HSV) #COLOR_BGR2HSV
    # boundaries for the color red
    boundaries = [([100, 43, 46], [124, 255, 255])]
    #bleu->[([100, 43, 46], [124, 255, 255])]
    for(lower, upper) in boundaries:
        # creates numpy array from boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # finds colors in boundaries a applies a mask
        mask = cv2.inRange(imager, lower, upper)
        output = cv2.bitwise_and(imager, imager, mask = mask)
    '''
    
    # saves the image
    #cv2.imwrite('2'+img_name, output)

    #tot_pixel = output.size
    #####red_pixel = np.count_nonzero(imager)
    #percentage = round(red_pixel * 100 / tot_pixel, 2)
    #print("le region des inondations sont:",red_pixel)
    return inundation

if __name__=="__main__":

    input_dir = "./06.png"
    #region_cal(input_dir)
    rotate_img("./08.png",matrice("./07.png"))
