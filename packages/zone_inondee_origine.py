import sys
import os
sys.path.append(os.path.join(os.getcwd()))
#import image_frame
#import img_enhance
import inrange_blanc
import img_adjust_tous_points
import del_hort
import draw_polygon
import convex_hull
#import centre_point
#import outline_sim
#import points_link
#import distance_courte
import edge
import del_vert
import pixel_colour
import shutil
import cv2
import imutils
import numpy as np
pic=['jpg','png']

def del_frame_ligne(image):
    image_out=image
    image=cv2.imread(image)
    w, h=image.shape[:2]
    #image[x,y]=(0,0,0)

    for i in range(w):
        for j in range(h):
            #橫跑
            if i == 0:
                for x in range(11):
                    x=i+x
                    image[x,j]=(0,0,0)
            elif i==(w-1):
                for x in range(11):
                    x=i-x
                    image[x,j]=(0,0,0)
            #直跑
            elif j == 0:
                for y in range(11):
                    y=j+y
                    image[i,y]=(0,0,0)
            elif j==(h-1):
                for y in range(11):
                    y=j-y
                    image[i,y]=(0,0,0)
    cv2.imwrite(image_out,image)

def fill_edge(im_in, input_type = 'image'):
    #route
    if input_type == 'image':
        im_in=cv2.imread(im_in)
    #opencv type
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    image=img = np.zeros([h, w, 3], np.uint8)
    #image=np.zeros([h, w], np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    '''
    # Combine the two images to get the foreground.
    im_out = image | im_floodfill_inv
    '''
    
    #im_out=np.zeros([h, w, 3], np.uint8)

    if input_type == 'image':
        cv2.imwrite('./testest.png',im_out)
    else:
        return im_floodfill_inv

def white_area_size(image):
    i = 0
    for a in range(image.shape[0]):
        for b in range(image.shape[1]):
            if image[a,b].all() > 0:
                i= i + 1
    return i

def max_minAreaRect(image, input_type = 'image'):
    if input_type == 'image':
        image = cv2.imread(image)
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #_, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area=0
    max_length=0 #rect[1][0]
    max_width=0 #rect[1][1]
    for cont in contours:
        # 对每个轮廓点求最小外接矩形
        rect = cv2.minAreaRect(cont)
        # cv2.boxPoints可以将轮廓点转换为四个角点坐标
        if max_area==0:
            max_length=rect[1][0]
            max_width=rect[1][1]
            max_area=max_length*max_width
        elif rect[1][0]*rect[1][1]>max_area:
            max_length=rect[1][0]
            max_width=rect[1][1]
            max_area=max_length*max_width
        #print('area:',area)
        #print(rect[1][0])
        #print(rect[1][1])
        #print(rect[1][0]*rect[1][1])
        
    return max_length,max_width

def compute_size_minAreaRect(matrix):
    rect = cv2.minAreaRect(matrix) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    #box = cv2.cv.BoxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    #box = np.int0(box)

    return rect[1][0], rect[1][1]
    
def compute_area(h,w,matrix):
    
    #im = np.zeros((10000,10000), dtype="uint8")  # 获取图像的维度: (h,w)=iamge.shape[:2]
    #####polygon_mask = cv2.fillPoly(im, matrix, 255)
    
    #img = np.zeros((h, w, 3), np.uint8)
    img = np.ones((h,w),dtype=np.uint8)
    #triangle = np.array([[0, 0], [1500, 800], [500, 400]])
    #triangle=matrix
    polygon_mask=cv2.fillConvexPoly(img, triangle, (255, 255, 255))
    
    area = np.sum(np.greater(polygon_mask, 0))
    cv2.imwrite('./area_test.png',polygon_mask)

def crosswalk_count(input, commencer, input_type = 'image'):
    if input_type == 'image':
        output=input
        threshold=cv2.imread(input)
    else:
        #input 轉 opencv
        threshold = cv2.cvtColor(np.asarray(input), cv2.COLOR_RGB2BGR)
        
    #inrange_blanc.inrange_blanc(input,output)
    ret,threshold=cv2.threshold(threshold, 127, 255, cv2.THRESH_BINARY)
    #cv2.imwrite(output,threshold)
    #cv2.imwrite('./threshodl_crosswalk.png',threshold)

    threshold = edge.edge(threshold, input_type = 'opencv_image') ##改輸入
    
    '''
    up,down,gauche,droit=pixel_colour.ligne_de_touche(input)
    
    if up==1 or down==1:
        #del_horizontal
        del_hort.del_hort(input,output) #gray
    elif gauche==1 or droit==1:
        #del_vertical
        del_vert.del_vert(input,output)
    '''
    
    #####del_frame_ligne(output)
    #########cv2.imwrite('./0_0_0_0.png',threshold)
    test_img=fill_edge(threshold, input_type = 'opencv_image') ##改輸入
    #####cv2.imwrite('./0_0_0.png',test_img)
    
    #output1=cv2.imread(input)
    #output1=test_img
    max_length,max_width=max_minAreaRect(test_img, input_type = 'opencv_image')
    
    crosswalk_area=white_area_size(test_img)
    nombres_crosswalk=round(crosswalk_area/(max_length*max_width))
    
    test_img_size = test_img.shape
    #cv2.imwrite('./0_0.png', output1)
    
    l1,nombre=img_adjust_tous_points.region_cal(test_img, input_type = 'opencv_image') #L1 ##改輸入->
    #nombres_crosswalk=nombre
    if nombre!=0:
        test_img = draw_polygon.draw_polygon(test_img, l1, input_type = 'opencv_image') #l1 img ##改輸入
        cv2.imwrite('./'+commencer+'/draw_polygon.png',test_img)
        #convex_hull
        test_img = convex_hull.convex_hull('./'+commencer+'/draw_polygon.png', input_type = 'image', output ='./'+commencer+'/draw_polygon.png') #1 img ##改輸入
        #cv2.imwrite('./convex_hulln.png',test_img)
    #tous_les_points
    
    l1,nombre=img_adjust_tous_points.region_cal('./'+commencer+'/draw_polygon.png', input_type = 'image') #L1 ##改輸入->
    if nombre!=1:
        nombre=0
    arr = np.array(l1)
    #compute_area(output1[0],output1[1],arr)
    #print(output1[0],output1[1])
    
    pro_max_length, pro_max_width = compute_size_minAreaRect(l1)
    rate_noir=(pro_max_length-nombres_crosswalk*max_width)/(nombres_crosswalk-1)/max_width
    
    vrai_length_crosswalk=40*max_length/max_width

    totale_length=nombres_crosswalk*40+(nombres_crosswalk-1)*40*rate_noir
    totale_width=vrai_length_crosswalk
    
    #os.remove('./Mask_RCNN/crosswalk.png')
    
    return l1, nombre, totale_length, totale_width

def min_in_file(router):
    route = os.listdir(os.path.join(os.getcwd(),router))
    liste=[] #mots.split(".")[1]
    for time2 in route:
        if time2.split(".")[-1] in pic:
            liste.append(int(time2.split(".")[0]))
    min1=min(liste)
    return min1
    
def max_in_file(router):
    route = os.listdir(os.path.join(os.getcwd(),router))
    liste=[] #mots.split(".")[1]
    for time2 in route:
        if time2.split(".")[-1] in pic:
            liste.append(int(time2.split(".")[0]))
    max1=max(liste)
    return max1

def min_fichier(fichief_nom):
    route=os.path.join(os.getcwd(),fichief_nom,str(min_in_file(fichief_nom))+".png")
    return route
    
def max_fichier(fichief_nom):
    route=os.path.join(os.getcwd(),fichief_nom,str(max_in_file(fichief_nom))+".png")
    return route
    
def crosswalk_output():

    '''
    nombre=0
    while nombre !=1 and os.listdir('./crosswalk'):
        l1, nombre, totale_length, totale_width = crosswalk_count(min_fichier('crosswalk'))
        if nombre==1:
            le_fichier=min_in_file('crosswalk')
            #print(le_fichier)
            #清空
            while os.listdir('./crosswalk'):
                os.remove(min_fichier('crosswalk'))
        else:
            os.remove(min_fichier('crosswalk'))
    '''
    #l1, nombre, totale_length, totale_width = crosswalk_count(min_fichier('crosswalk'))
    l1, nombre, totale_length, totale_width = crosswalk_count('./Mask_RCNN/crosswalk.png')
    return l1, nombre, totale_length, totale_width
    
if __name__=="__main__":
    print("c'est parti")
