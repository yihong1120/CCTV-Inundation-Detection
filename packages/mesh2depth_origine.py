import os
import numpy as np
import cv2

def obj2height(cal_ration_height,front_view_donner=None,ratio_height_donner=None):
    objFilePath='./Mask_RCNN/unique_voiture.obj'
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
    # points原本为列表，需要转变为矩阵，方便处理
    points = np.array(points)

    if points.mean()<1:
        points=points*500

    elif points.mean()<10:
        points=points*50
    elif points.mean()<100:
        points=points*5

    if points.min()<0:
        points=points+abs(points.min())+10

    size=[]
    points=np.around(points,decimals=0)
    xyz = np.array([[0,1],[1,2],[2,0]])

    for i in range(3):
        points0 = np.hsplit(points, 3) [xyz[i][0]]
        points1 = np.hsplit(points, 3) [xyz[i][1]]
        points3=np.append(points0,points1,axis=1)
        points3=points3.astype(int)
        points3 = points3.reshape((-1,1,2))
        
        rect = cv2.minAreaRect(points3)
        (x, y), (width, height), angle = rect
        petit_size=[width,height]
        size.append(petit_size)
        
    size = np.array(size, dtype = int)

    area_side=[]
    for i in range(3):
        area=size[i][0]*size[i][1]
        area_side.append(area)
        
    area_side = np.array(area_side, dtype = int)
    size = np.delete(size, np.argmax(area_side, axis=0), 0)
    area_side = np.delete(area_side, np.argmax(area_side, axis=0), 0)
    side_view=np.argmax(area_side, axis=0)
    front_view=np.argmin(area_side, axis=0)
    side_view=size[side_view]
    front_view=size[front_view]

    if side_view.min()>front_view.min():
        front_view=front_view*side_view.min()/front_view.min()
        front_view = np.array(front_view, dtype = int)
    elif side_view.min()<front_view.min():
        side_view=side_view*front_view.min()/side_view.min()
        side_view = np.array(side_view, dtype = int)
    
    ratio=front_view.max()/front_view.min()
    Real_height=1415#cm
    ratio_height=Real_height/front_view.min()
    Real_width=1745#cm
    ratio_width=Real_width/front_view.max()
    Real_length=4430#cm
    ration_length=Real_length/side_view.max()
    
    if cal_ration_height==0:
        return front_view,ratio_height
    
    elif cal_ration_height==1:
        front_view=front_view*front_view_donner.max()/front_view.max()
        inundation_depth=Real_height-front_view.min()*ratio_height_donner
        inundation_depth=round(inundation_depth,2)
        return inundation_depth
        del inundation_depth
    del xyz, petit_size, Real_height, ratio_height, size, front_view, side_view, Real_length, Real_width, area_side,ratio_height_donner, front_view_donner, cal_ration_height
    os.remove(objFilePath)
    
if __name__=="__main__":
    obj2height('./voiture5.obj',0)
