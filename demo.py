# imports
import os
import time
import sys
import shutil
import cv2
import gc
import glob

sys.path.append(os.path.join(os.getcwd(),"src"))
from image_processing import min_in_file, max_in_file, min_fichier, max_fichier, prendre_des_photos_CCTV
from model_inference import inundation_depth
from database_operations import create_database, store_data
import classify_rain
import classify_inundation
import voiture
import mix_image
import couleur_transparent
import water
import ground
import crosswalk
import zone_inondee

pic=['jpg','png']
            1: Calculate depth of inundation.
        front_view (numpy array, optional): The front view of the object. Required if mesh2obj_dec is 1.
        ratio_height (float, optional): The ratio of height of the object. Required if mesh2obj_dec is 1.
        
    Returns:
        If mesh2obj_dec is 0: tuple
            front_view (numpy array): The front view of the object.
            ratio_height (float): The ratio of height of the object.
        If mesh2obj_dec is 1: float
            inundation_depth: The depth of inundation.
    """
    pixel2mesh.pixel2obj()
    if mesh2obj_dec==0:
        front_view,ratio_height=mesh2depth.obj2height(0)
        return front_view,ratio_height
    elif mesh2obj_dec==1:
        inundation_depth=mesh2depth.obj2height(1,front_view,ratio_height)
        return inundation_depth
    
if __name__ == "__main__":
    # list of answers
    ans=["y","n"]
    
    # get user input
    commener=str(input("ficher:"))
    date_de_la_photo=input("Would you like to print time on images[y/n]?")
    inundation_area_calculation=str(input("calculate the inondation region[y/n]?"))
    database_usage=str(input("store data in database[y/n]?"))
    
    # create database if necessary
    if database_usage==ans[0]:
        database_exist=input("if database is exist[y/n]?")
        if database_exist==ans[1]:
            ville=str(input("le name de la ville est:"))
            database_name=ville+"_"+commener
            database.create_database(database_name)
    else:
        database_name=None
    
    # main loop
    while True:
        time_start=time.time()
        prendre_des_photos_CCTV()
        
        # classify rain
        rain_class=classify_rain.classify(min_fichier(commener))
        
        # calculate inundation depth if necessary
        if inundation_area_calculation==ans[0]:
            front_view,ratio_height=inundation_depth(0)
            inundation_depth_val=inundation_depth(1,front_view,ratio_height)
        else:
            front_view=None
            ratio_height=None
            inundation_depth_val=None
        
        # classify inundation
        inundation_class=classify_inundation.classify(max_fichier(commener),front_view,ratio_height)
        
        # detect cars
        car_detection=voiture.detection(min_fichier(commener))
        
        # create composite image
        mix_image.mix(min_fichier(commener),max_fichier(commener))
        
        # make image transparent
        couleur_transparent.transparent(min_fichier(commener))
        
        # detect water
        water_detection=water.detection(min_fichier(commener))
        
        # detect ground
        ground_detection=ground.detection(min_fichier(commener))
        
        # detect crosswalk
        crosswalk_detection=crosswalk.detection(min_fichier(commener))
        
        # detect inundated region
        inundated_region=zone_inondee.detection(max_fichier(commener),front_view,ratio_height)
        
        # store data in database if necessary
        if database_name!=None:
            database.store_data(database_name,time_start,rain_class,inundation_class,car_detection,water_detection,ground_detection,crosswalk_detection,inundated_region,inundation_depth_val)

        # print time on images if necessary
        if date_de_la_photo==ans[0]:
            img=cv2.imread(min_fichier(commener))
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText=(10,500)
            fontScale=1
            fontColor=(255,255,255)
            lineType=2
            cv2.putText(img,time.asctime(time.localtime(time.time())),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            cv2.imwrite(min_fichier(commener),img)
            del img
            gc.collect()

        # wait for next iteration
        time.sleep(60-(time.time()-time_start))