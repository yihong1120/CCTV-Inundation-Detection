import gc
import glob
import os
import shutil


def min_in_file(router):
    minium=None
    for file in glob.iglob(os.path.join(os.getcwd(),router,'*.png')):
        file=os.path.split(file)[1]
        if file.split(".")[0].isdigit():
            file=int(file.split(".")[0])
            if minium==None:
                minium=file
            elif minium>file:
                minium=file
    return minium

def max_in_file(router):
    maxium=None
    for file in glob.iglob(os.path.join(os.getcwd(),router,'*.png')):
        file=os.path.split(file)[1]
        if file.split(".")[0].isdigit():
            file=int(file.split(".")[0])
            if maxium==None:
                maxium=file
            elif maxium<file:
                maxium=file
    return maxium

def min_fichier(fichief_nom):
    return os.path.join(os.getcwd(),fichief_nom,str(min_in_file(fichief_nom))+".png")

def max_fichier(fichief_nom):
    return os.path.join(os.getcwd(),fichief_nom,str(max_in_file(fichief_nom))+".png")

def prendre_des_photos_CCTV(commener):
    min=str(min_in_file(commener))
    shutil.move(os.path.join(os.getcwd(),commener,min+".png"), os.path.join(os.getcwd(),"timestamps",min+".png"))
    del min
    gc.collect()
