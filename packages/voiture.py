from tensorflow.compat.v1 import reset_default_graph,get_default_graph,disable_eager_execution,get_default_session
import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import time
import gc
import glob

sys.path.append(os.getcwd())  # To find local version of the library
from mrcnn.config_mrcnn import Config
from mrcnn import model_mrcnn as modellib
from mix_image import mix_images

pic=['jpg','png']
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
    del router, minium
    
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
    del router, maxium

def min_fichier(fichief_nom):
    return os.path.join(os.getcwd(),fichief_nom,str(min_in_file(fichief_nom))+".png")
    
def max_fichier(fichief_nom):
    return os.path.join(os.getcwd(),fichief_nom,str(max_in_file(fichief_nom))+".png")

class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 6  # background + 1 class
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

COCO_MODEL_PATH = os.path.join(os.getcwd(), 'Mask_RCNN', 'cars_water.h5')
MODEL_DIR = os.path.join(os.getcwd(),'Mask_RCNN','logs')
config = ShapesConfig()
voiture_mrcnn_model = modellib.MaskRCNN(mode="inference",  model_dir=MODEL_DIR,config=config,)
voiture_mrcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)
voiture_mrcnn_graph = get_default_graph()
voiture_mrcnn_class_names = ['BG', 'car','bus','truck','motorcycle','person','water']

def mrcnn(input,output, try_erreur,commener): #->0;commener
    reset_default_graph()
    
    def apply_mask(image, mask):
        image[:, :, 0] = np.where(
            mask == 0,
            125,
            image[:, :, 0]
        )
        image[:, :, 1] = np.where(
            mask == 0,
            12,
            image[:, :, 1]
        )
        image[:, :, 2] = np.where(
            mask == 0,
            15,
            image[:, :, 2]
        )
        return image
    
    # This function is used to show the object detection result in original image.
    def display_instances(image_display_instances, boxes, masks, ids, names, scores, type_travailler):
        mask=[]
        if type_travailler==0:
            objs=["car"]
        elif type_travailler==1:
            objs=["car","motorcycle","truck","bus","person"]
        elif type_travailler==2:
            objs=["water"]
        elif type_travailler==3:
            objs=["crosswalk"]
        rang=0
        area_org=0
        for i in range(len(r['rois'])):
            if names[ids[i]] in objs:
                if type_travailler==0:
                #label = names[ids[i]]
                    mask = masks[:, :, i]
                    image_display_instances[mask] = 255
                    image_display_instances[~mask] = 0
                    unique, counts = np.unique(image_display_instances, return_counts=True)
                    try:
                        mask_area = counts[1] / (counts[0] + counts[1])
                    except:
                        mask_area = 0
                    
                    if area_org==0:
                        rang=i
                        area_org=mask_area
                        mask = masks[:, :, i]
                    elif mask_area>area_org:
                        area_org=mask_area
                        rang=i
                        mask = masks[:, :, i]
                    
                if type_travailler==1:
                    if mask==[]:
                        mask=masks[:, :, i]
                    else:
                        mask=mask + masks[:, :, i]

        if type_travailler==0:
            mask=masks[:, :, rang]
            
        image2 = apply_mask(image_display_instances, mask)
        image = image2
        return image
        del image, mask, type_travailler,objs

    def transparent_back(image):
        image = image.convert('RGBA')
        L,H = image.size
        color_0 = image.getpixel((0, 0))
        for h in range(H):
            for l in range(L):
                dot = (l, h)
                color_1 = image.getpixel(dot)
                if color_1 == color_0:
                    color_1 = color_1[:-1] + (0,)
                    image.putpixel(dot, color_1)

        return image

    image_CV2 = cv2.imread(input) #input
    img_terminal = Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGB)) #input image pillow
    img_terminal_voitures=Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGBA))
    height, width, channels = image_CV2.shape
    # Run detection
    with voiture_mrcnn_graph.as_default():
        results = voiture_mrcnn_model.detect([image_CV2], verbose=0)

    # Visualize results
    r = results[0]
    original_backup_img = Image.open(os.path.join(os.getcwd(),commener,"background.png"))
    if not len(r['rois']):
        original_backup_img_width = original_backup_img.size[0]#长度
        original_backup_img_height = original_backup_img.size[1]#宽度
        
        # 输入文件
        img = Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGB)) #input

        width = img.size[0]#长度
        height = img.size[1]#宽度
        img = img.convert("RGBA")
        
        if original_backup_img_width != width or original_backup_img_height != height:
            img.save(output)
        else: #尺寸吻合
            img.save(output)
            img.save(os.path.join(os.getcwd(),commener,"background.png"))#改成mix
        img.close()
        
    else:
        print('there are ',len(r['rois']),' cars in this image.')
        if try_erreur==0:
            for i in range(2):
                image_CV2=cv2.imread(input)
                frame = display_instances(image_CV2, r['rois'], r['masks'], r['class_ids'], voiture_mrcnn_class_names, r['scores'],i)
                frame=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
                image = transparent_back(frame)
                    # 输入文件
                width = image.size[0]#长度
                height = image.size[1]#宽度
                for ip in range(0,width):#遍历所有长度的点
                    for jp in range(0,height):#遍历所有宽度的点
                        data = (image.getpixel((ip,jp)))#打印该图片的所有
                        if (data[3]!=0):#RGBA的r值大于0，并且g值大于0,并且b值大于0
                            #一堆車
                            if i==1:
                                img_terminal_voitures.putpixel((ip,jp),(0,0,0,0))
                            #面積最大的車
                            if i==0:
                                frame.putpixel((ip,jp),(0,0,0,0))
                                
                if i==1:
                    input_file = os.path.split(input)[1]
                    image = mix_images(original_backup_img, img_terminal_voitures, input_type='pillow_img')
                    #image.save(os.path.join(os.getcwd(), 'non_noise', input_file))
                    image.save(output)
                    image.save(os.path.join(os.getcwd(),commener,"background.png"))
                    
                if i==0:
                    
                    unique_voiture = mix_images(img_terminal,frame, input_type='pillow_img')
                    unique_voiture=transparent_back(unique_voiture) #去背混出
                    unique_voiture.save('./Mask_RCNN/unique_voiture.png') #存去背
    
    os.remove(min_fichier(commener))
    get_default_graph().finalize()
