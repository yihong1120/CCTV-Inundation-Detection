from tensorflow.compat.v1 import reset_default_graph,get_default_graph,disable_eager_execution
import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import time
sys.path.append(os.path.join(os.getcwd(),'Mask_RCNN'))  # To find local version of the library
import img_adjust
from mrcnn.config_mrcnn import Config
from mrcnn import model_mrcnn as modellib
from mix_image import mix_images
import couleur_transparent

disable_eager_execution()
pic=['jpg','png']

# Directory to save logs and trained models
MODEL_DIR = os.path.join(os.path.join(os.getcwd(),'Mask_RCNN','logs'))

iter_num = 0

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 1 class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
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
    route=os.path.join(os.getcwd(),fichief_nom,str(min_in_file(fichief_nom))+".png")
    return route
    
# Load weights trained and models
COCO_MODEL_PATH = os.path.join(os.getcwd(), 'Mask_RCNN', 'cars_water.h5')
MODEL_DIR = os.path.join(os.getcwd(),'Mask_RCNN','logs')
config = ShapesConfig()
mrcnn_model = modellib.MaskRCNN(mode="inference",  model_dir=MODEL_DIR,config=config,)
mrcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)
mrcnn_graph = get_default_graph()

#classs names
mrcnn_class_names = ['BG', 'car','bus','truck','motorcycle','person','water']
    
def predict(input, output, date_de_la_photo, points_crosswalks, nombre, totale_length, totale_width, inundation_area_calculation, ground = None):

    # Local path to trained weights file
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
    def display_instances(image_display_instances, boxes, masks, ids, names, scores):
        # max_area will save the largest object for all the detection results
        max_area = 0
        
        # n_instances saves the amount of all objects
        n_instances = boxes.shape[0]

        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i in range(n_instances):
            if not np.any(boxes[i]):
                continue

            #把辨識出的淹水區域弄成一個陣列
            mask=[]
            objs=["water"]
            for i in range(len(r['rois'])):
                #label = names[ids[i]]
                if names[ids[i]] in objs:
                    if mask==[]:
                        mask=masks[:, :, i]
                    else:
                        mask=mask + masks[:, :, i]
            else:
                #print(len(r['rois']))
                break

        # by mistake you put apply_mask inside for loop or you can write continue in if also
        image2 = apply_mask(image_display_instances, mask)
        image = image2
        return image

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
        
    #load original images
    reset_default_graph()
    image_CV2 = cv2.imread(input)
    image_original_PIL = image_terminal = Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGB))
    img_area = image_CV2.copy()
    height, width, channels = image_CV2.shape
    
    # Run detection
    with mrcnn_graph.as_default():
        results = mrcnn_model.detect([image_CV2], verbose=0)

    # Visualize results
    r = results[0]
    
    #如果沒有辨識到物件
    if not len(r['rois']):
        label = 'the inundation area is not computed.'
        print("il n'y a pas l'eau dans cette image")
        if date_de_la_photo=="y":
            cv2.putText(img_area, label, (10, 100),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2) #img_area改成mix完的
        else:
            cv2.putText(img_area, label, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2) #img_area改成mix完的
        Inondation=0
        return Inondation
    
    #如果有辨識到物件
    else:
        #frame是辨識物件
        frame = display_instances(
                     image_CV2, r['rois'], r['masks'], r['class_ids'], mrcnn_class_names, r['scores']
                )
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transparent_back(frame)
        
        #如果沒有辨識到物件
        width = image.size[0]#长度
        height = image.size[1]#宽度
        for ip in range(0,width):#遍历所有长度的点
            for jp in range(0,height):#遍历所有宽度的点
                data = (image.getpixel((ip,jp)))#打印该图片的所有
                if data[3]!=0:#RGBA的r值大于0，并且g值大于0,并且b值大于0
                    image.putpixel((ip,jp),(0,255,255,127))#则这些像素点的颜色改成大红色#0,0,0
        
        #把道路辨識一下
        if ground != None:
            ground_img = ground
            image = mix_images(image,ground_img,input_type='pillow_image')
        image = couleur_transparent.transparent_back(0,0,0,255, image, input_type = 'pillow_image')
        
        #如果要計算道路面積
        if inundation_area_calculation == "y":
            try:
                #圖片辨識區域用矩陣轉換成面積
                Inondation=img_adjust.rotate_img(image,points_crosswalks,totale_length, totale_width,input_type='pillow_image')
                area_image_generate = image
                #把辨識出的淹水面積黏到原圖上
                area_image_generate = mix_images(image_original_PIL, area_image_generate, input_type = 'pillow_image')
                #image.pillow tp opencv
                area_image_generate = cv2.cvtColor(np.asarray(area_image_generate), cv2.COLOR_RGB2BGR)
                
                #把計算數據黏到原圖上
                label = "{}: {:.2f} metres carres".format("Inundation Area", Inondation)
                if date_de_la_photo=="y":
                    cv2.putText(area_image_generate, label, (10, 100),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2) #img_area改成mix完的
                else:
                    cv2.putText(area_image_generate, label, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2) #img_area改成mix完的
                print(label)
                cv2.imwrite(output, area_image_generate) #img_area
        
            except:
                #如果沒有辨識到淹水面積
                Inondation=0
                for ip in range(0,width):#遍历所有长度的点
                    for jp in range(0,height):#遍历所有宽度的点
                        data = (image.getpixel((ip,jp)))#打印该图片的所有
                        if (data[0]==0 and data[1]==255 and data[2]==255):#RGBA的r值大于0，并且g值大于0,并且b值大于0
                            Inondation+=1
                
                #把辨識出的淹水區域黏到原圖上
                image = mix_images(image_original_PIL, image, input_type = 'pillow_image')
                #image.pillow tp opencv
                image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                
                #把計算數據黏到原圖上
                Inondation = round(Inondation/(width*height), 2)
                label = "{}: {:.2f} % of images is inundation".format("Inundation Portion", Inondation)
                
                if date_de_la_photo=="y":
                    cv2.putText(image, label, (10, 100),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2) #img_area改成mix完的
                else:
                    cv2.putText(image, label, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2) #img_area改成mix完的
                print(label)
                cv2.imwrite(output, image) #img_area
        else:
            #把辨識出的淹水區域黏到原圖上
            image = mix_images(image_original_PIL, image, input_type = 'pillow_image')
            #image.pillow tp opencv
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            Inondation=0
            
            #把計算數據黏到原圖上
            label = 'the inundation area is not computed.'
            print(label)
            
            if date_de_la_photo=="y":
                cv2.putText(image, label, (10, 100),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2) #img_area改成mix完的
            else:
                cv2.putText(image, label, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2) #img_area改成mix完的
            
        return Inondation
        
    get_default_graph().finalize()


