import os
import sys
import time
import numpy as np
import cv2
#import plaidml
from PIL import Image
import time
from tensorflow.compat.v1 import reset_default_graph,get_default_graph
sys.path.append(os.getcwd())  # To find local version of the library
from mrcnn.config_mrcnn import Config
from mrcnn import model_mrcnn as modellib

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
    NUM_CLASSES = 1 + 1  # background + 1 class

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
    
COCO_MODEL_PATH = os.path.join(os.getcwd(), 'Mask_RCNN', 'crosswalk.h5')
MODEL_DIR = os.path.join(os.getcwd(),'Mask_RCNN','logs')
config = ShapesConfig()

global mrcnn_model_crosswalk
mrcnn_model_crosswalk = modellib.MaskRCNN(mode="inference",  model_dir=MODEL_DIR,config=config) #MODEL_DIR=logs
# Load weights trained on ground
mrcnn_model_crosswalk.load_weights(COCO_MODEL_PATH, by_name=True)
global mrcnn_graph_crosswalk
mrcnn_graph_crosswalk = get_default_graph()
mrcnn_class_names = ['BG', 'crosswalk']
    
def mrcnn(input): #output
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
    def display_instances(image_display_instances, boxes, masks, ids, names, scores,i):
        # max_area will save the largest object for all the detection results
        max_area = 0

        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        
        rang=0
        area_org=0
        objs=["crosswalk"]
        for i in range(len(r['rois'])):
            if names[ids[i]] in objs:
                mask = masks[:, :, i]
                '''
                image_display_instances[mask] = 255
                image_display_instances[~mask] = 0
                '''
                unique, counts = np.unique(image_display_instances, return_counts=True)
                mask_area = counts[1] / (counts[0] + counts[1])
                
                if area_org==0:
                    rang=i
                    area_org=mask_area
                    mask = masks[:, :, i]
                elif mask_area>area_org:
                    area_org=mask_area
                    rang=i
                    mask = masks[:, :, i]
        #####print(i)
        
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

    reset_default_graph()
    image_CV2 = cv2.imread(input) #input
    crosswalk_img=image_CV2.copy()
    height, width, channels = image_CV2.shape
    # Run detection
    with mrcnn_graph_crosswalk.as_default():
        results = mrcnn_model_crosswalk.detect([image_CV2], verbose=0)
    # Visualize results
    r = results[0]
   
    if not len(r['rois']):
        # 输入文件

        img = Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGB)) #input
        
        width = img.size[0]#长度
        height = img.size[1]#宽度
        img = img.convert("RGBA")
        for i in range(img.width):
            for j in range(img.height):
                '''
                r,g,b,a = img.getpixel((i,j))
                #if(r !=None  and g != None and b!= None):
                r=0
                g=0
                b=0
                a=0
                '''
                img.putpixel((i,j), (0,0,0,0))
        # 输出图片
        
        #####img.save('./crosswalk/0.png') #mid file
        #os.remove(image)
        #transparent

    else:
        print('crosswalks are detected')
        i=len(r['rois'])
        frame = display_instances(
                     image_CV2, r['rois'], r['masks'], r['class_ids'], mrcnn_class_names, r['scores'],i
                )
        #####cv2.imwrite(os.path.join(os.getcwd(),'mrcnn_bm','water.png'), image_CV2) #output[:-4]+"_pro.png"
        frame=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #####image_PIL = Image.open(os.path.join(os.getcwd(),'mrcnn_bm','water.png')) #output[:-4]+"_pro.png"
        image = transparent_back(frame)
        #image.save(os.path.join(os.getcwd(),'Mask_RCNN','crosswalk.png'))
        return image
        
        '''
        width = image.size[0]#长度
        height = image.size[1]#宽度
        for ip in range(0,width):#遍历所有长度的点
            for jp in range(0,height):#遍历所有宽度的点
                data = (image.getpixel((ip,jp)))#打印该图片的所有
                if data[3]!=0:#RGBA的r值大于0，并且g值大于0,并且b值大于0
                    frame.putpixel((ip,jp),(0,0,0,0))
                else:#RGBA的r值大于0，并且g值大于0,并且b值大于0
                    frame.putpixel((ip,jp),(0,0,0,255))
        '''

if __name__ == "__main__":
    #train_model()
    #file_handle()
    start = time.time()
    in1="./crosswalk_1.png"
    out1="./crosswalk_1_passage_cloute.png"
    mrcnn(in1, "1")
    end = time.time()
    print(end-start)
#做好model指定
#做好影像儲存
