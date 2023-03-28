import os
import sys
import numpy as np
import cv2
from PIL import Image
from tensorflow.compat.v1 import get_default_graph

sys.path.append(os.getcwd())
from mrcnn.config_mrcnn import Config
from mrcnn import model_mrcnn as modellib

class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

COCO_MODEL_PATH = os.path.join(os.getcwd(), 'Mask_RCNN', 'crosswalk.h5')
MODEL_DIR = os.path.join(os.getcwd(),'Mask_RCNN','logs')

config = ShapesConfig()

model = modellib.MaskRCNN(mode="inference",  model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

graph = get_default_graph()
class_names = ['BG', 'crosswalk']

def apply_mask(image, mask):
    image[:, :, 0] = np.where(mask == 0, 125, image[:, :, 0])
    image[:, :, 1] = np.where(mask == 0, 12, image[:, :, 1])
    image[:, :, 2] = np.where(mask == 0, 15, image[:, :, 2])

def display_instances(image, boxes, masks, ids, names, scores):
    max_area = 0
    for i in range(len(boxes)):
        if names[ids[i]] == 'crosswalk':
            mask = masks[:, :, i]
            mask_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
            if mask_area > max_area:
                max_area = mask_area
                box = boxes[i]
                mask = mask.astype(np.uint8)
                mask_image = apply_mask(image, mask)
                image = Image.fromarray(mask_image.astype(np.uint8))
                image, _, _ = model.detect_with_image(image, verbose=0)
                image = np.asarray(image)
    return image

def mrcnn(input_image):
    results = model.detect([input_image], verbose=1)
    r = results[0]
    image = display_instances(input_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return image

if __name__ == '__main__':
    test_image = cv2.imread('test_image.jpg')
    output_image = mrcnn(test_image)
    cv2.imshow('output', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
