import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import time
from tensorflow.compat.v1 import reset_default_graph, get_default_graph

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

mrcnn_model_crosswalk = modellib.MaskRCNN(mode="inference",  model_dir=MODEL_DIR, config=config)
mrcnn_model_crosswalk.load_weights(COCO_MODEL_PATH, by_name=True)

mrcnn_graph_crosswalk = get_default_graph()
mrcnn_class_names = ['BG', 'crosswalk']
    
def mrcnn(input_image):
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
    # This function is used to show the object detection result in the original image.
    def display_instances(image, boxes, masks, ids, names, scores):
        # max_area will save the largest object for all the detection results
        max_area = 0

        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        
        for i in range(len(boxes)):
            if names[ids[i]] == 'crosswalk':
                mask = masks[:, :, i]
                unique, counts = np.unique(mask, return_counts=True)
                mask_area = counts[1] / (counts[0] + counts[1])
                
                if mask_area > max_area:
                    max_area = mask_area
                    box = boxes[i]
                    mask = mask.astype(np.uint8)
                    mask_image = apply_mask(image, mask)
                    image = Image.fromarray(mask_image.astype(np.uint8))
                    image, _, _ = mrcnn_model_crosswalk.detect_with_image(image, verbose=0)
                    image = np.asarray(image)

        return image

    results = mrcnn_model_crosswalk.detect([input_image], verbose=1)
    r = results[0]
    image = display_instances(input_image, r['rois'], r['masks'], r['class_ids'], mrcnn_class_names, r['scores'])

    return image
    
if __name__ == '__main__':
    # Load an image to test object detection
    test_image = cv2.imread('test_image.jpg')
    output_image = mrcnn(test_image)

    # Display the output image
    cv2.imshow('output', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
