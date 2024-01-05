import os
import glob
import numpy as np
import cv2
from PIL import Image
from mrcnn.config import Config
from mrcnn import model as modellib
from image_mixer import mix_images
from tensorflow.compat.v1 import reset_default_graph, get_default_graph

# Constants
IMAGE_EXTENSIONS = ['jpg', 'png']
MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), 'Mask_RCNN', 'model_weights.h5')
LOGS_DIR = os.path.join(os.getcwd(), 'Mask_RCNN', 'logs')
CLASS_NAMES = ['BG', 'car', 'bus', 'truck', 'motorcycle', 'person', 'water']

# Configuration for the model
class InferenceConfig(Config):
    NAME = "inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6  # background + 6 classes
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

# Initialize model
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)
model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)
graph = get_default_graph()

# Utility functions
def find_min_image_number(directory):
    min_number = None
    for file_path in glob.iglob(os.path.join(directory, '*.png')):
        file_name = os.path.basename(file_path)
        number = os.path.splitext(file_name)[0]
        if number.isdigit():
            number = int(number)
            if min_number is None or number < min_number:
                min_number = number
    return min_number

def find_max_image_number(directory):
    max_number = None
    for file_path in glob.iglob(os.path.join(directory, '*.png')):
        file_name = os.path.basename(file_path)
        number = os.path.splitext(file_name)[0]
        if number.isdigit():
            number = int(number)
            if max_number is None or number > max_number:
                max_number = number
    return max_number

def construct_image_path(directory, image_number):
    return os.path.join(directory, f"{image_number}.png")

def apply_mask_to_image(image, mask):
    for channel in range(3):
        image[:, :, channel] = np.where(mask == 0, image[:, :, channel], 255)
    return image

def filter_and_apply_masks(image, boxes, masks, class_ids, class_names, scores, target_type):
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    target_objects = ["car", "motorcycle", "truck", "bus", "person"] if target_type == 1 else ["water"]
    for i in range(len(boxes)):
        if class_names[class_ids[i]] in target_objects:
            combined_mask = np.logical_or(combined_mask, masks[:, :, i])
    return apply_mask_to_image(image, combined_mask)

def make_background_transparent(image):
    image = image.convert('RGBA')
    width, height = image.size
    transparent_color = image.getpixel((0, 0))
    for y in range(height):
        for x in range(width):
            current_color = image.getpixel((x, y))
            if current_color == transparent_color:
                image.putpixel((x, y), current_color[:-1] + (0,))
    return image

def process_image(input_path, output_path, start_dir):
    reset_default_graph()
    image_cv2 = cv2.imread(input_path)
    image_pillow = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    image_pillow_rgba = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGBA))

    with graph.as_default():
        results = model.detect([image_cv2], verbose=0)
    detection_result = results[0]

    if not detection_result['rois'].any():
        image_pillow.save(output_path)
    else:
        for target_type in range(2):
            image_cv2 = cv2.imread(input_path)
            processed_frame = filter_and_apply_masks(image_cv2, detection_result['rois'], detection_result['masks'], detection_result['class_ids'], CLASS_NAMES, detection_result['scores'], target_type)
            processed_frame_pillow = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGBA))
            transparent_frame = make_background_transparent(processed_frame_pillow)
            if target_type == 0:
                car_only_path = os.path.join('Mask_RCNN', 'car_only.png')
                mix_images(image_pillow, transparent_frame, input_type='pillow_img').save(car_only_path)
            else:
                mix_images(image_pillow_rgba, transparent_frame, input_type='pillow_img').save(output_path)

    min_image_number = find_min_image_number(start_dir)
    if min_image_number is not None:
        os.remove(construct_image_path(start_dir, min_image_number))
    get_default_graph().finalize()