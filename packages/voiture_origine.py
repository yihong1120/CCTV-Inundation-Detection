from tensorflow.compat.v1 import reset_default_graph, get_default_graph
import os
import glob
import numpy as np
import cv2
from PIL import Image

from mrcnn.config_mrcnn import Config
from mrcnn import model_mrcnn as modellib
from mix_image import mix_images

# Define supported image extensions
SUPPORTED_EXTENSIONS = ['jpg', 'png']

def find_min_image_number(directory):
    """Find the minimum image number in a directory."""
    min_number = None
    for file_path in glob.iglob(os.path.join(directory, '*.png')):
        file_name = os.path.basename(file_path)
        if file_name.split(".")[0].isdigit():
            image_number = int(file_name.split(".")[0])
            if min_number is None or image_number < min_number:
                min_number = image_number
    return min_number

def find_max_image_number(directory):
    """Find the maximum image number in a directory."""
    max_number = None
    for file_path in glob.iglob(os.path.join(directory, '*.png')):
        file_name = os.path.basename(file_path)
        if file_name.split(".")[0].isdigit():
            image_number = int(file_name.split(".")[0])
            if max_number is None or image_number > max_number:
                max_number = image_number
    return max_number

def get_image_path_with_min_number(directory_name):
    """Get the path of the image with the minimum number in a directory."""
    return os.path.join(directory_name, str(find_min_image_number(directory_name)) + ".png")

def get_image_path_with_max_number(directory_name):
    """Get the path of the image with the maximum number in a directory."""
    return os.path.join(directory_name, str(find_max_image_number(directory_name)) + ".png")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset."""
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6  # background + 6 classes
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

# Set up paths for model weights and logs
COCO_MODEL_PATH = os.path.join('Mask_RCNN', 'Toulouse.h5')
MODEL_DIR = os.path.join('Mask_RCNN', 'logs')

# Initialize configuration
config = ShapesConfig()

# Initialize model
voiture_mrcnn_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
voiture_mrcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)

# Define class names
voiture_mrcnn_class_names = ['BG', 'car', 'bus', 'truck', 'motorcycle', 'person', 'water']

def mrcnn(input_image_path, output_image_path, try_erreur, background_dir):
    """Run Mask R-CNN on the input image and save the result."""
    reset_default_graph()

    def apply_mask(image, mask):
        """Apply the given mask to the image."""
        for c in range(3):
            image[:, :, c] = np.where(mask == 0, 125, image[:, :, c])
        return image

    def display_instances(image, boxes, masks, class_ids, class_names, scores, type_travailler):
        """Display instances of objects in the image."""
        mask = None
        objs = {
            0: ["car"],
            1: ["car", "motorcycle", "truck", "bus", "person"],
            2: ["water"],
            3: ["crosswalk"]
        }.get(type_travailler, [])
        
        largest_area = 0
        largest_index = 0
        for i, roi in enumerate(boxes):
            if class_names[class_ids[i]] in objs:
                if type_travailler == 0:
                    current_mask = masks[:, :, i]
                    mask_area = np.sum(current_mask)
                    if mask_area > largest_area:
                        largest_area = mask_area
                        largest_index = i
                elif type_travailler == 1:
                    mask = masks[:, :, i] if mask is None else mask + masks[:, :, i]

        if type_travailler == 0:
            mask = masks[:, :, largest_index]

        image = apply_mask(image, mask)
        return image

    def transparent_back(image):
        """Make the background of the image transparent."""
        image = image.convert('RGBA')
        L, H = image.size
        color_0 = image.getpixel((0, 0))
        for h in range(H):
            for l in range(L):
                dot = (l, h)
                color_1 = image.getpixel(dot)
                if color_1 == color_0:
                    color_1 = color_1[:-1] + (0,)
                    image.putpixel(dot, color_1)
        return image

    # Read the input image
    image_CV2 = cv2.imread(input_image_path)
    img_terminal = Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGB))
    img_terminal_voitures = Image.fromarray(cv2.cvtColor(image_CV2, cv2.COLOR_BGR2RGBA))

    # Run detection
    results = voiture_mrcnn_model.detect([image_CV2], verbose=0)
    r = results[0]

    # If no objects are detected, save the original image
    if not r['rois'].any():
        img_terminal.save(output_image_path)
        img_terminal.save(os.path.join(background_dir, "background.png"))
    else:
        # Process detected objects
        for i in range(2):
            image_CV2 = cv2.imread(input_image_path)
            frame = display_instances(image_CV2, r['rois'], r['masks'], r['class_ids'], voiture_mrcnn_class_names, r['scores'], i)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            image = transparent_back(frame)

            if i == 1:
                original_backup_img = Image.open(os.path.join(background_dir, "background.png"))
                image = mix_images(original_backup_img, img_terminal_voitures, input_type='pillow_img')
                image.save(output_image_path)
                image.save(os.path.join(background_dir, "background.png"))
            elif i == 0:
                unique_voiture = mix_images(img_terminal, frame, input_type='pillow_img')
                unique_voiture = transparent_back(unique_voiture)
                unique_voiture.save(os.path.join('Mask_RCNN', 'unique_voiture.png'))

    # Clean up
    os.remove(get_image_path_with_min_number(background_dir))
    get_default_graph().finalize()