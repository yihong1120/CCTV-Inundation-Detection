import cv2
import numpy as np
from PIL import Image
from mrcnn import model as modellib

class WaterConfig(modellib.Config):
    """
    Configuration for the water detection model.
    """
    NAME = "water"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + water class
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

def load_model(model_path, logs_dir):
    """
    Load the Mask R-CNN model with the given configuration.

    Parameters:
    - model_path: The path to the trained weights file.
    - logs_dir: The directory to save logs.

    Returns:
    - The loaded Mask R-CNN model.
    """
    config = WaterConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=logs_dir, config=config)
    model.load_weights(model_path, by_name=True)
    return model

def apply_mask(image, mask):
    """
    Apply the given mask to the image.

    Parameters:
    - image: The original image as a NumPy array.
    - mask: The mask to apply as a NumPy array.

    Returns:
    - The image with the mask applied.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0, 125, image[:, :, c])
    return image

def predict(input_path, output_path, model):
    """
    Run prediction on the input image and save the output.

    Parameters:
    - input_path: The path to the input image.
    - output_path: The path where the output image will be saved.
    - model: The Mask R-CNN model to use for prediction.

    Returns:
    - The inundation area as a float.
    """
    # Load original image
    image = cv2.imread(input_path)
    height, width, _ = image.shape

    # Run detection
    results = model.detect([image], verbose=0)
    r = results[0]

    if not r['rois'].any():
        print("No water detected in this image.")
        return 0

    # Process detections
    mask = np.zeros((height, width), dtype=np.uint8)
    for i, class_id in enumerate(r['class_ids']):
        if class_id == 1:  # Assuming class_id 1 is for water class
            mask = np.maximum(mask, r['masks'][:, :, i])

    # Apply mask to the image
    masked_image = apply_mask(image, mask)
    masked_image_pil = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

    # Save the output image
    masked_image_pil.save(output_path)

    # TODO: Calculate and return the inundation area
    # This part of the code is not provided in the task and needs to be implemented separately.
    return 0  # Placeholder for the inundation area calculation

# Example usage:
# model = load_model('path_to_weights.h5', 'path_to_logs')
# predict('input_image.jpg', 'output_image.jpg', model)