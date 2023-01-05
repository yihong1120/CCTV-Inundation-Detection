import os
import json
import glob
import numpy as np
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gc
import cv2
import sys

# Import model from model_cnn module
from model_cnn import efficientnet_b3 as create_model

# Set up efficientnet model for classification
model = create_model(num_classes=2)
model_weights_path = './Mask_RCNN/inundation_b3/efficientnet.ckpt'
model.load_weights(model_weights_path)

def classify_image(img_path, output_path, include_date):
    """
    Classify an image as either being inundated or not.
    
    Parameters:
    - img_path (str): Path to the image file
    - output_path (str): Path to save the output image with text showing the prediction
    - include_date (str): Flag indicating whether or not to include the date in the output image
    
    Returns:
    - label (str): Label of the prediction ('inundation')
    - out_percent (float): Prediction as a percentage
    """
    
    # Image size and scaling parameters
    img_size = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380, "B5": 456, "B6": 528, "B7": 600}
    model_num = "B3"
    img_height = img_width = img_size[model_num]
    
    # Load and resize image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_width, img_height))
    
    # Convert image to numpy array and add it to a batch with one member
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)
    
    # Get prediction from model
    result = np.squeeze(model.predict(img))
    
    # Calculate prediction as a percentage and round to 2 decimal points
    out_percent = round(float(result[0]) * 100, 2)
    
    # Print and format prediction
    print_res = f"{label}: {out_percent}%"
    print(print_res)
    
    # Set label for prediction
    label = 'inundation'
    
    # Load image in cv2 format for adding text
    img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
    # Image dimensions
    img_height = img_cv2.shape[0]
    img_width = img_cv2.shape[1]
    
    # Set font scaling and text position based on image dimensions
    if img_height > 240:
        font_scale = 0.7 / 240 * img_height
        text_y = int(75 / 240 * img_height)
        text_height = 20 / 240
        text_width = 220 / 320 * img_width
    else:
        font_scale = 0.7
        text_y = 75
        text_height = 20
        text_width = 220
    
    # Set color for text based on prediction percentage
    if out_percent >= 50:
        r, g, b = 255, 0, 0
    else:
        r, g, b = 0, 255, 0
        
    # Add text to image
    if include_date:
        cv2.rectangle(img_cv2, (8, text_y + 2), (8 + text_width, text_y - text_height + 2), (255, 255, 255), -1)
        cv2.putText(img_cv2, print_res, (10, text_y),  cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (b, g, r), 2)
    else:
        text_y = text_y * 2 // 3
        cv2.rectangle(img_cv2, (8, text_y + 2), (8 + text_width, text_y - text_height + 2), (255, 255, 255), -1)
        cv2.putText(img_cv2, print_res, (10, text_y),  cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (b, g, r), 2)
    
    # Save image with text to output path
    cv2.imwrite(output_path, img_cv2)
    
    return label, out_percent
