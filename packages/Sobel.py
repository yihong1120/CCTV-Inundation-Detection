#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

def apply_sobel_filter(input_image_path, output_image_path):
    """
    Apply the Sobel filter to an image and save the result.

    Parameters:
    input_image_path (str): The path to the input image.
    output_image_path (str): The path where the output image will be saved.
    """
    # Read the input image in grayscale mode
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not open or find the image {input_image_path}")

    # Apply Sobel filter in the x and y directions
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    # Compute the magnitude of the gradients
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.convertScaleAbs(magnitude)

    # Save the result
    cv2.imwrite(output_image_path, magnitude)

# Example usage:
# input_path = "/path/to/input/image.png"
# output_path = "/path/to/output/image.png"
# apply_sobel_filter(input_path, output_path)