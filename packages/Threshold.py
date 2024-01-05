#!/usr/bin/env python
# coding: utf-8

import cv2

def apply_binary_threshold(input_image_path, output_image_path, threshold=60, max_pixel_value=255):
    """
    Apply a binary threshold to an image.

    Parameters:
    input_image_path (str): The path to the input image.
    output_image_path (str): The path to save the thresholded image.
    threshold (int): The threshold value used to classify the pixel values.
    max_pixel_value (int): The value to use for pixels that exceed the threshold.

    Returns:
    None
    """
    # Read the image in grayscale mode
    grayscale_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise FileNotFoundError(f"Input image not found at {input_image_path}")

    # Apply the threshold
    ret, binary_image = cv2.threshold(grayscale_image, threshold, max_pixel_value, cv2.THRESH_BINARY)
    if ret == False:
        raise ValueError(f"Thresholding failed for {input_image_path} with threshold value {threshold}")

    # Write the thresholded image to the output path
    success = cv2.imwrite(output_image_path, binary_image)
    if not success:
        raise IOError(f"Failed to write the thresholded image to {output_image_path}")

# Example usage:
# input_path = "/path/to/input/image.png"
# output_path = "/path/to/output/image.png"
# apply_binary_threshold(input_path, output_path, 60, 255)