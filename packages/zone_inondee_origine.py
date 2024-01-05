import os
import cv2
import numpy as np
from packages import (
    inrange_blanc,
    adjust_image_regions,
    delete_horizontal_lines,
    draw_polygons_on_image,
    apply_convex_hull_to_image,
    detect_edges,
    delete_vertical_lines,
    get_pixel_colour
)

SUPPORTED_IMAGE_FORMATS = ['jpg', 'png']

def clear_border(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    border_thickness = 11
    for i in range(height):
        for j in range(width):
            if i < border_thickness or i >= height - border_thickness:
                image[i, j] = (0, 0, 0)
            if j < border_thickness or j >= width - border_thickness:
                image[i, j] = (0, 0, 0)

    cv2.imwrite(image_path, image)

def apply_flood_fill(image, input_type='image'):
    if input_type == 'image':
        image = cv2.imread(image)
    flood_filled_image = image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood_filled_image, mask, (0, 0), (255, 255, 255))
    inverted_flood_filled_image = cv2.bitwise_not(flood_filled_image)

    if input_type == 'image':
        cv2.imwrite('./flood_filled.png', inverted_flood_filled_image)
    else:
        return inverted_flood_filled_image

def count_white_pixels(image):
    return np.sum(image > 0)

def find_largest_contour_rect(image, input_type='image'):
    if input_type == 'image':
        image = cv2.imread(image)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_length = 0
    max_width = 0
    for contour in contours:
        length, width = get_min_area_rect_dimensions(contour)
        area = length * width
        if area > max_area:
            max_length = length
            max_width = width
            max_area = area
    return max_length, max_width

def get_min_area_rect_dimensions(contour):
    rect = cv2.minAreaRect(contour)
    return rect[1][0], rect[1][1]

def calculate_contour_area(height, width, contour):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    area = count_white_pixels(mask)
    cv2.imwrite('./contour_area.png', mask)
    return area

def estimate_crosswalks(image_path, folder_name, input_type='image'):
    if input_type == 'image':
        image = cv2.imread(image_path)
    else:
        image = cv2.cvtColor(np.asarray(image_path), cv2.COLOR_RGB2BGR)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    edges_image = detect_edges(binary_image, input_type='opencv_image')
    flood_filled_image = apply_flood_fill(edges_image, input_type='opencv_image')
    max_length, max_width = find_largest_contour_rect(flood_filled_image, input_type='opencv_image')
    crosswalk_area = count_white_pixels(flood_filled_image)
    crosswalk_count = round(crosswalk_area / (max_length * max_width))

    contours, crosswalk_count = adjust_image_regions(flood_filled_image, input_type='opencv_image')
    if crosswalk_count != 0:
        image_with_polygons = draw_polygons_on_image(flood_filled_image, contours, input_type='opencv_image')
        cv2.imwrite(f'./{folder_name}/draw_polygon.png', image_with_polygons)
        apply_convex_hull_to_image(f'./{folder_name}/draw_polygon.png', input_type='image', output=f'./{folder_name}/draw_polygon.png')

    contours, crosswalk_count = adjust_image_regions(f'./{folder_name}/draw_polygon.png', input_type='image')
    if crosswalk_count != 1:
        crosswalk_count = 0

    pro_length, pro_width = get_min_area_rect_dimensions(contours)
    rate_black = (pro_length - crosswalk_count * max_width) / (crosswalk_count - 1) / max_width
    true_crosswalk_length = 40 * max_length / max_width
    total_length = crosswalk_count * 40 + (crosswalk_count - 1) * 40 * rate_black
    total_width = true_crosswalk_length

    return contours, crosswalk_count, total_length, total_width

def get_min_file_number(directory):
    files = os.listdir(os.path.join(os.getcwd(), directory))
    numbers = [int(file.split(".")[0]) for file in files if file.split(".")[-1] in SUPPORTED_IMAGE_FORMATS]
    return min(numbers)

def get_max_file_number(directory):
    files = os.listdir(os.path.join(os.getcwd(), directory))
    numbers = [int(file.split(".")[0]) for file in files if file.split(".")[-1] in SUPPORTED_IMAGE_FORMATS]
    return max(numbers)

def get_file_path_with_min_number(directory_name):
    min_number = get_min_file_number(directory_name)
    return os.path.join(os.getcwd(), directory_name, f"{min_number}.png")

def get_file_path_with_max_number(directory_name):
    max_number = get_max_file_number(directory_name)
    return os.path.join(os.getcwd(), directory_name, f"{max_number}.png")

def process_crosswalks():
    contours, crosswalk_count, total_length, total_width = estimate_crosswalks('./Mask_RCNN/crosswalk.png', 'crosswalk')
    return contours, crosswalk_count, total_length, total_width

if __name__ == "__main__":
    print("Crosswalk processing initiated.")