import cv2
import numpy as np

# Import custom modules
import inrange_blanc
import img_adjust_tous_points
import del_hort
import draw_polygon
import convex_hull
import edge
import del_vert
import pixel_colour
import shutil
import imutils

# List of supported image file extensions
pic = ['jpg', 'png']

def delete_frame_lines(image):
    """
    Deletes the lines on the edges of an image by setting the pixel values to black (0, 0, 0).
    """
    image = cv2.imread(image)
    height, width = image.shape[:2]  # Get image dimensions

    # Iterate over all pixels in the image
    for i in range(width):
        for j in range(height):
            # Set the pixel values to black if the pixel is located on the edges of the image
            if i == 0 or i == width - 1 or j == 0 or j == height - 1:
                image[i, j] = (0, 0, 0)

    # Save the modified image
    cv2.imwrite(image, image)

def fill_edge(im_in, input_type='image'):
    """
    Fills the exterior of an image with white pixels.
    """
    # Read image if input is an image file path
    if input_type == 'image':
        im_in = cv2.imread(im_in)

    # Copy image for flood fill operation
    im_floodfill = im_in.copy()

    # Create mask for flood fill operation
    # The size of the mask should be 2 pixels larger than the image on each side
    height, width = im_in.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # Flood fill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), (255, 255, 255))

    # Invert flood filled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Return the inverted flood filled image if input is not an image file path
    if input_type != 'image':
        return im_floodfill_inv

    # Save the inverted flood filled image if input is an image file path
    cv2.imwrite('./testest.png', im_floodfill_inv)

def white_area_size(image):
    """
    Counts the number of white pixels in an image.
    """
    white_pixel_count = 0

    # Iterate over all pixels in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Increment white_pixel_count if the pixel value is greater than 0
            if image[i, j].all() > 0:
                white_pixel_count += 1

    return white_pixel_count

def max_min_area_rect(image, input_type='image'):
    """
    Returns the maximum area and dimensions of the minimum bounding rectangle of all the contours in an image.
    """
    # Read image if input is an image file path
    if input_type == 'image':
        image = cv2.imread(image)

    # Find all contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to keep track of the maximum area and dimensions of the bounding rectangles
    max_area = 0
    max_length = 0
    max_width = 0

    # Iterate over all contours
    for cont in contours:
        # Get the minimum bounding rectangle of the contour
        rect = cv2.minAreaRect(cont)
        # Get the dimensions of the bounding rectangle
        length, width = rect[1]
        # Calculate the area of the bounding rectangle
        area = length * width
        # Update max_area, max_length, and max_width if the area of the current bounding rectangle is greater
        if area > max_area:
            max_area = area
            max_length = length
            max_width = width

    return max_area, max_length, max_width

def sort_contours(contours):
    """
    Sorts a list of contours by the x coordinate of the top-left corner of the minimum bounding rectangle of each contour.
    """
    # Sort contours by the x coordinate of the top-left corner of the bounding rectangle
    contours.sort(key=lambda ctr: cv2.boundingRect(ctr)[0])
    return contours

def count_lines(image, input_type='image'):
    """
    Returns the number of lines in an image.
    """
    # Read image if input is an image file path
    if input_type == 'image':
        image = cv2.imread(image)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection to the image
    edges = cv2.Canny(blur, 50, 150)

    # Get lines in the image using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # Return the number of lines in the image
    return len(lines)

def line_distance(image, input_type='image'):
    """
    Returns the average distance between lines in an image.
    """
    # Read image if input is an image file path
    if input_type == 'image':
        image = cv2.imread(image)

    # Get the number of lines in the image
    num_lines = count_lines(image, input_type='array')

    # Get the height of the image
    height = image.shape[0]

    # Calculate the average distance between lines by dividing the height of the image by the number of lines
    average_distance = height / num_lines

    return average_distance

