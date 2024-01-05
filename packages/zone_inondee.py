import cv2
import numpy as np

# Import custom modules
from utilities import inrange_blanc, img_adjust_tous_points, del_hort, draw_polygon, convex_hull, edge, del_vert, pixel_colour
import shutil
import imutils

# List of supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = ['jpg', 'png']

def remove_frame_edges(image_path):
    """
    Removes the frame edges of an image by setting the pixel values to black (0, 0, 0).
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # Get image dimensions

    # Set the pixel values to black on the edges of the image
    image[0, :] = (0, 0, 0)
    image[height-1, :] = (0, 0, 0)
    image[:, 0] = (0, 0, 0)
    image[:, width-1] = (0, 0, 0)

    # Save the modified image
    cv2.imwrite(image_path, image)

def fill_edges_with_white(image_input, input_type='image'):
    """
    Fills the edges of an image with white pixels.
    """
    # Read image if input is an image file path
    if input_type == 'image':
        image_input = cv2.imread(image_input)

    # Copy image for flood fill operation
    im_floodfill = image_input.copy()

    # Create mask for flood fill operation
    # The size of the mask should be 2 pixels larger than the image on each side
    height, width = image_input.shape[:2]
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

def calculate_white_area(image):
    """
    Calculates the number of white pixels in an image using vectorization.
    """
    # Use numpy to count non-zero pixels which are white
    white_pixel_count = np.sum(np.all(image == 255, axis=2))

    return white_pixel_count

def find_max_min_area_rectangle(image, input_type='image'):
    """
    Finds the maximum area and dimensions of the minimum bounding rectangle of all the contours in an image.
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

def sort_contours_by_position(contours):
    """
    Sorts a list of contours by the x coordinate of the top-left corner of the minimum bounding rectangle of each contour.
    """
    # Sort contours by the x coordinate of the top-left corner of the bounding rectangle
    contours.sort(key=lambda ctr: cv2.boundingRect(ctr)[0])
    return contours

def get_line_count(image, input_type='image'):
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
    return len(lines) if lines is not None else 0

def calculate_average_line_distance(image, input_type='image'):
    """
    Calculates the average distance between lines in an image.
    """
    # Read image if input is an image file path
    if input_type == 'image':
        image = cv2.imread(image)

    # Get the number of lines in the image
    num_lines = get_line_count(image, input_type='array')

    # Avoid division by zero if there are no lines
    if num_lines == 0:
        return None

    # Get the height of the image
    height = image.shape[0]

    # Calculate the average distance between lines by dividing the height of the image by the number of lines
    average_distance = height / num_lines

    return average_distance