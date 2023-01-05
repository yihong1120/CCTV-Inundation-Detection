import numpy as np
import cv2

def obj2height(cal_ration_height, front_view_donner=None, ratio_height_donner=None):
    # Read in the vertices of the 3D object from the obj file
    obj_file_path = './Mask_RCNN/unique_voiture.obj'
    points = []
    with open(obj_file_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            # Store vertices in points list
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            # Stop reading file after texture coordinates
            if strs[0] == "vt":
                break
    # Convert points list to a matrix for easier processing
    points = np.array(points)

    # Scale points if necessary
    if points.mean() < 1:
        points *= 500
    elif points.mean() < 10:
        points *= 50
    elif points.mean() < 100:
        points *= 5
    # Shift points so that minimum value is positive
    if points.min() < 0:
        points += abs(points.min()) + 10

    # Calculate dimensions of 3D object
    size = []
    # Split points matrix into 3 matrices, one for each dimension
    points0, points1, points2 = np.hsplit(points, 3)
    # Calculate minimum bounding rectangle for each dimension
    for points in (points0, points1, points2):
        points = points.astype(int)
        points = points.reshape((-1, 1, 2))
        rect = cv2.minAreaRect(points)
        (x, y), (width, height), angle = rect
        size.append([width, height])
    size = np.array(size, dtype=int)

    # Calculate the area of each dimension
    area_side = [dim[0] * dim[1] for dim in size]
    area_side = np.array(area_side, dtype=int)
    # Identify the side view and front view dimensions
    side_view_index = np.argmax(area_side)
    front_view_index = np.argmin(area_side)
    side_view = size[side_view_index]
    front_view = size[front_view_index]

    # Scale dimensions so that the minimum value is the same for both front and side views
    if side_view.min() > front_view.min():
        front_view *= side_view.min() / front_view.min()
        front_view = np.array(front_view, dtype=int)
    elif side_view.min() < front_view.min():
        side_view *= front_view.min() / side_view.min()
        side_view = np.array(side_view, dtype=int)

    # Calculate ratios of real world dimensions to dimensions in obj file
    ratio_height = 1415 / front_view.min()  # Real height in cm
    ratio_width = 1745 / front_view.max()  # Real width in cm
    ratio_length = 4430 / side_view.max()  # Real length in cm
    ratio = front_view.max() / front_view.min()  # Ratio of height to width

    if cal_ration_height == 0:
        # Return front view and ratio of height to width
        return front_view, ratio_height
    elif cal_ration_height == 1:
        # Scale front view using provided front view and ratio of height to width
        front_view = front_view * front_view_donner.max() / front_view.max()
        # Calculate inundation depth
        inundation_depth = 1415 - front_view.min() * ratio_height_donner
        inundation_depth = round(inundation_depth, 2)  # Round to 2 decimal places
        return front_view, ratio_height, inundation_depth
    else:
        raise ValueError("Invalid value for cal_ration_height. Must be 0 or 1.")

if __name__=="__main__":
    obj2height('./voiture5.obj',0)
