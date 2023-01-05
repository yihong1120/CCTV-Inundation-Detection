import cv2
import numpy as np

# Read in image and convert to grayscale
image = cv2.imread('./crosswalk/1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply inverse binary threshold to grayscale image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = np.array(contours)

# Find minimum bounding rectangle for contours
rect = cv2.minAreaRect(contours)
# Get coordinates of minimum bounding rectangle
box = cv2.boxPoints(rect)
# Convert coordinates to integers
box = np.int0(box)
# Draw minimum bounding rectangle on original image
cv2.drawContours(image, [box], 0, (0, 0, 255), 3)

# Draw contours on original image
cv2.drawContours(image, contours, -1, (255, 0, 0), 1)

# Display image and save to file
cv2.imshow("image", image)
cv2.imwrite("./img_1.jpg", image)
