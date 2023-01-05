import cv2

def ligne_de_touche(input):
    # Read in image and get image dimensions
    img = cv2.imread(input)
    height, width = img.shape[:2]

    # Initialize output values to 0
    up, down, left, right = 0, 0, 0, 0

    # Check for white pixels at top of image
    for x in range(width):
        for y in range(3):
            if img[y, x] == [255, 255, 255]:
                up = 1
                break

    # Check for white pixels at bottom of image
    for x in range(width):
        for y in range(3):
            if img[height - y, x] == [255, 255, 255]:
                down = 1
                break

    # Check for white pixels at left side of image
    for y in range(height):
        for x in range(4):
            if img[y, x] == [255, 255, 255]:
                left = 1
                break

    # Check for white pixels at right side of image
    for y in range(height):
        for x in range(4):
            if img[y, width - x] == [255, 255, 255]:
                right = 1
                break

    # Return output values
    return up, down, left, right

if __name__ == "__main__":
    input = './17.png'
    up, down, left, right = ligne_de_touche(input)
    print("up =", up)
    print("down =", down)
    print("left =", left)
    print("right =", right)
