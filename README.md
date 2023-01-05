# inundation_analysis_system
This code is a collection of functions for analyzing and processing images. The functions perform a variety of tasks including taking photos from a CCTV camera, calculating the depth and area of inundation in an image, classifying whether an image contains rain or inundation, detecting vehicles in an image, mixing two images together, making a color transparent in an image, detecting water in an image, detecting the ground in an image, storing data in a database, and detecting crosswalks in an image.

## Requirements
The following libraries are required to run this code:

* os
* time
* sys
* shutil
* cv2
* gc
* numpy
* glob
* pixel2mesh
* mesh2depth
* classify_rain
* classify_inundation
* voiture
* mix_image
* couleur_transparent
* water
* ground
* database
* crosswalk
* zone_inondee

## Functionality
**min_in_file**
This function takes a router as input and returns the minimum value in the file names of the files in the specified router. It does this by looping through the files in the router and extracting the numeric portion of the file name (assumed to be at the beginning of the file name before the first period). It then keeps track of the minimum value it has encountered and returns it at the end.

**max_in_file**
This function is similar to min_in_file, but it returns the maximum value in the file names of the files in the specified router instead of the minimum value.

**min_fichier**
This function takes a fichier name as input and returns the full path to the file with the minimum value in its name in the specified fichier.

**max_fichier**
This function is similar to min_fichier, but it returns the full path to the file with the maximum value in its name in the specified fichier instead of the minimum value.

**prendre_des_photos_CCTV**
This function moves the file with the minimum value in its name in the commener router to the timestamps router.

**inundation_depth**
This function calculates the depth of inundation in an image. It does this by first calling the pixel2mesh.pixel2obj() function which converts pixel data in an image to a 3D mesh. Then, it calls the mesh2depth.obj2height() function, which calculates the depth of inundation in the image based on the 3D mesh. The mesh2obj_dec argument is used to specify whether the function should return the front view of the 3D mesh and the ratio of height to width for the mesh (if mesh2obj_dec is 0), or the depth of inundation (if mesh2obj_dec is 1). If mesh2obj_dec is 1, the front_view and ratio_height arguments must be provided.

**main**
The main function is the entry point for the program. It prompts the user for input and then calls the appropriate functions based on the user's input. It allows the user to specify a fichier, choose whether to print timestamps on images, calculate the inundation region, store data in an Excel sheet or database, and specify the other functions in the code include:

**classify_rain**
This function takes an image as input and returns whether the image contains rain or not.

**classify_inundation**
This function takes an image as input and returns whether the image contains inundation or not.

**voiture**
This function takes an image as input and returns the number of vehicles detected in the image.

**mix_image**
This function takes two images as input and combines them into a single image.

**couleur_transparent**
This function takes an image and a color as input and makes the specified color transparent in the image.

**water**
This function takes an image as input and returns whether the image contains water or not.

**ground**
This function takes an image as input and returns the ground level in the image.

**database**
This function stores data in a database.

**crosswalk**
This function takes an image as input and returns whether the image contains a crosswalk or not.

**zone_inondee**
This function takes an image as input and returns the inundated area in the image.

## Usage
To use this code, run the main function and follow the prompts. You will be asked to enter a fichier name, choose whether to print timestamps on images, calculate the inundation region, store data in an Excel sheet or database, and specify the database or Excel sheet to use if applicable. Then, the appropriate functions will be called based on your input.
