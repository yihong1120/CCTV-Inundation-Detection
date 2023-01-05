# Load required libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the TensorFlow model
rain_model = tf.keras.models.load_model('Mask_RCNN/Rain_EfficientNet.h5')

def classify(img_path, output_route, date_de_la_photo, commener):
    # Set image size
    img_size = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380, "B5": 456, "B6": 528, "B7": 600}
    num_model = "B3"
    im_height = im_width = img_size[num_model]

    # Load and resize image
    img = Image.open(img_path).convert('RGB')
    img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = img.resize((im_width, im_height))

    # Read image into a NumPy array and add it to a batch where it is the only member
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)

    # Run image through TensorFlow model and get prediction
    result = np.squeeze(rain_model.predict(img))

    # Calculate probability of 'rain' class
    out_percent = round(float(result[1]) * 100, 2)

    # Set up variables for text label position, height, and width
    pic_height = 240
    pic_width = 320
    if pic_height > 240:
        zoom_taille = 0.7 / 240 * pic_height
        positionner_temps = 25 / 240 * pic_height
        positionner_pluie = 50 / 240 * pic_height
        height_mot = 20 / 240 * pic_height
        width_mot = 150 / 320 * pic_width
    elif pic_height <= 240:
        positionner_temps = int(25 / 240 * pic_height)
        zoom_taille = 0.7
        positionner_pluie = int(50)
        height_mot = 20
        width_mot = 150

    # Set color of text labels based on probability of 'rain' class
    if out_percent >= 50:
        r, g, b = 255, 0, 0
    elif out_percent < 50:
        r, g, b = 0, 255, 0

    # Add text labels to image
    if date_de_la_photo == "y":
        positionner_pluie = positionner_pluie // 2
        height_rectangle = positionner_temps - height_mot

        # Generate timestamp
        time_now = os.path.split(img_path)[1].split(".")[0]
        time_now = TimestampsToTime(time_now)
        print(time_now)

        cv2.rectangle(img_cv2, (8, positionner_temps + 2), (8 + width_mot, positionner_temps - height_mot + 2), (255, 255, 255), -1)
        cv2.putText(img_cv2, time_now, (10, positionner_temps), cv2.FONT_HERSHEY_SIMPLEX, zoom_taille, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(img_cv2, (8, positionner_pluie + 2), (8 + width_mot, positionner_pluie - height_mot + 2), (255, 255, 255), -1)
    cv2.putText(img_cv2, "rain: {}%".format(out_percent), (10, positionner_pluie), cv2.FONT_HERSHEY_SIMPLEX, zoom_taille, (r, g, b), 2, cv2.LINE_AA)

    # Save modified image
    cv2.imwrite(output_route, img_cv2)
    
    return out_percent
