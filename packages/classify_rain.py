import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

rain_model = tf.keras.models.load_model('Mask_RCNN/Rain_EfficientNet.h5')

def classify(img_path, output_route, date_de_la_photo, commener):
    img_size = {"B3": 300}
    im_height = im_width = img_size["B3"]

    img = Image.open(img_path).convert('RGB')
    img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = img.resize((im_width, im_height))

    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)

    out_percent = round(float(np.squeeze(rain_model.predict(img))[1]) * 100, 2)

    if out_percent >= 50:
        r, g, b = 255, 0, 0
    else:
        r, g, b = 0, 255, 0

    pic_height = 240
    pic_width = 320
    positionner_temps = int(25 / 240 * pic_height)
    zoom_taille = 0.7
    positionner_pluie = int(50)
    height_mot = 20
    width_mot = 150

    if date_de_la_photo == "y":
        positionner_pluie //= 2
        height_rectangle = positionner_temps - height_mot

        time_now = TimestampsToTime(os.path.splitext(os.path.basename(img_path))[0])

        cv2.rectangle(img_cv2, (8, positionner_temps + 2), (8 + width_mot, positionner_temps - height_mot + 2), (255, 255, 255), -1)
        cv2.putText(img_cv2, time_now, (10, positionner_temps), cv2.FONT_HERSHEY_SIMPLEX, zoom_taille, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(img_cv2, (8, positionner_pluie + 2), (8 + width_mot, positionner_pluie - height_mot + 2), (255, 255, 255), -1)
    cv2.putText(img_cv2, f"rain: {out_percent:.2f}%", (10, positionner_pluie), cv2.FONT_HERSHEY_SIMPLEX, zoom_taille, (r, g, b), 2, cv2.LINE_AA)

    cv2.imwrite(output_route, img_cv2)

    return out_percent
