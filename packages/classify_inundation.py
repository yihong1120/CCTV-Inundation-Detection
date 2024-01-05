import os
import numpy as np
from PIL import Image
import cv2

from model_cnn import efficientnet_b4 as create_model

model_weights_path = './Mask_RCNN/inundation_b3/efficientnet.ckpt'

def load_model():
    model = create_model(num_classes=2)
    model.load_weights(model_weights_path)
    return model

model = load_model()

def classify_image(img_path, output_path, include_date):
    img_size = 300
    img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    img_array = np.expand_dims(np.array(img).astype(np.float32), 0)
    result = np.squeeze(model.predict(img_array))
    out_percent = round(float(result[0]) * 100, 2)

    label = 'inundation'
    print(f"{label}: {out_percent}%")

    img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_height, img_width = img_cv2.shape[:2]

    font_scale = img_height / 240 * 0.7
    text_y = int(75 / 240 * img_height)
    text_height = int(20 / 240 * img_height)
    text_width = int(220 / 320 * img_width)

    color = (0, 255, 0) if out_percent < 50 else (255, 0, 0)

    if include_date:
        cv2.rectangle(img_cv2, (8, text_y + 2), (8 + text_width, text_y - text_height + 2), (255, 255, 255), -1)
        cv2.putText(img_cv2, f"{label}: {out_percent}%", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    else:
        text_y = text_y * 2 // 3
        cv2.rectangle(img_cv2, (8, text_y + 2), (8 + text_width, text_y - text_height + 2), (255, 255, 255), -1)
        cv2.putText(img_cv2, f"{label}: {out_percent}%", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    cv2.imwrite(output_path, img_cv2)

    return label, out_percent
