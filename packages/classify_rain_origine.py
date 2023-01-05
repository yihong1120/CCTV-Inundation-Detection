from tensorflow.compat.v1 import reset_default_graph,get_default_graph,disable_eager_execution
import tensorflow as tf
import os
import json
import glob
import numpy as np
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import sys
import gc
sys.path.append(os.path.join(os.getcwd()))
from TimeTransition import TimestampsToTime
disable_eager_execution()

# create graph
global rain_graph
rain_graph = get_default_graph()
# create model
rain_model=tf.keras.models.load_model(os.path.join(os.getcwd(),'Mask_RCNN','Rain_EfficientNet.h5'))

def classify(img_path,output_route,date_de_la_photo,commener):
    
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B3"
    im_height = im_width = img_size[num_model]
    
    # load image
    img = Image.open(img_path).convert('RGB')
    img_cv2 = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    # resize image
    img = img.resize((im_width, im_height))
    # read image
    img = np.array(img).astype(np.float32)
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))
    
    reset_default_graph()
    with rain_graph.as_default():
        result = np.squeeze(rain_model.predict(img))

    out_percent=round(float(result[1]) * 100,2)

    print_res = "{}: {}%".format('rain', out_percent)#:.2f
    
    #positionner des mots et height des rectangles
    ###pic_height=img_cv2.shape[0]
    pic_height=240
    ###pic_width=img_cv2.shape[1]
    pic_width=320
    if pic_height>240:
        zoom_taille=0.7/240*pic_height
        positionner_temps=25/240*pic_height
        positionner_pluie=50/240*pic_height
        height_mot=20/240*pic_height
        width_mot=150/320*pic_width
    elif pic_height<=240:
        positionner_temps=int(25/240*pic_height)
        zoom_taille=0.7
        positionner_pluie=int(50)
        height_mot=20
        width_mot=150
    
    #couleurs des mots
    if out_percent>=50:
        r,g,b=255,0,0
    elif out_percent<50:
        r,g,b=0,255,0
        
    #mettre les mots
    if date_de_la_photo=="y":
        positionner_pluie=positionner_pluie//2
        height_rectangle=positionner_temps-height_mot
        
        #generate timestamp
        time_now = os.path.split(img_path)[1].split(".")[0]
        time_now = TimestampsToTime(time_now)
        print(time_now)
        
        cv2.rectangle(img_cv2, (8, positionner_temps+2), (8+width_mot, positionner_temps-height_mot+2 ), (255, 255, 255), -1)
        cv2.putText(img_cv2, time_now, (10, positionner_temps),  cv2.FONT_HERSHEY_SIMPLEX,
            zoom_taille, (b, g, r), 2)
        cv2.rectangle(img_cv2, (8, positionner_pluie+2), (8+width_mot, positionner_pluie-height_mot+2 ), (255, 255, 255), -1)
        cv2.putText(img_cv2, print_res, (10, positionner_pluie),  cv2.FONT_HERSHEY_SIMPLEX,
            zoom_taille, (b, g, r), 2)
    else:
        positionner_pluie=positionner_pluie//2
        cv2.rectangle(img_cv2, (8, positionner_pluie+2), (8+width_mot, positionner_pluie-height_mot+2 ), (255, 255, 255), -1)
        cv2.putText(img_cv2, print_res, (10, positionner_pluie),  cv2.FONT_HERSHEY_SIMPLEX,
            zoom_taille, (r, g, b), 2)
    cv2.imwrite(output_route, img_cv2)
    print(print_res)

    del img, img_cv2
    get_default_graph().finalize()
    return out_percent
    gc.collect()
    
if __name__ == '__main__':
    start=time.time()
    img_path = './non_rain2.png'
    output_route='./test.png'
    date_de_la_photo='n'
    commener='./'
    classify(img_path,output_route,date_de_la_photo,commener)
    end=time.time()
    print(end-start)
