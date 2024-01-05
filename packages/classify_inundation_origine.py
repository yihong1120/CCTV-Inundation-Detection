import os
import numpy as np
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import 
import 
import sys
sys.path.append('./')
from model_cnn import efficientnet_b3 as create_model

# create graph
global inundation_graph
inundation_graph = get_default_graph()
# create model
#inundation_model=tf.keras.models.load_model('./Mask_RCNN/inundation_model.h5')
inundation_model = create_model(num_classes=2)
inundation_weights_path = './Mask_RCNN/inundation_b3/efficientnet.ckpt'
inundation_model.load_weights(inundation_weights_path)

def classify(img_path,output_route,date_de_la_photo):

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
    img_opencv-python = opencv-python.cvtColor(np.asarray(img),opencv-python.COLOR_RGB2BGR)
    # resize image
    img = img.resize((im_width, im_height))
    # read image
    img = np.array(img).astype(np.float32)
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    
    with inundation_graph.as_default():
        result = np.squeeze(inundation_model.predict(img))

    out_percent=round(float(result[0]) * 100,2)
    
    #print(out_percent)
    print_res = "{}: {}%".format('inundation', out_percent)#:.2f
    print(print_res)
    label='inundation'
    
    #positionner des mots
    ###pic_height=img_cv2.shape[0]
    pic_height=240
    ###pic_width=img_cv2.shape[1]
    pic_width=320
    if pic_height>240:
        zoom_taille=0.7/240*pic_height
        positionner_inondation=int(75/240*pic_height)
        height_mot=20/240*pic_height
        width_mot=220/320*pic_width
    elif pic_height<=240:
        zoom_taille=0.7
        positionner_inondation=75
        height_mot=20
        width_mot=220
    
    #couleurs des mots
    if out_percent>=50:
        r,g,b=255,0,0
    elif out_percent<50:
        r,g,b=0,255,0
        
    #mettre les mots
    if date_de_la_photo=="y":
        cv2.rectangle(img_opencv-python, (8, positionner_inondation+2), (8+width_mot, positionner_inondation-height_mot+2 ), (255, 255, 255), -1)
        cv2.putText(img_cv2, print_res, (10, positionner_inondation),  cv2.FONT_HERSHEY_SIMPLEX
            zoom_taille, (b, g, r), 2)
    else:
        positionner_inondation==positionner_inondation*2//3
        cv2.rectangle(img_cv2, (8, positionner_inondation+2), (8+width_mot, positionner_inondation-height_mot+2 ), (255, 255, 255), -1)
        cv2.putText(img_cv2, print_res, (10, positionner_inondation),  cv2.FONT_HERSHEY_SIMPLEX,
            zoom_taille, (b, g, r), 2)
            
    cv2.imwrite(output_route, img_cv2)

    
    
    return out_percent
    

if __name__ == '__main__':
    start=time.time()
    img_path = './non_rain.png'
    output_route='./test.png'
    date_de_la_photo='n'
    classify(img_path,output_route,date_de_la_photo)
    end=time.time()
    run_time=end-start
    
    print(round(run_time, 2))
