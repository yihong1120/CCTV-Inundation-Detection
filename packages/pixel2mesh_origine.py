#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import gc
from PIL import Image
from tensorflow.compat.v1 import reset_default_graph,get_default_graph,set_random_seed, placeholder, sparse_placeholder, float32,int32, ConfigProto, Session, global_variables_initializer, app
#tf.disable_v2_behavior()
#from tensorflow.keras.backend import clear_session
import pickle
import sys
import os
from skimage import io,transform
sys.path.append('./')
from p2m.api_depth import GCN
from p2m.utils_depth import construct_feed_dict


# Set random seed
seed_depth = 1024
np.random.seed(seed_depth)
set_random_seed(seed_depth)



# Define placeholders(dict) and model
num_blocks_depth = 3
num_supports_depth = 2



placeholders_depth = {
    'features_depth': placeholder(float32, shape=(None, 3)), # initial 3D coordinates
    'img_inp_depth': placeholder(float32, shape=(224, 224, 3)), # input image to network
    'labels_depth': placeholder(float32, shape=(None, 6)), # ground truth (point cloud with vertex normal)
    'support1_depth': [sparse_placeholder(float32) for _ in range(num_supports_depth)], # graph structure in the first block
    'support2_depth': [sparse_placeholder(float32) for _ in range(num_supports_depth)], # graph structure in the second block
    'support3_depth': [sparse_placeholder(float32) for _ in range(num_supports_depth)], # graph structure in the third block
    'faces_depth': [placeholder(int32, shape=(None, 4)) for _ in range(num_blocks_depth)], # helper for face loss (not used)
    'edges_depth': [placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth)], # helper for normal loss
    'lape_idx_depth': [placeholder(int32, shape=(None, 10)) for _ in range(num_blocks_depth)], # helper for laplacian regularization
    'pool_idx_depth': [placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth-1)] # helper for graph unpooling
}


'''
class placeholders_depth():
    num_blocks_depth = 3
    num_supports_depth = 2
    features_depth=placeholder(float32, shape=(None, 3)) # initial 3D coordinates
    img_inp_depth=placeholder(float32, shape=(224, 224, 3)) # input image to network
    labels_depth=placeholder(float32, shape=(None, 6)) # ground truth (point cloud with vertex normal)
    support1_depth=[sparse_placeholder(float32) for _ in range(num_supports_depth)] # graph structure in the first block
    support2_depth=[sparse_placeholder(float32) for _ in range(num_supports_depth)] # graph structure in the second block
    support3_depth=[sparse_placeholder(float32) for _ in range(num_supports_depth)] # graph structure in the third block
    faces_depth=[placeholder(int32, shape=(None, 4)) for _ in range(num_blocks_depth)] # helper for face loss (not used)
    edges_depth=[placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth)] # helper for normal loss
    lape_idx_depth=[placeholder(int32, shape=(None, 10)) for _ in range(num_blocks_depth)] # helper for laplacian regularization
    pool_idx_depth=[placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth-1)] # helper for graph unpooling
'''

#global depth_model

depth_model = GCN(placeholders_depth, logging=True)

# Load data, initialize session
config_depth=ConfigProto()
config_depth.gpu_options.allow_growth=True
config_depth.allow_soft_placement=True
sess_depth = Session(config=config_depth)
sess_depth.run(global_variables_initializer())
depth_model.load(sess_depth)

# Runing the demo
#reload(sys)
#sys.setdefaultencoding('utf-8')

pkl_depth = pickle.load(open('./Mask_RCNN/Data/ellipsoid/info_ellipsoid.dat', 'rb'),encoding='bytes')
feed_dict_depth = construct_feed_dict(pkl_depth, placeholders_depth)
#reset_default_graph()
#global img_inp
depth_graph = get_default_graph()


def pixel2obj():
    
    '''
    sess_depth.graph.finalize()
    '''
    
    '''
    class placeholders_depth():
        # Define placeholders(dict) and model
        num_blocks_depth = 3
        num_supports_depth = 2
        features_depth=placeholder(float32, shape=(None, 3)) # initial 3D coordinates
        img_inp_depth=placeholder(float32, shape=(224, 224, 3)) # input image to network
        labels_depth=placeholder(float32, shape=(None, 6)) # ground truth (point cloud with vertex normal)
        support1_depth=[sparse_placeholder(float32) for _ in range(num_supports_depth)] # graph structure in the first block
        support2_depth=[sparse_placeholder(float32) for _ in range(num_supports_depth)] # graph structure in the second block
        support3_depth=[sparse_placeholder(float32) for _ in range(num_supports_depth)] # graph structure in the third block
        faces_depth=[placeholder(int32, shape=(None, 4)) for _ in range(num_blocks_depth)] # helper for face loss (not used)
        edges_depth=[placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth)] # helper for normal loss
        lape_idx_depth=[placeholder(int32, shape=(None, 10)) for _ in range(num_blocks_depth)] # helper for laplacian regularization
        pool_idx_depth=[placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth-1)] # helper for graph unpooling
    '''
    
    '''
    # Set random seed
    seed_depth = 1024
    np.random.seed(seed_depth)
    set_random_seed(seed_depth)

    # Define placeholders(dict) and model
    num_blocks_depth = 3
    num_supports_depth = 2
    placeholders_depth = {
        'features_depth': placeholder(float32, shape=(None, 3)), # initial 3D coordinates
        'img_inp_depth': placeholder(float32, shape=(224, 224, 3)), # input image to network
        'labels_depth': placeholder(float32, shape=(None, 6)), # ground truth (point cloud with vertex normal)
        'support1_depth': [sparse_placeholder(float32) for _ in range(num_supports_depth)], # graph structure in the first block
        'support2_depth': [sparse_placeholder(float32) for _ in range(num_supports_depth)], # graph structure in the second block
        'support3_depth': [sparse_placeholder(float32) for _ in range(num_supports_depth)], # graph structure in the third block
        'faces_depth': [placeholder(int32, shape=(None, 4)) for _ in range(num_blocks_depth)], # helper for face loss (not used)
        'edges_depth': [placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth)], # helper for normal loss
        'lape_idx_depth': [placeholder(int32, shape=(None, 10)) for _ in range(num_blocks_depth)], # helper for laplacian regularization
        'pool_idx_depth': [placeholder(int32, shape=(None, 2)) for _ in range(num_blocks_depth-1)] # helper for graph unpooling
    }

    #global depth_model
    depth_model = GCN(placeholders_depth, logging=True)

    # Load data, initialize session
    config_depth=ConfigProto()
    config_depth.gpu_options.allow_growth=True
    config_depth.allow_soft_placement=True
    sess_depth = Session(config=config_depth)
    #sess_depth.run(global_variables_initializer())#####
    depth_model.load(sess_depth) #sess_depth

    # Runing the demo
    #reload(sys)
    #sys.setdefaultencoding('utf-8')
    pkl_depth = pickle.load(open('./Mask_RCNN/Data/ellipsoid/info_ellipsoid.dat', 'rb'),encoding='bytes')
    feed_dict_depth = construct_feed_dict(pkl_depth, placeholders_depth)
    #reset_default_graph()
    #global img_inp
    img_inp = get_default_graph()
    '''
    
    
    reset_default_graph()
    def load_image(img_path):
        img = io.imread(img_path)
        if img.shape[2] == 4:
            img[np.where(img[:,:,3]==0)] = 255
        img = transform.resize(img, (224,224))
        img = img[:,:,:3].astype('float32')
        return img
        
    pkl_depth = pickle.load(open('./Mask_RCNN/Data/ellipsoid/info_ellipsoid.dat', 'rb'),encoding='bytes')
    feed_dict_depth = construct_feed_dict(pkl_depth, placeholders_depth)
    #reset_default_graph()
    #global img_inp
        
    #reset_default_graph()
    img_inp = Image.open('./Mask_RCNN/unique_voiture.png')
    img_inp = img_inp.resize((224, 224),Image.ANTIALIAS)
    #img_inp = Image.merge("RGB",img_inp)
    #img_inp = np.array(img_inp, dtype='float32')
    img_inp = np.array(img_inp).astype(np.float32)
    img_inp = img_inp[:, :, :3]
        
    #img_inp = get_default_graph()
    #####reset_default_graph()
    #####img_inp = load_image('./Mask_RCNN/unique_voiture.png') #FLAGS.image
    with depth_graph.as_default():
        feed_dict_depth.update({placeholders_depth['img_inp_depth']: img_inp})
        feed_dict_depth.update({placeholders_depth['labels_depth']: np.zeros([10,6])})
    
    #with img_inp.as_default():
    vert = sess_depth.run(depth_model.output3, feed_dict=feed_dict_depth)
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    face = np.loadtxt('./Mask_RCNN/Data/ellipsoid/face3.obj', dtype='|S32')
    mesh = np.vstack((vert, face))
    #pred_path = FLAGS.image.replace('.png', '.obj') #FLAGS.image
    pred_path='./Mask_RCNN/unique_voiture.obj'
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

    del img_inp
    get_default_graph().finalize()
    #####gc.collect()
    
    '''
    # Load data, initialize session
    config_depth=ConfigProto()
    config_depth.gpu_options.allow_growth=True
    #config_depth.allow_soft_placement=False
    sess_depth = Session(config=config_depth)
    sess_depth.run(global_variables_initializer())
    sess_depth.graph.finalize()
    '''
    
if __name__=="__main__":
    pixel2obj()
