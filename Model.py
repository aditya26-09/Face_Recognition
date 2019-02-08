# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:01:40 2019

@author: Aditya
"""
#importing libraries
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K


#setting up the model
K.set_image_data_format("channels_first")
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

#triplet loss
def triplet_loss(y_true,y_pred,alpha=0.3):
    anchor,positive,negative = y_pred[0],y_pred[1],y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)))
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))
    return loss

load_weights_from_FaceNet(FRmodel)
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

def prepare_database():
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)
    return database

#creating database
database = {}
database["gates"] = img_path_to_encoding("images/gates.jpg", FRmodel)
database["bezoz"] = img_path_to_encoding("images/jeff-Bezos1.jpg", FRmodel)



def who_is_it(image, database, model):
    encoding = img_path_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.75:
        return None
    else:
        return identity
    
    
