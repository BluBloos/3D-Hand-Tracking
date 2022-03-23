# Authored by Lucas Coster
# Test for the MANO Linear Blend Skinning functions

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os
import pickle
import time

def lbs(beta, pose, J_, P_, K_, W_, S_, V_):
    
    v_shaped =V_ + blend_shape(beta, S_)

    j_rest = vertices2joints(v_shaped, J_)

def vertices2joints(vert, J_):

    return tf.einsum ('bvt,jv->bjt', vert, J_)

def blend_shape(beta, S_):

    return tf.einsum ('bvt,jv->bjt', beta, S_)