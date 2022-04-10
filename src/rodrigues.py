import tensorflow as tf
from qmindcolors import cstr
import numpy as np
import time

def rodrigues(rvector):
    batch_size = rvector.shape[0]
    angle = tf.norm(rvector, axis = 1, keepdims= True)
    value = tf.fill((batch_size,1), 1e-8)
    angle = tf.math.add(angle, value)


    sin = tf.expand_dims(tf.math.sin(angle), axis = 2)
    cos = tf.expand_dims(tf.math.cos(angle), axis = 2)
   
    runit = rvector/angle
    
    
    kx, ky, kz = tf.split(runit, 3, axis = 1)
    zero = tf.zeros((batch_size,1))
    K = tf.reshape(tf.concat([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis = 1),[batch_size, 3, 3])
    I = tf.eye(3, batch_shape = [batch_size])
    rotationMatrix = I + sin * K + (1 - cos) * tf.linalg.matmul(K, K) 
    return rotationMatrix

    



