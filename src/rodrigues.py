import tensorflow as tf
from qmindcolors import cstr
import numpy as np

def rodrigues(rvector):
    batch_size = rvector.shape[0]
    angle = tf.norm(rvector, axis = 1, keepdims= True)
    print(cstr("rodrigues_angle"), angle)

    sin = tf.expand_dims(tf.math.sin(angle), axis = 2)
    cos = tf.expand_dims(tf.math.cos(angle), axis = 2)
    angles = tf.unstack(angle, axis = 0)
    where = np.where(angles == tf.constant([0]))[0]
    for i in where:
        angles[i] = tf.constant([1.0])
    
    # print('\n', angles)
    
    # for i in range(batch_size):
    #     angles[i] = tf.cond(angles[i] == 0, true_fn=lambda:tf.constant([1.0]), false_fn=lambda:angles[i])
    angle = tf.stack(angles) 
    print('\n', angle)
    runit = rvector/angle
    
    
    kx, ky, kz = tf.split(runit, 3, axis = 1)
    zero = tf.zeros((batch_size,1))
    K = tf.reshape(tf.concat([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis = 1),[batch_size, 3, 3])
    I = tf.eye(3, batch_shape = [batch_size])
    rotationMatrix = I + sin * K + (1 - cos) * tf.linalg.matmul(K, K)    
    
    return rotationMatrix

    



