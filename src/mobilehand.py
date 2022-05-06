# We have a three step plan.
# Implementing MobileHand is step 1.

# We are referencing the existing Pytorch implementation to do this.
# https://gmntu.github.io/mobilehand/

import numpy as np
import tensorflow as tf

from mobilenet_layer import MAKE_MOBILE_NET
from ireg_layer import MAKE_REGRESSION_MODULE
from mano_layer import MANO_Model
from rodrigues import rodrigues

# take 3D points and apply orthographic projection.
def ortho_camera(points):
  intrinsic = tf.constant(
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0]
    ], dtype=tf.float32)
  return tf.matmul(points, intrinsic)

# take 3D points and apply the extrinsic camera transform.
def camera_extrinsic(scale, points):
  points *= scale
  return points

def MAKE_MOBILE_HAND(image_size, image_channels, batch_size, mano_dir):
  inputs = tf.keras.layers.Input(shape=[image_size, image_size, image_channels])

  mobile_net = MAKE_MOBILE_NET(image_size, image_channels)
  print("mobile_net.trainable_variables", mobile_net.trainable_variables)
  reg_module = MAKE_REGRESSION_MODULE(batch_size)
  mano_model = MANO_Model(mano_dir)

  m_output = mobile_net(inputs)
  reg_output = reg_module(m_output)

  beta = tf.slice(reg_output, tf.constant([ 0, 0 ]), tf.constant([ batch_size, 10 ]))
  pose = tf.slice(reg_output, tf.constant([ 0, 10 ]), tf.constant([ batch_size, 48 ]))
  scale = tf.slice(reg_output, tf.constant([ 0, 58 ]), tf.constant([ batch_size, 1 ]))  
  scale = tf.expand_dims(scale, axis=1) # new shape is [bs, 1, 1]

  mano_mesh, mano_keypoints = mano_model(beta, pose)
  return tf.keras.Model(inputs=[inputs], outputs=[beta, pose, mano_mesh, mano_keypoints, scale])


