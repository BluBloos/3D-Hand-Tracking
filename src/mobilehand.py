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
    ])
  return tf.matmul(points, intrinsic)

# take 3D points and apply the extrinsic camera transform.
def camera_extrinsic(cam_R, depth, scale, points):
  # We make a rodrigues matrix from cam_R angle params.
  rot_mat = tf.reshape(rodrigues(cam_R,),(-1, 3, 3))
  points = tf.matmul(points, rot_mat) # rotate points
  scaling_factor = scale * tf.constant((0.154 / 0.0906426), dtype=tf.float32)
  points *= tf.expand_dims(scaling_factor, axis=1) # scale points
  root_trans = tf.expand_dims(depth, axis=1)
  points += root_trans # apply root translation to points
  return points

def MAKE_MOBILE_HAND(image_size, image_channels, batch_size, mano_dir):
  inputs = tf.keras.layers.Input(shape=[image_size, image_size, image_channels])
  #scale = tf.keras.layers.Input(shape=[1])
  #z_depth = tf.keras.layers.Input(shape=[3])

  mobile_net = MAKE_MOBILE_NET(image_size, image_channels)
  reg_module = MAKE_REGRESSION_MODULE(batch_size)
  mano_model = MANO_Model(mano_dir)

  m_output = mobile_net(inputs)
  reg_output = reg_module(m_output)

  beta = tf.slice(reg_output, tf.constant([ 0, 0 ]), tf.constant([ batch_size, 10 ]))
  pose = tf.slice(reg_output, tf.constant([ 0, 10 ]), tf.constant([ batch_size, 45 ]))
  
  cam_R = tf.slice(reg_output, tf.constant([ 0, 55 ]), tf.constant([ batch_size, 3 ]))

  full_pose = tf.concat( [ tf.repeat(tf.constant([[0.0,0.0,0.0]]), repeats=32, axis=0), pose], axis=1 )
  # root_trans = tf.slice(reg_output, tf.constant([ 0, 58 ]), tf.constant([ batch_size, 3 ]))
  mano_mesh, mano_keypoints = mano_model(beta, full_pose)

  return tf.keras.Model(inputs=[inputs], outputs=[beta, full_pose, mano_mesh, mano_keypoints, cam_R])


