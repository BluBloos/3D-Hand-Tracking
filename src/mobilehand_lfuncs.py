# Author: Oscar Lu

import tensorflow as tf
import numpy as np
from mobilehand import camera_extrinsic
from mobilehand import ortho_camera

# works for [bs, point_count, 3]
def mse(pred, gt):
  bs = pred.shape[0]
  point_count = pred.shape[1]
  mse = tf.reduce_sum(tf.reduce_sum(tf.square(pred - gt), axis=2), axis=1) / point_count
  loss = tf.reduce_sum(mse, axis=0) / bs 
  return loss

#_mse = tf.keras.losses.MeanSquaredError()
_mse = mse

# pred is a tensor with shape of [bs, 21, 3]. These are the estimated 3D keypoints by MANO.
# gt is a tensor with shape of [bs, 21, 3]. These are the ground-truth 3D keypoints provided by RHD.
def LOSS_2D(cam_R, depth, scale, pred, gt):
  pred = ortho_camera(camera_extrinsic(cam_R, depth, scale, pred))
  gt = ortho_camera(gt)
  return _mse(pred, gt)

def LOSS_3D(cam_R, depth, scale, pred, gt):
  # MSE = np.square(np.subtract(pred,gt)).mean()
  pred = camera_extrinsic(cam_R, depth, scale, pred)
  return _mse(pred, gt)

# beta is a tensor with shape of [bs, 10]. These are the estimated beta parameters that get fed to MANO.
# pose is a tensor with shape of [bs, 48]. These are the estimated pose parameters that get fed to MANO.
def LOSS_REG(beta, pose, L, U):
  # U and L are upper and lower limits for the alpha params (which are the things that get mapped into theta MANO params).
  # U and L are only shape R^45.
  pose = pose[ :, 3:]
  loss = tf.square(tf.norm(beta)) + \
    tf.reduce_sum(
      tf.math.maximum(L - pose, tf.zeros(pose.shape)) + \
      tf.math.maximum(pose - U, tf.zeros(pose.shape))
    )
  return loss

# Master loss function
def LOSS( beta, pose, L, U, cam_R, depth, scale, pred, gt ):
  return 1e3 * LOSS_REG(beta, pose, L, U) + 1e2 * ( LOSS_2D(cam_R, depth, scale, pred, gt)) + \
    1e2 * LOSS_3D(cam_R, depth, scale, pred, gt)
  #return (LOSS_2D(cam_R, depth, scale, pred, gt))