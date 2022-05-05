# Author: Oscar Lu

import tensorflow as tf
import numpy as np

_mse = tf.keras.losses.MeanSquaredError()

# pred is a tensor with shape of [bs, 21, 3]. These are the estimated 3D keypoints by MANO.
# gt is a tensor with shape of [bs, 21, 3]. These are the ground-truth 3D keypoints provided by RHD.
def LOSS_2D(pred, gt):
  intrinsic = tf.constant(
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0]
    ])
  pred = tf.matmul(pred, intrinsic)
  gt = tf.matmul(gt, intrinsic)
  return _mse(pred, gt)

def LOSS_3D(pred, gt):
  # MSE = np.square(np.subtract(pred,gt)).mean()
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
def LOSS( beta, pose, L, U, pred, gt ):
  return 1e3 * LOSS_REG(beta, pose, L, U) + 1e2 * ( LOSS_2D(pred, gt) + LOSS_3D(pred, gt))

def distance(arr1,arr2):
  diff = tf.math.subtract(arr1,arr2)
  distance = tf.norm(diff,axis = 2)
  distance = tf.squeeze(distance)
  return distance