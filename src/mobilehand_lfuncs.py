# Author: Oscar Lu

import tensorflow as tf
import numpy as np

_mse = tf.keras.losses.MeanSquaredError()

# pred is a tensor with shape of [bs, 21, 3]. These are the estimated 3D keypoints by MANO.
# gt is a tensor with shape of [bs, 21, 3]. These are the ground-truth 3D keypoints provided by RHD.
def LOSS_2D(pred, gt):
  return _mse(pred, gt)

# TODO: Implement.
def LOSS_3D(pred, gt):
  # MSE = np.square(np.subtract(pred,gt)).mean()
  return _mse(pred, gt)

# beta is a tensor with shape of [bs, 10]. These are the estimated beta parameters that get fed to MANO.
# pose is a tensor with shape of [bs, 48]. These are the estimated pose parameters that get fed to MANO.
def LOSS_REG(beta, pose):

  bs = beta.shape[0]
  U = tf.constant([bs, 48])
  L = tf.constant([bs, 48])

  return tf.zeros([1])


  