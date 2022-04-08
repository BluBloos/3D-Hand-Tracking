# Author: Oscar Lu

import tensorflow as tf
import numpy as np

# TODO: Implement.
# Oscar.
def LOSS_2D(pred, gt):
  return tf.zeros([1])

_mse = tf.keras.losses.MeanSquaredError()

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


  