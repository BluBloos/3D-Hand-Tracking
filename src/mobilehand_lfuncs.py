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

# TODO: Implement.
def LOSS_REG(pred, gt):
  # U and L are upper and lower limits for the alpha params (which are the things that get mapped into theta MANO params).
  return tf.zeros([1])