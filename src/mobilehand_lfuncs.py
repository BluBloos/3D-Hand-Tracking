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
  #MSE = np.square(np.subtract(pred,gt)).mean()
  # NOTE(Noah): Right now, predictions are not put through MANO layer, so we need to shrink our
  # vector to be of shape (21, 3)
  paddings = tf.constant([[0,0], [0, (21 * 3) - 39]])
  kin_tree = tf.pad( pred, paddings)
  kin_tree = tf.reshape(kin_tree, [tf.shape(kin_tree)[0], 21, 3])
  return _mse(kin_tree, gt)

# TODO: Implement.
def LOSS_REG(pred, gt):
  # U and L are upper and lower limits for the alpha params (which are the things that get mapped into theta MANO params).
  return tf.zeros([1])