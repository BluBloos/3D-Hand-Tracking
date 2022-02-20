# Author: Lucas Coster

import tensorflow as tf
from tensorflow.keras.layers import Flatten

def _MAKE_MOBILE_NET():
  inputs = tf.keras.layers.Input(shape=[224,224,3])

  mobileNet = tf.keras.applications.MobileNetV3Small(input_shape=[224, 224, 3])
  # NOTE(Noah): Need to restart the runtime in Collab, otherwise 
  # tensorflow appends increasing numbers
  # to the model names.
  mobileNetOut = mobileNet.get_layer('global_average_pooling2d').output

  model = tf.keras.models.Sequential()
  model.add(tf.keras.Model(inputs=mobileNet.input, outputs=mobileNetOut))
  model.add(Flatten())

  out = model(inputs)

  return tf.keras.Model(inputs=inputs, outputs=out)

MOBILE_NET = _MAKE_MOBILE_NET()