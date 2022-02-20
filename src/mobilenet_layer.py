# Author: Lucas Coster

import tensorflow as tf
from tensorflow.keras.layers import Flatten

def MAKE_MOBILE_NET(image_size, image_channels):
  inputs = tf.keras.layers.Input(shape=[image_size,image_size,image_channels])

  mobileNet = tf.keras.applications.MobileNetV3Small(input_shape=[image_size, image_size, image_channels])
  # NOTE(Noah): Need to restart the runtime in Collab, otherwise 
  # tensorflow appends increasing numbers
  # to the model names.
  mobileNetOut = mobileNet.get_layer('global_average_pooling2d').output

  model = tf.keras.models.Sequential()
  model.add(tf.keras.Model(inputs=mobileNet.input, outputs=mobileNetOut))
  model.add(Flatten())

  out = model(inputs)

  return tf.keras.Model(inputs=inputs, outputs=out)

