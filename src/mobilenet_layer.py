# Author: Lucas Coster

import tensorflow as tf
from tensorflow.keras.layers import Flatten

def MAKE_MOBILE_NET(image_size, image_channels):
  inputs = tf.keras.layers.Input(shape=[image_size,image_size,image_channels])

  # NOTE(Noah): As per the documentation on 
  # https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small
  # MobileNetV3 models expect their inputs to be float tensors of pixels with values in the [0-255] range.

  mobileNet = tf.keras.applications.MobileNetV3Small(
    input_shape=[image_size, image_size, image_channels],
    weights="imagenet" # Ensures that mobileNet is pretrained for our transfer learning!
  )

  # NOTE(Noah): Need to restart the runtime in Collab, otherwise 
  # tensorflow appends increasing numbers
  # to the model names.
  mobileNetOut = mobileNet.get_layer('global_average_pooling2d').output

  model = tf.keras.models.Sequential()
  model.add(tf.keras.Model(inputs=mobileNet.input, outputs=mobileNetOut))
  model.add(Flatten())

  # NOTE(Noah): For the purposes of transfer learning, we need to explicitly ensure
  # that the weights do not change!
  #model.trainable = False
  out = model(inputs)

  return tf.keras.Model(inputs=inputs, outputs=out)

