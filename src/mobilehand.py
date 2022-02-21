# We have a three step plan.
# Implementing MobileHand is step 1.

# We are referencing the existing Pytorch implementation to do this.
# https://gmntu.github.io/mobilehand/

import numpy as np
import tensorflow as tf

from mobilenet_layer import MAKE_MOBILE_NET
from ireg_layer import MAKE_REGRESSION_MODULE
from mano_layer import MANO_Model

def MAKE_MOBILE_HAND(image_size, image_channels, batch_size, mano_dir):
  inputs = tf.keras.layers.Input(shape=[image_size, image_size, image_channels])

  mobile_net = MAKE_MOBILE_NET(image_size, image_channels)
  reg_module = MAKE_REGRESSION_MODULE(batch_size)
  mano_model = MANO_Model(mano_dir)

  m_output = mobile_net(inputs)
  reg_output = reg_module(m_output)
  mano_output = mano_model(reg_output)

  # TODO: Get the 3D keypoints from the MANO model.

  # TODO:
  # Quick and dirty visualization of the predictions versus the image.

  return tf.keras.Model(inputs=inputs, outputs=mano_output)


