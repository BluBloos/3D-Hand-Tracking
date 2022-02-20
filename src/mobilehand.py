# We have a three step plan.
# Implementing MobileHand is step 1.

# We are referencing the existing Pytorch implementation to do this.
# https://gmntu.github.io/mobilehand/

import numpy as np
import tensorflow as tf

from mobilenet_layer import MOBILE_NET
from ireg_layer import REGRESSION_MODULE

# Mobile net test
MOBILE_NET( np.zeros( (224, 224, 3) ).reshape(1,224,224,3) )

# Regression module testing
input_test = tf.random.uniform(shape = (32,576))
input_test = tf.cast(input_test, tf.float32)
output_test = REGRESSION_MODULE(input_test)
print(output_test)

IMAGE_SIZE = 224

def _MAKE_MOBILE_HAND():
  inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])

  m_output = MOBILE_NET(inputs)
  reg_output = REGRESSION_MODULE(m_output)

  # TODO
  # mano_output = MANO_LAYER(reg_output)
  # TODO:
  # Quick and dirty visualization of the predictions versus the image.

  return tf.keras.Model(inputs=inputs, outputs=reg_output)

MOBILE_HAND = _MAKE_MOBILE_HAND()

# INTEGRATION TEST
input_test = tf.random.uniform(shape = (32,IMAGE_SIZE, IMAGE_SIZE, 3))
input_test = tf.cast(input_test, tf.float32)
output_test = MOBILE_HAND(input_test)
print(output_test)
