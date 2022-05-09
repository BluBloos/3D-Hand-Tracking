# Author: Lucas Coster
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model

class MobileNetV3Small(Model):
  
  def __init__(self, image_size, image_channels, **kwargs):
    super().__init__(**kwargs)
    mobileNet = tf.keras.applications.MobileNetV3Small(
      input_shape=[image_size, image_size, image_channels],
      pooling="avg",
      weights="imagenet" # Ensures that mobileNet is pretrained for our transfer learning!
    )
    self.mobileNet = tf.keras.Model(inputs=mobileNet.input, 
      outputs=mobileNet.get_layer('global_average_pooling2d').output)
    self.f = Flatten()
    self.mobileNet.summary()

  def freeze(self):
    self.mobileNet.trainable = False

  def unfreeze(self):
    self.mobileNet.trainable = True

  def call(self, x, training=False):
    y = self.mobileNet(x, training)
    return self.f(y)
