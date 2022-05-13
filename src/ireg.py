# Author: Max Vincent, Noah Cabral

import tensorflow as tf
from tensorflow.keras import Model

class IterativeRegression(Model):
  
  def __init__(self, outputs, dropout_prob, T, **kwargs):
    super().__init__(**kwargs)
    self.MODEL_OUTPUTS = outputs
    self.DROP_PROB = dropout_prob
    self.T = T
    self._layers = [
      tf.keras.layers.Dense(288, activation = 'relu', input_shape=(576 + self.MODEL_OUTPUTS,)),
      tf.keras.layers.Dropout(self.DROP_PROB),
      tf.keras.layers.Dense(288, activation = 'relu'),
      tf.keras.layers.Dropout(self.DROP_PROB),
      tf.keras.layers.Dense(self.MODEL_OUTPUTS)
    ]
  
  def call(self, features, training=False):
    bs = tf.shape(features)[0] 
    y_t = tf.zeros((bs, self.MODEL_OUTPUTS))
    y_vals = []
    for t in range(self.T):
      new_input = tf.concat([features, y_t],axis=1)     
      err = self._layers[0](new_input)
      for layer in self._layers[1:]:
        err = layer(err)
      y_t += err
      # print(cstr("y_t"), y_t)
      y_vals.append(y_t)
    return y_t if not training else y_vals

