# Author: Max Vincent

import tensorflow as tf

# TODO: Make it so that we do not have to pass in batch_size like this.
def _MAKE_REGRESSION_MODULE(batch_size):
  inputs = tf.keras.layers.Input(shape=[576,])
  
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape = (615,)))
  model.add(tf.keras.layers.Dense(288, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(288, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(39, activation = 'relu'))
  
  params = tf.zeros((batch_size,39))
  model.summary()

  for i in range(3):
    new_input = tf.concat([inputs,params],axis=1)     
    params = model(new_input)
  
  return tf.keras.Model(inputs=inputs, outputs=params)

REGRESSION_MODULE = _MAKE_REGRESSION_MODULE(32)