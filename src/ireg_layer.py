# Author: Max Vincent

import tensorflow as tf

# TODO: Make it so that we do not have to pass in batch_size like this.
def MAKE_REGRESSION_MODULE(batch_size):
  
  MODEL_OUPUTS = 55 + 3 # +3 for the camera rot matrix param (3 = angle axis vector). 
  
  inputs = tf.keras.layers.Input(shape=[576,])
  
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape = (576 + MODEL_OUPUTS,)))
  model.add(tf.keras.layers.Dense(288, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(288, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(MODEL_OUPUTS, activation = 'relu'))
  
  params = tf.zeros((batch_size, MODEL_OUPUTS))
  model.summary()

  for i in range(4):
    new_input = tf.concat([inputs, params],axis=1)     
    params += model(new_input)
  
  return tf.keras.Model(inputs=inputs, outputs=params)

