# Author: Max Vincent

import tensorflow as tf

# TODO: Make it so that we do not have to pass in batch_size like this.
def MAKE_REGRESSION_MODULE(batch_size):
  
  MODEL_OUPUTS = 55 + 3 # +3 for the camera rot matrix param (3 = angle axis vector). 
  DROP_PROB = 0.4
  
  inputs = tf.keras.layers.Input(shape=[576,])
  
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape = (576 + MODEL_OUPUTS,)))
  model.add(tf.keras.layers.Dense(288, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(DROP_PROB))
  model.add(tf.keras.layers.Dense(288, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(DROP_PROB))
  model.add(tf.keras.layers.Dense(MODEL_OUPUTS))
  
  params = tf.zeros((batch_size, MODEL_OUPUTS))
  model.summary()

  for i in range(3):
    new_input = tf.concat([inputs, params],axis=1)     
    params += model(new_input)
  
  return tf.keras.Model(inputs=inputs, outputs=params)

