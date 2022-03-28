import tensorflow as tf

def blend_shape(beta, S):
  # we note that l is the number of beta parameters.
  # v is the number of vertices in the MANO hand (778)
  # t is the dimensionality of each vertex (3).
  # b is the batch_size.
  return tf.einsum('bl,vtl->bvt', beta, S)