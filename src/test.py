import numpy as np
import tensorflow as tf
from rodrigues import rodrigues

a = tf.constant([[1,2,3],[2,3,4],[3,4,5],[0,0,0],[1,1,1],[0,0,0]], dtype = tf.float32)
b = rodrigues(a)
# c = np.array([1,2,3,4,5,6])
# d = [0,2,4]
# print(d.shape)
# e = c[d]

print(b)