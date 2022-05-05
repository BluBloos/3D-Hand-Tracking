import tensorflow as tf
from mobilehand_lfuncs import distance
# a = tf.constant([[[0,0,1],[1,1,1],[9,1,0]]],dtype = 'float64')
# b = tf.constant([[[1,0,0],[1,2,3],[9,1,5]]],dtype = 'float64')
# c = distance(a,b)
a = [tf.constant([1,1,1]),tf.constant([2,2,2]),tf.constant([3,3,3]),tf.constant([4,4,4])]
c = tf.stack(a)
c = c[:,0]


print(c)
