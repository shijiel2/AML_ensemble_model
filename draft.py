import tensorflow as tf
import sys
import numpy as np
from settings import Settings

# y_train = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

# y = tf.placeholder(y_train.dtype, (None,) + y_train.shape[1:])
# print(y.dtype)

# zeros = tf.zeros_like(y)
# ones = tf.ones((tf.shape(y)[0], 1), dtype=y.dtype)
# dummy_y = tf.concat([zeros, ones], 1)

# # paddings = tf.constant([[0, 0,], [0, 1]])
# # dummy_y = tf.pad(y, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
# #                                  #  [0, 0, 1, 2, 3, 0, 0],
# #                                  #  [0, 0, 4, 5, 6, 0, 0],
#                                  #  [0, 0, 0, 0, 0, 0, 0]]

# with tf.Session() as sess:  
#     print(dummy_y.eval(feed_dict={y: y_train})) 

x = np.zeros((10, 3))
x[:,-1] = 1.

print(x)