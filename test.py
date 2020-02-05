import numpy as np
import tensorflow as tf

x1 = [[1., 1., 1., 1.],
      [1., 1., 1., 1.],
      [0., 0., 0., 0.]]

x2 = [[2., 2., 4., 4.],
      [0., 0., 0., 0.],
      [0., 0., 0., 0.]]

x = tf.stack([x1, x2], 1)

w = [[0.5, 0.5],
     [0.6, 0.4],
     [0., 0.]]
w = np.array(w)

y = x * w[:, :, None]
yt = tf.math.reduce_sum(y, 1)
sess = tf.Session()


print(sess.run(yt))