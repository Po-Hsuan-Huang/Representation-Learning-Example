#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 00:32:14 2018

@author: pohsuanh
"""

import tensorflow as tf

# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
label = tf.constant([2.,3.],[ 4.,5.])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
loss = tf.pow(y -label, 2)
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  # `sess.graph` provides access to the graph used in a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
  writer = tf.summary.FileWriter("'/home/pohsuanh/Documents/Schweighofer Lab'", sess.graph)

  # Perform your computation...
  for i in range(1000):
    sess.run(train_op)
    # ...

  writer.close()
