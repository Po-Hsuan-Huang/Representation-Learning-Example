as#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:53:27 2018

@author: pohsuanhuang
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
import os
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "autoencoders"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
def plot_image(image, shape=[28,28]):
    plt.imshow(image.reshape(shape), camp="Greys", interpolation = "nearest")
    plt.axis("off")

""" Data generation"""
import numpy.random as rnd
rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)
    
'''Normalization'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])



assure = input('start training with the produced data ? If yes, enter 1 ') 

assert assure , 'Abort mission.' 

"""Network Sturcture"""
n_inputs = 3

n_hidden = 2

n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape =[None, n_inputs])

hidden = fully_connected(X, n_hidden, activation_fn = None)

outputs = fully_connected(hidden, n_outputs, activation_fn = None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) #MSE

optimizer = tf.train.AdamOptimizer(learning_rate)

training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

n_iterations = 1000

codings = hidden # the output of the hidden layers provies the coding.

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X:X_train}) # no lables (unsupervised)
    codings_val = codings.eval(feed_dict={X:X_test})


fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:,1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
save_fig("linear_autoencoder_pca_plot")
plt.show()