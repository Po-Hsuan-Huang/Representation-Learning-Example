#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:18:22 2018

@author: pohsuanh
"""
import tensorflow as tf

#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

import matplotlib.pyplot as plt
import os, sys
from functools import partial
import numpy as np

# Basic model parameters.
BATCH_SIZE = 16
NUM_EXAMPLES = 10000
IMG_SIZE = 28


n_digits =60


class NMIST(object):
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data

        self.mnist = input_data.read_data_sets("/tmp/data/")
        global NUM_EXAMPLES, IMG_SIZE, BATCH_SIZE
        NUM_EXAMPLES = self.mnist.train.num_examples
        IMG_SIZE = 28
    def mnist_data(self,batch_size=BATCH_SIZE):
        # load and return data and label X_batch, Y_batch
        return self.mnist.train.next_batch(batch_size)
    
class CIFAR10(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        global NUM_EXAMPLES, IMG_SIZE, BATCH_SIZE
        IMG_SIZE = 32
        NUM_EXAMPLES = self.x_train.shape[0]
        # Image Normalization
        for i , img in enumerate(self.x_train) :
            img_norm = img - np.mean(img)
            img_norm = img_norm/(np.max(img_norm)-np.min(img_norm))
            self.x_train[i] = img_norm
        for i , img in enumerate(self.x_test) :
            img_norm = img - np.mean(img)
            img_norm = img_norm/(np.max(img_norm)-np.min(img_norm))
            self.x_test[i] = img_norm   
         
    def _cifar10_data(self, batch_size=BATCH_SIZE, _eval=False):
        # laod and return data and label X_batch, Y_batch
        if not _eval : # laod trainset
          X_batch = tf.data.Dataset.from_tensor_slices( self.x_train).repeat().batch(batch_size)
          Y_batch = tf.data.Dataset.from_tensor_slices( self.y_train ).repeat().batch(batch_size)
          return X_batch, Y_batch
        else:
          X_batch = tf.data.Dataset.from_tensor_slices( self.x_test ).repeat().batch(batch_size)
          Y_batch = tf.data.Dataset.from_tensor_slices( self.y_test ).repeat().batch(batch_size)
          return X_batch, Y_batch
      
    def cifar10_input(self, batch_size=BATCH_SIZE, _eval=False):
         X_batch, Y_batch = self._cifar10_data(batch_size,_eval)
         return X_batch.make_one_shot_iterator(), Y_batch.make_one_shot_iterator()
 
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)    
#%%    
reset_graph()    
cifar10 = CIFAR10()
X, Y = cifar10.cifar10_input()
X_batch, Y_batch = X.get_next(), Y.get_next()
n_inputs = IMG_SIZE * IMG_SIZE * 3
n_hidden1 = 200
n_hidden2 = 100
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 10**-4

initializer = tf.contrib.layers.variance_scaling_initializer()
my_dense_layer = partial(
    tf.layers.dense,
    activation=None, #tf.nn.elu,
    kernel_initializer=initializer)
inputs =  tf.cast(X_batch, tf.float32)
print_in = tf.Print(inputs,[inputs[0]],'input_img')
inputs = tf.reshape(inputs,[-1,IMG_SIZE * IMG_SIZE*3])
#inputs = tf.layers.batch_normalization(inputs)
hidden1 = my_dense_layer(inputs, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
print_logit = tf.Print(logits,[hidden1],'output_logit')
outputs = tf.sigmoid(logits)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
latent_loss = 0.5 * tf.reduce_sum(
    tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
loss = reconstruction_loss + latent_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 15


with tf.Session() as sess:
    writer = tf.summary.FileWriter("/home/pohsuanh/Documents/Schweighofer Lab", sess.graph)
    init.run()
    saver.recover_last_checkpoints("./my_model_variational.ckpt")
    for epoch in range(n_epochs):
        n_batches = NUM_EXAMPLES // BATCH_SIZE
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            sess.run(training_op)
        # eval_inputs     
        X, Y = cifar10.cifar10_input( BATCH_SIZE, True)
        X_batch, Y_batch = X.get_next(), Y.get_next()
        inputs =  tf.cast(X_batch, tf.float32)
        inputs = tf.reshape(inputs,[-1,IMG_SIZE * IMG_SIZE*3])
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss])
        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
        saver.save(sess, "./my_model_variational.ckpt")
    writer.close()


    #(    """     Generate digits from random latent state samples      """     )

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})
#%%
def plot_image(image, shape=[32,32,3], colors="gray"):
    # NMINST shape: [32,32] colors: 'Grey'
    plt.imshow(image.reshape(shape)*1/np.max(image), cmap=colors, interpolation="nearest")
    plt.axis("off")
    
def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")   
#%% """     Generate digits      """ 
def generate_digits() :
    print(    """     Generate digits      """     )
    global outputs_eval
    global n_digits
    
    plt.figure() # not shown in the book
    for iteration in range(n_digits):
        plt.subplot(6, 10, iteration + 1)
        plot_image(outputs_val[iteration])
    
#%%    Encode and Decode
def encode_decode():
    print( '''Encode''' )
    n_digits = 5
    codings = hidden3
    X, Y = CIFAR10().cifar10_input(n_digits,True)
    inputs_sample = X.get_next()
    input_array = tf.Session().run(inputs_sample).reshape(-1,IMG_SIZE*IMG_SIZE*3).astype(np.float32)  
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_variational.ckpt")
        codings_val = codings.eval(feed_dict={inputs: input_array})
    print( '''Decode''' )    
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_variational.ckpt")
        outputs_val = outputs.eval(feed_dict={codings: codings_val})
    fig = plt.figure(figsize=(8, 2.5 * n_digits))
    
    for iteration in range(n_digits):
        plt.subplot(n_digits, 2, 1 + 2 * iteration)
        plot_image(input_array[iteration])
        plt.subplot(n_digits, 2, 2 + 2 * iteration)
        plot_image(outputs_val[iteration])
    
#%% Interpolate digits¶
def interpolate_digits():
    print(""" Interpolate digits  """)
    n_iterations = 5
    n_digits = 6
    codings = hidden3

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_variational.ckpt")
        target_codings = np.roll(codings_rnd, -1, axis=0)
        plt.figure()
        for iteration in range(n_iterations + 1):
            codings_interpolate = codings_rnd + (target_codings - codings_rnd) * iteration / n_iterations
            outputs_val = outputs.eval(feed_dict={codings: codings_interpolate})
            for digit_index in range(n_digits):
                plt.subplot(n_iterations + 1, n_digits, digit_index + 1 + (n_digits)*iteration)
                plot_image(outputs_val[digit_index])

#%%
generate_digits()

encode_decode()
