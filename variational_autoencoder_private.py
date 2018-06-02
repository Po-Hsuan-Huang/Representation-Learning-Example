#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:24:54 2018

@author: pohsuanh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-




"""
Created on Tue Apr 10 13:18:22 2018

@author: pohsuanh


The code trains a variational autoencoder with CIFAR10 Dataset. The 
"""
import tensorflow as tf

#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

import matplotlib.pyplot as plt
import os, sys
from functools import partial
import numpy as np

# Basic model parameters.

BATCH_SIZE = 64

NUM_EXAMPLES = 10000

IMG_SIZE = 28

n_epochs = 15

saved_path = "/home/pohsuanh/Documents/Schweighofer Lab/my_model_variational.ckpt"

learning_rate = 10**-4

n_digits =20


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
    def __init__(self, normalize= True):
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        global NUM_EXAMPLES, IMG_SIZE, BATCH_SIZE
        IMG_SIZE = 32
        NUM_EXAMPLES = self.x_train.shape[0]
        if normalize :
            self.x_train = self.x_train.astype(np.float16)
            self.x_test = self.x_test.astype(np.float16)
            # Image Normalization
            for i , img in enumerate(self.x_train) :
#                img_norm = img - np.mean(img)
#                img_norm = img_norm/(np.max(img_norm)-np.min(img_norm))
                img_norm = img/np.max(img)
                self.x_train[i] = img_norm
                
            for i , img in enumerate(self.x_test) :
#                img_norm = img - np.mean(img)
#                img_norm = img_norm/(np.max(img_norm)-np.min(img_norm))
                img_norm = img/np.max(img)
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

X, Y = cifar10.cifar10_input()
X_batch, Y_batch = X.get_next(), Y.get_next()
n_inputs = IMG_SIZE * IMG_SIZE * 3
n_hidden1 = 400
n_hidden2 = 100
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

initializer = tf.contrib.layers.variance_scaling_initializer()
my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)
print('constructing graph ...')
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

tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
tfconfig.gpu_options.allow_growth = True
with tf.Session(config = tfconfig) as sess:

    print('run session...')
   # writer = tf.summary.FileWriter("/home/pohsuanh/Documents/Schweighofer Lab/tensorgraphs", sess.graph)
    
    try :
        print('load session checkpoint...')
        saver.restore(sess,saved_path)
    except tf.errors.NotFoundError:
        print('initailize graph variables')
        init.run()

    for epoch in range(n_epochs):
        n_batches = NUM_EXAMPLES // BATCH_SIZE
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            sess.run(training_op)

        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss])
        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
    
    saver.save(sess, saved_path)
    #writer.close()


    #(    """     Generate digits from random latent state samples      """     )

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})
#%%
def plot_image(image, shape=[32,32,3], colors="gray"):
    # NMINST shape: [32,32] colors: 'Grey'
    # restore from [0,1] for floats to [0, 255] for integers
#    if image.dtype == np.int8 :
#        pass
#    elif image.dtype == np.float64 :
#        image = image.reshape(shape)*1/( np.max(image)-np.min(image) )
#        image = image + (np.max(image)-np.min(image))*0.5
#        image = image.astype(np.int8)
    image = image.reshape(shape)
    plt.imshow(image, cmap=colors, interpolation="nearest")
    plt.axis("off")
    
def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    plt.figure(figsize=(20, 2.5 * n_digits//5)) # not shown in the book
    for iteration in range(n_digits):
        plt.subplot(n_digits//5, 5, iteration + 1)
        plot_image(images[iteration])
#%% """     Generate digits      """ 
def generate_digits() :
    print("Generate digits")
    global outputs_eval
    global n_digits
    
    plt.figure(figsize=(20, 2.5 * n_digits//5)) # not shown in the book
    for iteration in range(n_digits):
        plt.subplot(n_digits//5, 5, iteration + 1)
        plot_image(outputs_val[iteration])
    
#%%    Encode and Decode
def encode_decode():
    print( '''Encode''' )
    codings = hidden3
    X, Y = cifar10.cifar10_input(n_digits,True)
    inputs_sample = X.get_next()
    input_array = tf.Session().run(inputs_sample)
    
    print("input images")
    plot_multiple_images(input_array.astype(np.float32),4,5)
    

    with tf.Session() as sess:
        saver.restore(sess, saved_path)
        input_array = input_array.reshape(-1,IMG_SIZE*IMG_SIZE*3).astype(np.float16)  
        codings_val = codings.eval(feed_dict={inputs: input_array})
    
    print( '''Decode''' )    
    with tf.Session() as sess:
        saver.restore(sess, saved_path)
        outputs_val = outputs.eval(feed_dict={codings: codings_val})
    
    print("output images")
    plot_multiple_images(outputs_val,4,5)
    
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
print("input images")
cifar10_eval = CIFAR10(False).x_test[:n_digits]
plot_multiple_images(cifar10_eval,4,5)

#%%
print("input images normalize and restore")
images, label = CIFAR10().cifar10_input(n_digits,True)
inputs_sample = images.get_next()
input_array = tf.Session().run(inputs_sample)
plot_multiple_images(input_array,4,5)

#%%    

encode_decode()
