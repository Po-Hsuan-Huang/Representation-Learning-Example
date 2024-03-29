#!/usr/bin/env python3
# -*- coding: utf-8 -*-




"""
Created on Tue Apr 10 13:18:22 2018

@author: pohsuanh

The code trains a variational autoencoder with CIFAR10 Dataset. 

Training : 
    
    Run the session 

Evaluation :   
    
    Generate_digits() : generate images from ramdom samples in latent space.
    
    Encode_decode() : encode test image and decode it.
    
    Interpolate() : interpolate in the latent space to generate an continum of images. 
    
    
"""
import tensorflow as tf

import matplotlib.pyplot as plt

import sys, os

from functools import partial

import numpy as np

import pickle

import time

# Parameters

BATCH_SIZE = 64

NUM_EXAMPLES = 10000

IMG_SIZE = 24

N_EPOCHS = 15

SAVED_PATH = "/home/pohsuanh/Documents/Schweighofer Lab/my_model_variational.ckpt"

SAVED_DATASET_PATH = "/home/pohsuanh/Documents/Schweighofer Lab/cifar10_dataset.pickle"


# Number of digits to print in evaluation

N_DIGITS =20


class NMIST(object):

    def __init__(self):

        from tensorflow.examples.tutorials.mnist import input_data

        self.mnist = input_data.read_data_sets("/tmp/data/")

        global NUM_EXAMPLES, IMG_SIZE, BATCH_SIZE

        self.NUM_EXAMPLES = self.mnist.train.num_examples

        self.IMG_SIZE = 28

    def mnist_data(self,batch_size=BATCH_SIZE):

        # load and return data and label X_batch, Y_batch

        return self.mnist.train.next_batch(batch_size)
    
class CIFAR10(object):

    def __init__(self, normalize= True):
        
        global NUM_EXAMPLES, IMG_SIZE, BATCH_SIZE

        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        
        self.x_train = self.x_train[ : NUM_EXAMPLES]

        self.y_train = self.y_train[ : NUM_EXAMPLES]

        self.x_test = self.x_test[ : N_DIGITS]

        self.y_test = self.y_test[ : N_DIGITS]  
        
        self.x_train_resize = np.empty( [NUM_EXAMPLES, IMG_SIZE, IMG_SIZE, 3], dtype = np.float16)
        
        self.x_test_resize  =np.empty( [N_DIGITS, IMG_SIZE, IMG_SIZE, 3], dtype = np.float16)
         
    def _cifar10_data(self, batch_size=BATCH_SIZE, _eval=False):

        # laod and return data and label X_batch, Y_batch

        if not _eval : # laod trainset
            
            with tf.Session( ) as sess:

                self.x_train = self.x_train.astype(np.float16)
    
                for i , img in enumerate(self.x_train) :
                    
                    img = tf.image.resize_image_with_crop_or_pad( img, IMG_SIZE, IMG_SIZE)
                            
                    img_norm = img/256
    
                    self.x_train_resize[i] = img_norm.eval()
            
                    X_batch = tf.data.Dataset.from_tensor_slices(self.x_train_resize).repeat(N_EPOCHS).batch(batch_size)

#          Y_batch = tf.data.Dataset.from_tensor_slices( self.y_train ).repeat().batch(batch_size)

            return X_batch 

        else:
            with tf.Session( ) as sess:

                self.x_test = self.x_test.astype(np.float16)
    
                for i , img in enumerate(self.x_test) :
                    
                    img = tf.image.resize_image_with_crop_or_pad( img, IMG_SIZE, IMG_SIZE)
                                                            
                    img_norm = img/256

                    self.x_test_resize[i] = img_norm.eval()   

                    X_batch = tf.data.Dataset.from_tensor_slices( self.x_test_resize).repeat(N_EPOCHS).batch(batch_size)
          
#          Y_batch = tf.data.Dataset.from_tensor_slices( self.y_test ).repeat().batch(batch_size)
          
            return X_batch
      
    def cifar10_input(self, batch_size=BATCH_SIZE, _eval=False):

         X_batch = self._cifar10_data(batch_size,_eval)

         return X_batch.make_one_shot_iterator()

     
def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)    
    
def load_data(saved_dataset_path = SAVED_DATASET_PATH):
        
    if os.path.isfile(SAVED_DATASET_PATH):
    
        print('load dataset from pikcle...')
    
        with open(SAVED_DATASET_PATH, 'rb') as handle:
    
            dataset = pickle.load(handle)
    
    else :
    
        print('download and preprocess dataset')
    
        dataset = CIFAR10()
        
        with open(SAVED_DATASET_PATH, 'wb') as handle:
        
            pickle.dump( dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return dataset
#%%
time_start = time.time()    

reset_graph()    

cifar10 = load_data()

time_loaddata = time.time()

X = cifar10.cifar10_input()

X_batch = X.get_next()

n_inputs = IMG_SIZE * IMG_SIZE * 3

n_hidden1 = 400

n_hidden2 = 100

n_hidden3 = 20  # codings

n_hidden4 = n_hidden2

n_hidden5 = n_hidden1

n_outputs = n_inputs

## initial_lr = 10**-4 (<60 epochs)
learning_rate = 10**-3

initializer = tf.contrib.layers.variance_scaling_initializer()

my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)

print('constructing graph ...')

time_construct_grahp = time.time()

inputs = tf.cast(X_batch, tf.float16)

inputs = tf.reshape(inputs,[-1,IMG_SIZE * IMG_SIZE*3])

hidden1 = my_dense_layer(inputs, n_hidden1)

hidden2 = my_dense_layer(hidden1, n_hidden2)

hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)

hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)

noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float16)

hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise

hidden4 = my_dense_layer(hidden3, n_hidden4)

hidden5 = my_dense_layer(hidden4, n_hidden5)

logits = my_dense_layer(hidden5, n_outputs, activation=None)

print_logit = tf.Print(logits,[hidden1],'output_logit')

outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=logits)

reconstruction_loss = tf.reduce_sum(xentropy)

latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)

loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

tfconfig.gpu_options.allow_growth = True


with tf.Session(config = tfconfig) as sess:

    time_run_session = time.time()
    
    print('run session...')
    
    try :

        print('load session checkpoint...')

        saver.restore(sess,SAVED_PATH)

    except tf.errors.NotFoundError:

        print('initailize graph variables')

        init.run()

    for epoch in range(N_EPOCHS):

        n_batches = NUM_EXAMPLES // BATCH_SIZE

        for iteration in range(n_batches):

            print("\r{}%".format(100 * iteration // n_batches), end="")

            sys.stdout.flush()

            sess.run(training_op)

        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss])

        print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
    
    saver.save(sess, SAVED_PATH)


    #(    """     Generate digits from random latent state samples      """     )

    codings_rnd = np.random.normal(size=[N_DIGITS, n_hidden3])

    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

#%%

def plot_image(image, shape=[32,32,3], colors="gray"):

    image = image.reshape(shape)

    plt.imshow(image, cmap=colors, interpolation="nearest")

    plt.axis("off")
    
def plot_multiple_images(images, n_rows, n_cols, pad=2):

    images = images - images.min()  # make the minimum == 0, so the padding looks white

    plt.figure(figsize=(20, 2.5 * N_DIGITS//5)) # not shown in the book

    for iteration in range(N_DIGITS):

        plt.subplot(N_DIGITS//5, 5, iteration + 1)

        plot_image(images[iteration])
#%%

def generate_digits() :
    print("Generate digits")
    global outputs_eval
    global N_DIGITS
    
    plt.figure(figsize=(20, 2.5 * N_DIGITS//5)) # not shown in the book
    for iteration in range(N_DIGITS):
        plt.subplot(N_DIGITS//5, 5, iteration + 1)
        plot_image(outputs_val[iteration])
    
#%%    Encode and Decode
def encode_decode():
    print( '''Encode''' )
    codings = hidden3
    X, Y = cifar10.cifar10_input(N_DIGITS,True)
    inputs_sample = X.get_next()
    input_array = tf.Session().run(inputs_sample)
    
    print("input images")
    plot_multiple_images(input_array.astype(np.float32),4,5)
    

    with tf.Session() as sess:
        saver.restore(sess, SAVED_PATH)
        input_array = input_array.reshape(-1,IMG_SIZE*IMG_SIZE*3).astype(np.float16)  
        codings_val = codings.eval(feed_dict={inputs: input_array})
    
    print( '''Decode''' )    
    with tf.Session() as sess:
        saver.restore(sess, SAVED_PATH)
        outputs_val = outputs.eval(feed_dict={codings: codings_val})
    
    print("output images")
    plot_multiple_images(outputs_val,4,5)
    
#%% 
def interpolate_digits():

    print(""" Interpolate digits  """)

    n_iterations = 5

    N_DIGITS = 6

    codings = hidden3

    codings_rnd = np.random.normal(size=[N_DIGITS, n_hidden3])
    
    with tf.Session() as sess:

        saver.restore(sess, "./my_model_variational.ckpt")

        target_codings = np.roll(codings_rnd, -1, axis=0)

        plt.figure()

        for iteration in range(n_iterations + 1):

            codings_interpolate = codings_rnd + (target_codings - codings_rnd) * iteration / n_iterations

            outputs_val = outputs.eval(feed_dict={codings: codings_interpolate})

            for digit_index in range(N_DIGITS):

                plt.subplot(n_iterations + 1, N_DIGITS, digit_index + 1 + (N_DIGITS)*iteration)

                plot_image(outputs_val[digit_index])

#%%    
time_eval = time.time()
encode_decode()

print("load time ", time_loaddata - time_start)
print("graph construct time ", time_construct_grahp - time_loaddata)
print("run session time ", time_run_session - time_construct_grahp)
print("eval time ", time_eval - time_run_session)
