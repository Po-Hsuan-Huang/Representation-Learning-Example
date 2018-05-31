
"""
Created on Wed Apr 18 03:19:59 2018

@author: pohsuanhuang

Variational Autoencoder on CIFAR dataset

"""
import numpy as np
from matplotlib import pylab as plt
import tensorflow as tf
import sys, os
import tarfile
from six.moves import urllib
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/pohsuanh/Documents/Schweighofer Lab/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")



# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# PLOT OPTION
GENERATE_IMAGES = False
ENCODER_DECODER = False
INTERPOLATE = False

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
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
    
#%%
from functools import partial
reset_graph()
maybe_download_and_extract()
n_inputs = 24 * 24 * 3
n_hidden1 = 500
n_hidden2 = 100
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()

my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)
with tf.device('/cpu:0'):
    X_batch, Y_batch = distorted_inputs(False) 
X = tf.reshape(X_batch,[-1, 24*24*3])
hidden1 = my_dense_layer( X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_sigma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels= X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
eps = 1e-10 # smoothing term to avoid computing log(0) which is NaN
latent_loss = 0.5 * tf.reduce_sum(
    tf.square(hidden3_sigma) + tf.square(hidden3_mean)
    - 1 - tf.log(eps + tf.square(hidden3_sigma)))

loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_digits = 60
n_epochs = 5
#batch_size = 128
#%%
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size
        print("epoch :",epoch)
        for iteration in range(n_batches):
            sys.stdout.flush()
            print("iteration :",iteration)
            X = tf.reshape(X_batch,[-1, 24*24*3])
            sess.run(training_op)
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss])
        print ("\r",epoch, "Train total loss:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
        saver.save(sess, "./my_model_variational.ckpt")


    print(    """     Generate digits      """     )

    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})


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
    n_digits = 30
    X_test, y_test = inputs(1)
    codings = hidden3
    
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_variational.ckpt")
        codings_val = codings.eval(feed_dict={X: X_test})
    print( '''Decode''' )    
    with tf.Session() as sess:
        saver.restore(sess, "./my_model_variational.ckpt")
        outputs_val = outputs.eval(feed_dict={codings: codings_val})
    fig = plt.figure(figsize=(8, 2.5 * n_digits))
    
    for iteration in range(n_digits):
        plt.subplot(n_digits, 2, 1 + 2 * iteration)
        plot_image(X_test[iteration])
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
