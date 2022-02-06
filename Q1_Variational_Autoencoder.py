#!/usr/bin/env python
# coding: utf-8

# # Neural Networks and Deep Learning Spring 1400 <img src = 'https://ece.ut.ac.ir/cict-theme/images/footer-logo.png' alt="Tehran-University-Logo" width="150" height="150" align="right">
# ## Project 3 - Question 1
# ### By Omid Vaheb and Mahsa Masoud
# ### 810196582  -  810196635

# Variational Autoencoders or VAEs can be used to visualize high-dimensional data in a meaningful, lower-dimensional space. Following, we go over some details about autoencoding and VAEs and after that, we will construct and train a deep VAE on the MNIST dataset. We will see the data clusters in the lower-dimensional space regarding their classes. Plotting the test set data in this space shows where the images with unknown digit classes fall with respect to the known digit classes.

# ## Importing Required Libraries:

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import norm
from keras.datasets import mnist
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch 
import torchvision
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import sys


# In[13]:


import copy
import itertools
import json
import os
import warnings
from tensorflow.python.autograph.lang import directives
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import constants as sm_constants
from tensorflow.python.saved_model import loader_impl as sm_loader
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


# ## VAE:
# Autoencoding is a semi-supervised algorithm for data compression where the functions for compression and decompression are learned from the data. They are more likely being used to preprocess data and dimensionality reduction. In fact, the hidden layers of simple autoencoders are doing something like principal component analysis (PCA).
# Autoencoders generally have three parts: an encoder, a decoder, and a 'loss' function that maps one to the other. Usually this loss is the amount of information lost in the process of reconstruction. In training the autoencoder, we're optimizing the parameters of the neural networks to minimize the 'loss' using stochastic gradient descent.
# Variational autoencoders or VAEs don't learn to morph the data in and out of a compressed representation of itself instead, they learn the parameters of the probability distribution that the data came from which these parameters are mean and standard deviation. It's essentially an inference model and a generative model daisy-chained together.
# VAEs have received a lot of attention because of their generative ability. Since they learn about the distribution the inputs came from, we can sample from that distribution to generate novel data. As we'll see, VAEs can also be used to cluster data in useful ways.

# At first, we define a layer class with a callback function in which it uses mean and variance given from the last layer to sample from the data and samples data for generating new data. It constructs a term and adds it to the mean of the random variable. This term is equal to multiply of epsilon and exponential of 0.5*logarithim value of varaince in which epsilon is a random variable calculated using mean of Z.

# In[14]:


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Now, we construct the first part of VAE which is encoder and consists of two 2D convolutional layers after input layer and a flatten and dense layers afterwards. After all of these layers, we have the sampling layer which consistf of mean and logarithmic variance. Activation function of these layers are rectified linear unit and convolutional layers have padding and stride of 2. Also, latent space dimension is 2 as requested by the question.

# In[15]:


latent_dim = 2
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


# The next step is to construct decoder which is lie encoder but in reverse and it also has a 2D convolutional transposed layer to generate picture from outputs of last layers with sigmoid activation function.

# In[16]:


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# The final step is to build the VAE  using decoder and encoders written in the previous steps and defining loss fucntion. Metrics of this network are total loss, reconstruction loss and KL loss. Despite being trained in a semi-supervised way, the VAE algorithm entails minimizing a 'loss' function. Loss is actually two different losses combined, one that describes the difference between the input images and the images reconstructed from samples from the latent distribution, and another that is the difference between the latent distribution and the prior (the inputs). Our encoder and decoder are deep convnets constructed using the Keras Functional API. We'll need to separate the inputs from the labels, normalize them by dividing the max pixel value, and reshape them into 28x28 pixel images

# In[17]:


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def call(self, inputs, training=None, mask=None):

      raise NotImplementedError('When subclassing the `Model` class, you should '
                                'implement a `call` method.')
    def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):

      base_layer.keras_api_gauge.get_cell('predict').set(True)
      version_utils.disallow_legacy_graph('Model', 'predict')
      self._check_call_args('predict')
      _disallow_inside_tf_function('predict')

      if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        raise NotImplementedError('`model.predict` is not yet supported with '
                                  '`ParameterServerStrategy`.')

      outputs = None
      with self.distribute_strategy.scope():
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        dataset_types = (dataset_ops.DatasetV1, dataset_ops.DatasetV2)
        if (self._in_multi_worker_mode() or _is_tpu_multi_host(
            self.distribute_strategy)) and isinstance(x, dataset_types):
          try:
            options = dataset_ops.Options()
            data_option = distribute_options.AutoShardPolicy.DATA
            options.experimental_distribute.auto_shard_policy = data_option
            x = x.with_options(options)
          except ValueError:
            warnings.warn('Using Model.predict with '
                          'MultiWorkerDistributionStrategy or TPUStrategy and '
                          'AutoShardPolicy.FILE might lead to out-of-order result'
                          '. Consider setting it to AutoShardPolicy.DATA.')

        data_handler = data_adapter.get_data_handler(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            initial_epoch=0,
            epochs=1,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            model=self,
            steps_per_execution=self._steps_per_execution)

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
          callbacks = callbacks_module.CallbackList(
              callbacks,
              add_history=True,
              add_progbar=verbose != 0,
              model=self,
              verbose=verbose,
              epochs=1,
              steps=data_handler.inferred_steps)

        self.predict_function = self.make_predict_function()
        self._predict_counter.assign(0)
        callbacks.on_predict_begin()
        batch_outputs = None
        for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
          with data_handler.catch_stop_iteration():
            for step in data_handler.steps():
              callbacks.on_predict_batch_begin(step)
              tmp_batch_outputs = self.predict_function(iterator)
              if data_handler.should_sync:
                context.async_wait()
              batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
              if outputs is None:
                outputs = nest.map_structure(lambda batch_output: [batch_output],
                                            batch_outputs)
              else:
                nest.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs, batch_outputs)
              end_step = step + data_handler.step_increment
              callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
        if batch_outputs is None:
          raise ValueError('Expect x to be a non-empty array or dataset.')
        callbacks.on_predict_end()
      all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
      return tf_utils.sync_to_numpy_or_python_type(all_outputs)


# Now, we define a function to draw the pictures recovered in the latent space after the decoder. We can use the decoder network to take a peak at what samples from the latent space look like as we change the latent variables. What we end up with is a smoothly varying space where each digit transforms into the others as we dial the latent variables up and down

# In[18]:


def plot_latent_space(vae, n=10, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()       


# Also, we define a function to plot 2D plots of data regarding their digit classes in the lower-dimensional space in order to observe data in this space and finding the range of each class in both dimensions in this space.

# In[19]:


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


# In this step, we load data and compile the model and train it with the available data. Batch size is set to 128 and we train model for 100 epochs in which it does not improve afterwards. Optimizer is set to adam and we save loss values and outputs of model in each stage.

# In[20]:


(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255 # Each Pixel is between 0 and 255--> to normalize between 0 and 1
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())


# In[21]:


history = vae.fit(mnist_digits, epochs=10, batch_size=128)
loss1 = history.history['loss']
x_pred = vae.decoder.predict(vae.encoder.predict(mnist_digits[0:100])[2])
plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(x_pred[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
history = vae.fit(mnist_digits, epochs=10, batch_size=128)
loss1 += history.history['loss']
x_pred = vae.decoder.predict(vae.encoder.predict(mnist_digits[0:100])[2])
plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(x_pred[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
history = vae.fit(mnist_digits, epochs=10, batch_size=128)
loss1 += history.history['loss']
x_pred = vae.decoder.predict(vae.encoder.predict(mnist_digits[0:100])[2])
plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(x_pred[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
history = vae.fit(mnist_digits, epochs=10, batch_size=128)
loss1 += history.history['loss']
x_pred = vae.decoder.predict(vae.encoder.predict(mnist_digits[0:100])[2])
plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(x_pred[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
history = vae.fit(mnist_digits, epochs=10, batch_size=128)
loss1 += history.history['loss']
x_pred = vae.decoder.predict(vae.encoder.predict(mnist_digits[0:100])[2])
plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(x_pred[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()


# In[22]:


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss1, 'darkblue')
plt.grid(c='lightgrey')
plt.show()


# Now, we plot clusters of data in the latent space. We can make predictions on the validation set using the encoder network. This has the effect of translating the images from the 784-dimensional input space into the 2-dimensional latent space. When we color-code those translated data points according to their known digit class, we can see how the digits cluster together

# In[23]:


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
plot_label_clusters(vae, x_train, y_train)


# In[24]:


plot_latent_space(vae)


# In[25]:


plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(mnist_digits[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')


# In[26]:


x_pred = vae.decoder.predict(vae.encoder.predict(mnist_digits[0:100])[2])
plt.figure(figsize=(10,10))
for i in range(10):
  for j in range(10):
    plt.subplot(10, 10,10 * i + j + 1)
    plt.imshow(x_pred[10 * i + j].squeeze(), cmap='gray')
    plt.axis('off')


# ## Conditional VAE

# ## Parameter Initialization

# In[27]:


batch_size = 100
learning_rate = 1e-3
max_epoch = 40
device = torch.device("cuda")
num_workers = 5
load_epoch = -1
generate = True


# ## Model Definition
# In this part we define the model and use code and theory in the paper to implement the conditional autoencoder.

# In[28]:


class Model(nn.Module):
    def __init__(self,latent_size=2,num_classes=10):
        super(Model,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes

        # For encode
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(4*4*32,300)
        self.mu = nn.Linear(300, self.latent_size)
        self.logvar = nn.Linear(300, self.latent_size)

        # For decoder
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300,4*4*32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(1, 1, kernel_size=4)

    def encoder(self,x,y):
        y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        y = torch.ones(x.shape).to(device)*y
        t = torch.cat((x,y),dim=1)
        
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.reshape((x.shape[0], -1))
        
        t = F.relu(self.linear1(t))
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps*std + mu

    def unFlatten(self, x):
        return x.reshape((x.shape[0], 32, 4, 4))

    def decoder(self, z):
        t = F.relu(self.linear2(z))
        t = F.relu(self.linear3(t))
        t = self.unFlatten(t)
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = F.relu(self.conv5(t))
        return t
        
    def forward(self, x, y):
        mu, logvar = self.encoder(x,y)
        z = self.reparameterize(mu,logvar)
        # Class conditioning
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar


# Now, we write a function to plot and save images of prediction and input.

# In[29]:


def plot(epoch, pred, y,name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16,16))
    for i in range(6):
        ax = fig.add_subplot(3,2,i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
        ax.title.set_text(str(y[i]))
    plt.savefig("./images/{}epoch_{}.jpg".format(name, epoch))
    plt.figure(figsize=(10,10))
    plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


# A fnction is defined to clculate loss of model.

# In[30]:


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld


# Train and test methods are written using pytorch library.

# In[31]:


def train(epoch, model, train_loader, optim):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i,(x,y) in enumerate(train_loader):
        try:
            label = np.zeros((x.shape[0], 10))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)
            optim.zero_grad()   
            pred, mu, logvar = model(x.to(device),label.to(device))  
            recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optim.step()
            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]
            if i == 0:
                print("Gradients")
                for name,param in model.named_parameters():
                    if "bias" in name:
                        print(name,param.grad[0],end=" ")
                    else:
                        print(name,param.grad[0,0],end=" ")
                    print()
        except Exception as e:
            traceback.print_exe()
            torch.cuda.empty_cache()
            continue
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss


# In[32]:


def test(epoch, model, test_loader):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                label = np.zeros((x.shape[0], 10))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)
                pred, mu, logvar = model(x.to(device),label.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
                loss = recon_loss + kld
                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
            except Exception as e:
                traceback.print_exe()
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        


# We also, write a function to generate images by getting input from lateent space. This function uses the plot function that we wrote previously.

# In[33]:


def generate_image(epoch,z, y, model):
    with torch.no_grad():
        label = np.zeros((y.shape[0], 10))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)
        pred = model.decoder(torch.cat((z.to(device),label.float().to(device)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy(),name='Eval_')
        print("data Plotted")


# It is also required to write a data loader for this problem.

# In[34]:


def load_data():
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return train_loader, test_loader


# In[35]:


def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)


# Now, we train and test the data.

# In[36]:


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    print("dataloader created")
    model = Model().to(device)
    print("model created")
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("model {} loaded".format(load_epoch))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch+1, max_epoch):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader)
            if generate:
                z = torch.randn(6, 32).to(device)
                y = torch.tensor([1,2,3,4,5,6]) - 1
                generate_image(i,z, y, model) 
        print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss:{}".format(i, max_epoch,train_total, train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kld, test_loss))
        save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))


# In[37]:


i, (example_data, exaple_target) = next(enumerate(test_loader))
print(example_data[0,0].shape)
plt.figure(figsize=(5,5), dpi=100)
plt.imsave("example.jpg", example_data[0,0], cmap='gray',  dpi=1000)
plt.imshow(example_data[0,0], cmap='gray')
plt.show()


# In[38]:


test_loss_list


# In[39]:


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    print("dataloader created")
    model = Model().to(device)
    print("model created")
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("model {} loaded".format(load_epoch))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch+1, max_epoch):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader)
            if generate:
                z = torch.randn(6, 32).to(device)
                y = torch.tensor([1,2,3,4,5,6]) - 1
                generate_image(i,z, y, model)   
        print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss:{}".format(i, max_epoch,train_total, train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kld, test_loss))

        save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))


# In[41]:


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    preds,z_mean, _ = vae.forward(data,labels)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


# In[45]:


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
plot_label_clusters(model, x_train, y_train)


# Ploting Loss Curve

# In[45]:


for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        running_loss =+ loss.item() * images.size(0)

    loss_values.append(running_loss / len(train_dataset))

plt.plot(loss_values)

