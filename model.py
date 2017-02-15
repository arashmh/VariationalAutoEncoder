import os
import sys
import h5py
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import norm
from sklearn import manifold

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import objectives
from config import latent_dim

from samplingLayer import SamplingLayer

original_dim = 784
intermediate_dim = 128
imageSize = 128

nb_filters = 64
nb_conv = 3

# Global for vaeLoss
z_mean = None
z_log_var = None

# Loss function used for VAE
def VAELoss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batchSize by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = imageSize * imageSize * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# Convolutional models
def getModels():
    global z_mean
    global z_log_var

    # Define layers 
    x = Input(batch_shape=(None, imageSize, imageSize, 3))

    # Conv + Pool
    conv_1 = Convolution2D(8, 1, 1, border_mode='same', activation='relu')(x)
    conv_2 = Convolution2D(16, 3, 3,   border_mode='same', activation='relu', subsample=(2, 2))(conv_1)
    conv_3 = Convolution2D(32, 3, 3,   border_mode='same', activation='relu', subsample=(2, 2))(conv_2)
    conv_4 = Convolution2D(64, 3, 3,   border_mode='same', activation='relu', subsample=(2, 2))(conv_3)
    
    # Flatten + Fully connected
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    # Fully connected for Mean / Var
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    # Sample Z from latent space distribution
    z = SamplingLayer(latent_dim)([z_mean,z_log_var]) #Lambda(sampling)([z_mean, z_log_var])

    # We instantiate these layers separately so as to reuse them later
    # Dense from latent space to image dimension
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(64 * 16 * 16, activation='relu')

    # Reshape for image
    decoder_reshape = Reshape((16, 16, 64))

    # Upsample + Conv
    decoder_upsampl_1 = UpSampling2D(size=(2, 2))
    decoder_conv_1 = Convolution2D(32, 3, 3, border_mode='same', activation="relu")
    decoder_upsampl_2 = UpSampling2D(size=(2, 2))
    decoder_conv_2 = Convolution2D(32, 3, 3, border_mode='same', activation="relu")
    decoder_upsampl_3 = UpSampling2D(size=(2, 2))

    # Convert to RGB with last Conv
    decoder_mean_squash = Convolution2D(3, 2, 2, border_mode='same', activation='tanh')

    # Build second part of model
    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)

    # Add Upsample + Conv
    upsampl_1_decoded = decoder_upsampl_1(reshape_decoded)
    conv_1_decoded = decoder_conv_1(upsampl_1_decoded)

    upsampl_2_decoded = decoder_upsampl_2(conv_1_decoded)
    conv_2_decoded = decoder_conv_2(upsampl_2_decoded)

    upsampl_3_decoded = decoder_upsampl_3(conv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(upsampl_3_decoded)

    # Build full VAE
    vae = Model(x, x_decoded_mean_squash)

    # Build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # Build an image generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)

    # Upsample + Conv for generator
    _upsampl_1_decoded = decoder_upsampl_1(_reshape_decoded)
    _conv_1_decoded = decoder_conv_1(_upsampl_1_decoded)

    _upsampl_2_decoded = decoder_upsampl_2(_conv_1_decoded)
    _conv_2_decoded = decoder_conv_2(_upsampl_2_decoded)

    _upsampl_3_decoded = decoder_upsampl_3(_conv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_upsampl_3_decoded)

    # Build generator
    generator = Model(decoder_input, _x_decoded_mean_squash)
    return vae, encoder, generator

