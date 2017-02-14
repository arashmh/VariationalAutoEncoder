import os
import sys
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import objectives

batch_size = 10
original_dim = 784
latent_dim = 10
intermediate_dim = 128
nbEpoch = 1
imageSize = 128


nb_filters = 64
nb_conv = 3

# Global for vaeLoss
z_mean = None
z_log_var = None

# Loss function used for VAE
def VAELoss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = imageSize * imageSize * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# Samples a vector with given mean and variance
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., std=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Loads the dataset
def loadDataset():
    if os.path.isfile("coins.h5"):
        # Load hdf5 dataset
        h5f = h5py.File("coins.h5", 'r')
        X_train = h5f['X']
        return X_train[:10], X_train[-10:]
    else:
        #We don't generate the dataset in this example
        print "[!] No dataset found (coins.h5)"
        return None

# Convolutional models
def getModels():
    global z_mean
    global z_log_var

    # Define layers 
    x = Input(batch_shape=(batch_size, imageSize, imageSize, 3))

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
    z = Lambda(sampling)([z_mean, z_log_var])

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

# Trains the VAE
def trainModel():

    for model in getModels():
        print model.summary()

    # Create models
    vae, _, _ = getModels()
    vae.compile(optimizer='rmsprop', loss=VAELoss)

    X_train, X_test = loadDataset()
    # Train the VAE on dataset
    vae.fit(X_train, X_train, shuffle=True, nb_epoch=nbEpoch, batch_size=batch_size, validation_data=(X_test, X_test))

    # Serialize weights to HDF5
    print("Saving weights...")
    vae.save_weights("model.h5")

# Generates images and plots
def testModel():
    # Create models
    vae, encoder, generator = getModels()

    # Load VAE weights
    print("Loading weights...")
    vae.load_weights("model.h5")

    # Load dataset to test
    _, X_test = loadDataset()


    # Display a 2D plot of the images in the latent space
    X_test_encoded = encoder.predict(X_test, batch_size=batch_size)
    fig, ax = plt.subplots()
    imscatter(X_test_encoded[:, 0], X_test_encoded[:, 1], imageData=X_test, ax=ax, zoom=0.7)
    plt.show()
    return

    # Display a 2D manifold of the images
    gridSize = 3
    imageDisplaySize = 28
    figure = np.zeros((imageDisplaySize * gridSize, imageDisplaySize * gridSize))
    
    # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, gridSize))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, gridSize))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            image = x_decoded[0].reshape(imageDisplaySize, imageDisplaySize)
            figure[i * imageDisplaySize: (i + 1) * imageDisplaySize, j * imageDisplaySize: (j + 1) * imageDisplaySize] = image

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def main():
    x = np.linspace(0, 10, 20)
    y = np.cos(x)
    fig, ax = plt.subplots()
    imscatter(x, y, ax=ax, zoom=0.2)
    plt.show()

def imscatter(x, y, ax, imageData, zoom=1):
    images = []

    for i in range(len(x)):
        x0 = x[i]
        y0 = y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8)
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()





if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) == 2 else None 
    if arg is None:
        print "Need argument"
    elif arg == "train":
        trainModel()
    elif arg == "test":
        testModel()
    else:
        print "Wrong argument"




