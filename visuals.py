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

from model import getModels
from datasetTools import loadDataset
from config import batch_size

# Scatter with images instead of points
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

# Show dataset images with T-sne projection of latent space encoding
def computeLatentSpaceTSNEProjection(X, encoder, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder.predict(X, batch_size=batch_size)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.1)
        plt.show()
    else:
        return X_tsne
    
def visualizeGeneratedImages(generator, gridSize=3):
    # Display a 2D manifold of the images
    imageDisplaySize = 64
    figure = np.zeros((imageDisplaySize * gridSize, imageDisplaySize * gridSize, 3))
    
    # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, gridSize))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, gridSize))

    # We walk through the dimension 0 and 1 of the latent space
    # Note: these might not be the most relevant
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros(latent_dim)
            z_sample[0] = xi
            z_sample[1] = yi
            z_sample = np.tile(z_sample,batch_size)
            z_sample = z_sample.reshape([batch_size, latent_dim])
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            image = cv2.resize(x_decoded[0],(imageDisplaySize,imageDisplaySize))
            figure[i * imageDisplaySize: (i + 1) * imageDisplaySize, j * imageDisplaySize: (j + 1) * imageDisplaySize] = image

    plt.figure(figsize=(5, 5))
    plt.imshow(figure)
    plt.show()

def visualizeReconstructedImages(X, vae):
    # Crop X
    X = X[:10]
    print("Generating 10 image reconstructions...")
    reconstructedX = vae.predict(X)

    result = None
    for i in range(len(X)/2-1):
        img = X[2*i]
        reconstruction = reconstructedX[2*i]
        img2 = X[2*i+1]
        reconstruction2 = reconstructedX[2*i+1]
        image = np.hstack([img,reconstruction,img2,reconstruction2])*255.
        image = image.astype(np.uint8)
        result = image if result is None else np.vstack([result,image])

    cv2.imshow("LOL",result)
    cv2.waitKey()

# Show every image, good for showing interplation candidates
def visualizeDataset(X):
    for i,image in enumerate(X):
        image = (image*255).astype(np.uint8)
        cv2.imshow(str(i),image)
        cv2.waitKey()

# Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(X, encoder, generator):
    # Extract random indices
    startIndex = 2
    endIndex = 53
    X0 = np.zeros([batch_size,imageSize,imageSize,3])
    X0[0] = X[startIndex]
    X0[1] = X[endIndex]
    X = X0

    print("Generating 10 image reconstructions...")
    latentX = encoder.predict(X)
    latentStart, latentEnd = latentX[0], latentX[1]

    # True image for comparison
    startImage = X[0]
    endImage = X[1]

    vectors = []
    normalImages = []

    #Linear interpolation
    alphaValues = np.linspace(0, 1, 10)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart*(1-alpha) + latentEnd*alpha
        vectors.append(vector)
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage,1-alpha,endImage,alpha,0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = generator.predict(vectors)

    # Put final image together
    resultLatent = None
    resultImage = None
    for i in range(len(reconstructions)):
        interpolatedImage = normalImages[i]*255
        interpolatedImage = interpolatedImage.astype(np.uint8)
        resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage,interpolatedImage])

        reconstructedImage = reconstructions[i]*255.
        reconstructedImage = reconstructedImage.astype(np.uint8)
        resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent,reconstructedImage])
    
        result = np.vstack([resultImage,resultLatent])

    cv2.imshow("Interpolation in Image Space vs Latent Space",result)
    cv2.waitKey()



