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

from model import getModels
from config import latent_dim, imageSize

# Show every image, good for showing interplation candidates
def visualizeDataset(X):
    for i,image in enumerate(X):
        image = (image*255).astype(np.uint8)
        cv2.imshow(str(i),image)
        cv2.waitKey()

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom=1):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8)
        # OpenCV uses BGR and plt uses RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOnLatentSpace(X, encoder, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder.predict(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.15)
        plt.show()
    else:
        return X_tsne

# Show dataset images with T-sne projection of pixel space
def computeTSNEProjectionOnPixelSpace(X, display=True):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1,imageSize*imageSize*3]))

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.15)
        plt.show()
    else:
        return X_tsne
    
# Reconstructions for samples in dataset
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

    cv2.imshow("Reconstructed images",result)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Variations according to sampling
def visualizeReconstructedVariations(X, vae):
    exampleIndex = 2#53
    print("Generating image reconstructions...")
    while True:
        # Micro batch
        imageData = np.array([X[exampleIndex]])
        reconstructedImage = vae.predict(imageData)
        image = np.hstack([imageData[0],reconstructedImage[0]])*255.
        image = image.astype(np.uint8)
        cv2.imshow("Reconstructed image",image)
        cv2.waitKey()

# Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(X, encoder, generator):
    print("Generating interpolations...")

    startIndex = 2
    endIndex = 53
    # Extract random indices
    X = np.array([X[startIndex], X[endIndex]])

    # Compute latent space projection
    latentX = encoder.predict(X)
    latentStart, latentEnd = latentX

    # Get original image for comparison
    startImage, endImage = X

    vectors = []
    normalImages = []
    #Linear interpolation
    alphaValues = np.linspace(0, 1, 6)
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
    cv2.destroyAllWindows()



