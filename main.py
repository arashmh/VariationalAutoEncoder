import sys

from config import batch_size, latent_dim, modelsPath
from model import getModels
from visuals import visualizeDataset, visualizeReconstructedImages, computeLatentSpaceTSNEProjection, visualizeInterpolation, visualizeGeneratedImages, visualizeReconstructedVariations
from datasetTools import loadDataset

# Trains the VAE
def trainModel():
    # Create models
    vae, _, _ = getModels()
    vae.compile(optimizer='rmsprop', loss=VAELoss)

    X_train, X_test = loadDataset()
    # Train the VAE on dataset
    vae.fit(X_train, X_train, shuffle=True, nb_epoch=nbEpoch, batch_size=batch_size, validation_data=(X_test, X_test))

    # Serialize weights to HDF5
    print("Saving weights...")
    vae.save_weights(modelsPath+"model.h5")

# Generates images and plots
def testModel():
    # Create models
    print("Creating VAE, Encoder and Generator...")
    vae, encoder, generator = getModels()

    # Load VAE weights
    print("Loading weights...")
    vae.load_weights(modelsPath+"model.h5")

    # Load dataset to test
    print("Loading dataset...")
    _, X_test = loadDataset()

    #visualizeDataset(X_test)
    #visualizeReconstructedImages(X_test, vae)
    visualizeReconstructedVariations(X_test, vae)
    #computeLatentSpaceTSNEProjection(X_test, encoder, display=True)
    #visualizeInterpolation(X_test, encoder, generator)
    #visualizeGeneratedImages(generator, gridSize=5)

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




