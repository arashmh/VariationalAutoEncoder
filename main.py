import sys

from config import latent_dim, modelsPath, imageSize
from keras.callbacks import TensorBoard
from model import getModels, VAELoss
from visuals import visualizeDataset, visualizeReconstructedImages, computeLatentSpaceTSNEProjection, visualizeInterpolation, visualizeReconstructedVariations
from datasetTools import loadDataset

nbEpoch = 3
batchSize = 128

# Trains the VAE
def trainModel():
    # Create models
    print("Creating VAE...")
    vae, _, _ = getModels()
    vae.compile(optimizer='rmsprop', loss=VAELoss)

    print("Loading dataset...")
    X_train, X_test = loadDataset()
    X_train = X_train
    X_test = X_test

    # Train the VAE on dataset
    print("Training VAE...")
    runID = "VAE - ZZZ"
    tb = TensorBoard(log_dir='/tmp/logs/'+runID)
    vae.fit(X_train, X_train, shuffle=True, nb_epoch=nbEpoch, batch_size=batchSize, validation_data=(X_test, X_test), callbacks=[tb])

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
    #visualizeReconstructedVariations(X_test, vae)
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




