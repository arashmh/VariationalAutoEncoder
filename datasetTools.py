import os
import h5py
from config import datasetsPath

# Loads the dataset
def loadDataset():
    if os.path.isfile(datasetsPath+"coins.h5"):
        # Load hdf5 dataset
        h5f = h5py.File(datasetsPath+"coins.h5", 'r')
        X_train = h5f['X_train']
        X_test = h5f['X_test']
        return X_train, X_test
    else:
        #We don't generate the dataset in this example
        print "[!] No dataset found (coins.h5)"
        return None