""""Example usage of DenseTensor layer on MNIST dataset (~0.2% train/2% test error with single layer). """

import os

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from .utils import fit


def mnist_data():
    """Rescale and reshape MNIST data"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return x_train, y_train, x_test, y_test


def experiment(path, model, epochs=100):
    if not os.path.exists(path):
        os.makedirs(path)
    print "Training %s" % path
    model.summary()
    csvpath = os.path.join(path, "history.csv")
    modelpath = os.path.join(path, "model.h5")
    if os.path.exists(csvpath):
        print "Already exists: %s" % csvpath
        return
    x_train, y_train, x_test, y_test = mnist_data()

    batch_size = 32
    k = 10
    history = fit(model, x_train, to_categorical(y_train, k),
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(x_test, to_categorical(y_test, k)))
    model.save_weights(modelpath)
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)
