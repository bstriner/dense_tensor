""""Example usage of DenseTensor layer on MNIST dataset (~0.2% train/2% test error with single layer). """

import os
import logging
import logging.config
from sklearn.utils import shuffle
from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import pickle
import keras.backend as K
from tqdm import tqdm
from dense_tensor import DenseTensor
from keras.regularizers import WeightRegularizer, l1, l2
import pandas as pd


def one_hot(labels, m):
    """Convert labels to one-hot representations"""
    n = labels.shape[0]
    y = np.zeros((n, m))
    y[np.arange(n), labels.ravel()] = 1
    return y


def mnist_data():
    """Rescale and reshape MNIST data"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return (x_train, y_train, x_test, y_test)


def experiment(path, model, nb_epoch=100):
    if not os.path.exists(path):
        os.makedirs(path)
    print "Training %s" % path
    csvpath = os.path.join(path, "history.csv")
    modelpath = os.path.join(path, "model.h5")
    if os.path.exists(csvpath):
        print "Already exists: %s"%csvpath
        return
    x_train, y_train, x_test, y_test = mnist_data()

    batch_size = 32
    k = 10
    model.summary()
    history = model.fit(x_train, one_hot(y_train, k), nb_epoch=nb_epoch, batch_size=batch_size,
                        validation_data=(x_test, one_hot(y_test, k)))
    model.save_weights(modelpath)
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)
