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


def tensor_model(input_dim=28 * 28, regularization=1e-5, k=10):
    """Create two layer MLP with softmax output"""
    _x = Input(shape=(input_dim,))
    reg = lambda: l1(regularization)
    y = DenseTensor(k, activation='softmax', W_regularizer=reg(), V_regularizer=reg())
    _y = y(_x)
    m = Model(_x, _y)
    m.summary()
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


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
    x_train, y_train, x_test, y_test = mnist_data()

    batch_size = 32
    k = 10
    model.summary()
    history = model.fit(x_train, one_hot(y_train, k), nb_epoch=nb_epoch, batch_size=batch_size,
                        validation_data=(x_test, one_hot(y_test, k)))
    model.save_weights('%s/model.hd5' % path)
    csvpath = os.path.join(path, "history.csv")
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/dense_tensor/test1"
    m = tensor_model()
    experiment(path, m)
