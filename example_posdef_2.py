""""Example usage of DenseTensorSymmetric layer on MNIST dataset (~0.2% train/2% test error with single layer). """

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
from dense_tensor import DenseTensorSymmetric
from keras.regularizers import WeightRegularizer, l1, l2
from example import experiment


def posdef_model_2(hidden_dim=64, input_dim=28 * 28, regularization=1e-5, k=10, activation="tanh"):
    """Create two layer MLP with softmax output"""
    _x = Input(shape=(input_dim,))
    reg = lambda: l1(regularization)

    h = DenseTensorSymmetric(alpha=1e-3, beta=1, q=24, output_dim=hidden_dim, activation=activation,
                             W_regularizer=reg(),
                             V_regularizer=reg(), name="h")

    y = DenseTensorSymmetric(alpha=1e-3, beta=1, q=24, output_dim=k, activation='softmax', W_regularizer=reg(),
                             V_regularizer=reg(), name="y")

    _y = y(h(_x))
    m = Model(_x, _y)
    m.summary()
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/dense_tensor_posdef_2"
    model = posdef_model_2()
    experiment(path, model)
