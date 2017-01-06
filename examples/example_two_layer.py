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
from dense_tensor import DenseTensorLowRank
from keras.regularizers import WeightRegularizer, l1, l2
from example import experiment

"""
h (Dense)                        (None, 64)            50240       input_1[0][0]
y (DenseTensorLowRank)           (None, 10)            31370       h[0][0]
Total params: 81610
"""


def two_layer_model(input_dim=28 * 28, hidden_dim=64, regularization=1e-5, k=10, activation='tanh', qy=24):
    """Create two layer MLP with softmax output"""
    _x = Input(shape=(input_dim,))
    reg = lambda: l1(regularization)

    h = Dense(output_dim=hidden_dim, activation=activation, W_regularizer=reg(), name="h")
    y = DenseTensorLowRank(q=qy, output_dim=k, activation='softmax', W_regularizer=reg(), V_regularizer=reg(), name="y")

    _y = y(h(_x))
    m = Model(_x, _y)
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/two_layer_model"
    model = two_layer_model()
    experiment(path, model)
