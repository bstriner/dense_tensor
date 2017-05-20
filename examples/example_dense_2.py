""""Example usage of DenseTensorSymmetric layer on MNIST dataset (~0.2% train/2% test error with single layer). """

import logging.config
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam


# Total params: 50890
def dense_model_2(hidden_dim=64, input_dim=28 * 28, regularization=1e-5, k=10, activation="tanh"):
    """Create two layer MLP with softmax output"""
    _x = Input(shape=(input_dim,))
    reg = lambda: l1(regularization)
    h = Dense(output_dim=hidden_dim, activation=activation, W_regularizer=reg(), name="h")
    y = Dense(output_dim=k, activation='softmax', W_regularizer=reg(), name="y")
    _y = y(h(_x))
    m = Model(_x, _y)
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/dense_model_2"
    model = dense_model_2()
    experiment(path, model)
