""""Example usage of DenseTensor layer on MNIST dataset (~0.2% train/2% test error with single layer). """

import logging.config
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from dense_tensor import DenseTensor

from dense_tensor.utils import l1l2
from dense_tensor.example_utils import experiment
from dense_tensor.tensor_factorization import simple_tensor_factorization


def tensor_model(input_dim=28 * 28, output_dim=10, reg=lambda: l1l2(1e-6, 1e-6)):
    """
    One layer of a DenseTensor
    """
    _x = Input(shape=(input_dim,))
    tensor_factorization = simple_tensor_factorization(tensor_regularizer=reg())
    y = DenseTensor(units=output_dim,
                    activation='softmax',
                    kernel_regularizer=reg(),
                    tensor_factorization=tensor_factorization)
    _y = y(_x)
    m = Model(_x, _y)
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/dense_tensor"
    model = tensor_model()
    experiment(path, model)
