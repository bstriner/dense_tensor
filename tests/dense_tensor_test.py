import pytest
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from dense_tensor import DenseTensor
from dense_tensor import simple_tensor_factorization
from dense_tensor import tensor_factorization_low_rank
from dense_tensor import tensor_factorization_symmetric
from dense_tensor.example_utils import mnist_data
from dense_tensor.utils import l1l2, fit


# Models

def tensor_model(input_dim=28 * 28, output_dim=10, reg=lambda: l1l2(1e-6, 1e-6)):
    """
    One layer of a DenseTensor
    """
    _x = Input(shape=(input_dim,))
    factorization = simple_tensor_factorization(tensor_regularizer=reg())
    y = DenseTensor(units=output_dim,
                    activation='softmax',
                    kernel_regularizer=reg(),
                    factorization=factorization)
    _y = y(_x)
    m = Model(_x, _y)
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


def tensor_model_low_rank(input_dim=28 * 28, output_dim=10, reg=lambda: l1l2(1e-6, 1e-6)):
    """
    One layer of a DenseTensor low rank
    """
    _x = Input(shape=(input_dim,))
    factorization = tensor_factorization_low_rank(q=10, tensor_regularizer=reg())
    y = DenseTensor(units=output_dim,
                    activation='softmax',
                    kernel_regularizer=reg(),
                    factorization=factorization)
    _y = y(_x)
    m = Model(_x, _y)
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


def tensor_model_symmetric(input_dim=28 * 28, output_dim=10, reg=lambda: l1l2(1e-6, 1e-6)):
    """
    One layer of a DenseTensor low rank
    """
    _x = Input(shape=(input_dim,))
    factorization = tensor_factorization_symmetric(q=10, tensor_regularizer=reg())
    y = DenseTensor(units=output_dim,
                    activation='softmax',
                    kernel_regularizer=reg(),
                    factorization=factorization)
    _y = y(_x)
    m = Model(_x, _y)
    m.compile(Adam(1e-3, decay=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
    return m


@pytest.mark.parametrize("model_function_name", ["tensor_model", "tensor_model_low_rank", "tensor_model_symmetric"])
def dense_tensor_test(model_function_name):
    batch_size = 32
    batches = 20
    k = 10
    epochs = 1
    model_function = globals()[model_function_name]
    model = model_function()
    x_train, y_train, x_test, y_test = mnist_data()
    x_train = x_train[:batches * batch_size, ...]
    y_train = y_train[:batches * batch_size, ...]
    x = x_train
    y = to_categorical(y_train, k)
    fit(model, x, y,
        epochs=epochs,
        batch_size=batch_size)


if __name__ == "__main__":
    pytest.main([__file__])
