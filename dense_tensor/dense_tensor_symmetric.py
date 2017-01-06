"""
DenseTensor Layer
Based on Dense Layer (https://github.com/fchollet/keras/blob/master/keras/layers/core.py)
Calculates f_i = a( xV_ix^T + W_ix^T + b_i)
"""

from keras.layers.core import Layer

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec, Layer, Merge
from keras.regularizers import ActivityRegularizer, Regularizer
import numpy as np
from .dense_tensor import DenseTensor
import theano.tensor as T


class DenseTensorSymmetric(DenseTensor):
    '''Tensor layer: a = f(xVx^T + Wx + b)
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(DenseTensor(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # this is equivalent to the above:
        model = Sequential()
        model.add(DenseTensor(32, input_shape=(16,)))
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(DenseTensor(32))
    ```
    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        V_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the V matrix (input_dim x output_dim x input_dim).
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''

    def __init__(self, q=10, alpha=1e-3, beta=1, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.q = q
        super(DenseTensorSymmetric, self).__init__(**kwargs)

    def build_V(self, input_dim):
        self.Q = self.init((self.output_dim, input_dim, self.q),
                           name='{}_Q'.format(self.name))  # p,m,q
        tmp = K.batch_dot(self.Q, self.Q, axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m
        self.V = self.beta * (T.eye(input_dim, input_dim).dimshuffle(['x', 0, 1]) * self.alpha + tmp)  # m,p,p
        return [self.Q]

    def get_config(self):
        config = {'q': self.q, 'alpha': self.alpha, 'beta': self.beta}
        base_config = super(DenseTensorSymmetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
