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
import theano.tensor as T
import numpy as np


class DenseTensorLowRank(Layer):
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

    def __init__(self, output_dim, q=10, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, V_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.q = q

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseTensorLowRank, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]
        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))  # m,p
        self.Q1 = self.init((self.output_dim, input_dim, self.q),
                            name='{}_Q1'.format(self.name))  # p,m,q

        self.Q2 = self.init((self.output_dim, input_dim, self.q),
                            name='{}_Q2'.format(self.name))  # p,m,q

        self.V = T.batched_tensordot(self.Q1, self.Q2, axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m

        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.Q1, self.Q2, self.b]
        else:
            self.trainable_weights = [self.W, self.Q1, self.Q2]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        tmp1 = T.tensordot(x, self.V, axes=[[1], [2]])  # n,m + p,m,m = n,p,m
        tmp2 = T.batched_tensordot(x, tmp1, axes=[[1], [2]])  # n,m + n,p,m = n,p
        output += tmp2
        if self.bias:
            output += self.b.dimshuffle(['x', 0])
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(DenseTensorSymmetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
