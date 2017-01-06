"""
DenseTensorLowRank Layer
Based on Dense Layer (https://github.com/fchollet/keras/blob/master/keras/layers/core.py)
Calculates f_i = a( xV_ix^T + W_ix^T + b_i)
Where V is a low rank approximation by the multiplication of Q1 and Q2
"""

from keras import backend as K
from .dense_tensor import DenseTensor


class DenseTensorLowRank(DenseTensor):
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

    def __init__(self, q=10, **kwargs):
        self.q = q
        super(DenseTensorLowRank, self).__init__(**kwargs)

    def build_V(self, input_dim):
        self.Q1 = self.init((self.output_dim, input_dim, self.q),
                            name='{}_Q1'.format(self.name))  # p,m,q
        self.Q2 = self.init((self.output_dim, input_dim, self.q),
                            name='{}_Q2'.format(self.name))  # p,m,q
        self.V = K.batch_dot(self.Q1, self.Q2, axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m
        return self.Q1, self.Q2


    def get_config(self):
        config = {'q': self.q}
        base_config = super(DenseTensorLowRank, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
