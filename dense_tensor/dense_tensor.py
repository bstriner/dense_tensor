"""
DenseTensor Layer
Based on Dense Layer (https://github.com/fchollet/keras/blob/master/keras/layers/core.py)
Calculates f_i = a( xV_ix^T + W_ix^T + b_i)
"""

from keras import activations, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec, Layer
from .backend import quadratic_batch
from .tensor_factorization import simple_tensor_factorization
from .utils import add_weight, get_initializer, add_activity_regularizer


class DenseTensor(Layer):
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

    def __init__(self, units,
                 activation='linear',
                 weights=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='uniform',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 bias=True,
                 input_dim=None,
                 factorization=simple_tensor_factorization(),
                 **kwargs):
        self.activation = activations.get(activation)
        self.units = units
        self.input_dim = input_dim
        self.factorization = factorization

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = add_weight(layer=self,
                            shape=(input_dim, self.units),
                            name='{}_W'.format(self.name),
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint)
        self.V_weights, self.V = self.factorization(name='{}_V'.format(self.name),
                                                    layer=self,
                                                    input_dim=input_dim,
                                                    units=self.units)
        if self.bias:
            self.b = add_weight(layer=self,
                                shape=(self.units,),
                                name='{}_b'.format(self.name),
                                initializer=self.bias_initializer,
                                regularizer=self.bias_regularizer,
                                constraint=self.bias_constraint)

        add_activity_regularizer(layer=self)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        del self.initial_weights

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        q = quadratic_batch(x, self.V)
        output += q
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.units

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation.__name__,
                  'kernel_initializer': self.kernel_initializer.__name__,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if self.kernel_regularizer else None,
                  'kernel_constraint': self.kernel_constraint.get_config() if self.kernel_constraint else None,
                  'bias_initializer': self.bias_initializer.__name__,
                  'bias_regularizer': self.bias_regularizer.get_config() if self.bias_regularizer else None,
                  'bias_constraint': self.bias_constraint.get_config() if self.bias_constraint else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(DenseTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
