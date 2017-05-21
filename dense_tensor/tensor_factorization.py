from keras import backend as K

from .backend import eye
from .utils import add_weight

"""
Factorizations of inner tensor. Each factorization should return a tuple of parameters and the tensor.
"""


def simple_tensor_factorization(tensor_initializer='uniform',
                                tensor_regularizer=None,
                                tensor_constraint=None):
    def fun(layer, units, input_dim, name):
        V = add_weight(layer=layer,
                       initializer=tensor_initializer,
                       regularizer=tensor_regularizer,
                       constraint=tensor_constraint,
                       shape=(units, input_dim, input_dim),
                       name=name)
        return [V], V

    return fun


def tensor_factorization_low_rank(q,
                                  tensor_initializer='uniform',
                                  tensor_regularizer=None,
                                  tensor_constraint=None):
    def fun(layer, units, input_dim, name):
        qs = [add_weight(layer=layer,
                         initializer=tensor_initializer,
                         regularizer=tensor_regularizer,
                         constraint=tensor_constraint,
                         shape=(units, q, input_dim),
                         name="{}_Q{}".format(name, i)) for i in range(2)]
        V = K.batch_dot(qs[0], qs[1], axes=[[1], [1]])  # p,m,q + p,q,m = p,m,m
        return qs, V

    return fun


def tensor_factorization_symmetric(q,
                                   alpha=1e-7,
                                   beta=1.0,
                                   tensor_initializer='uniform',
                                   tensor_regularizer=None,
                                   tensor_constraint=None):
    """
    :param q: rank of inner parameter
    :param alpha: scale of eye to add. 0=pos/neg semidefinite, >0=pos/neg definite
    :param beta: multiplier of tensor. 1=positive,-1=negative
    """

    def fun(layer, units, input_dim, name):
        Q = add_weight(layer=layer,
                       initializer=tensor_initializer,
                       regularizer=tensor_regularizer,
                       constraint=tensor_constraint,
                       shape=(units, q, input_dim),
                       name=name)  # units, input_dim, q
        tmp = K.batch_dot(Q, Q, axes=[[1], [1]])  # p,m,q + p,m,q = p,m,m
        V = beta * ((eye(input_dim, input_dim) * alpha) + tmp)  # m,p,p
        return [q], V

    return fun
