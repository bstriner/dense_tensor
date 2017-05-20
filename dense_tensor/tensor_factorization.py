from .utils import add_weight

"""
Factorizations of inner tensor. Each factorization should return a tuple of parameters and the tensor.
"""


def simple_tensor_factorization(tensor_initializer='random_uniform',
                                tensor_regularizer=None,
                                tensor_constraint=None):
    def fun(model, units, input_dim, name):
        V = add_weight(model=model,
                       initializer=tensor_initializer,
                       regularizer=tensor_regularizer,
                       constraint=tensor_constraint,
                       shape=(units, input_dim, input_dim),
                       name=name)
        return [V], V

    return fun


def tensor_factorization_two(q,
                             tensor_initializer='random_uniform',
                             tensor_regularizer=None,
                             tensor_constraint=None):
    def fun(model, units, input_dim, name):
        qs = [add_weight(model=model,
                         initializer=tensor_initializer,
                         regularizer=tensor_regularizer,
                         constraint=tensor_constraint,
                         shape=(units, input_dim, q),
                         name="{}_Q{}".format(name, i)) for i in range(2)]
        V = K.batch_dot(q[0], q[1], axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m
        return qs, V

    return fun


def tensor_factorization_symmetric(q,
                                   alpha=1e-7,
                                   beta=1.0,
                                   tensor_initializer='random_uniform',
                                   tensor_regularizer=None,
                                   tensor_constraint=None):
    """
    :param q: rank of inner parameter
    :param alpha: scale of eye to add. 0=pos/neg semidefinite, >0=pos/neg definite
    :param beta: multiplier of tensor. 1=positive,-1=negative
    :param tensor_initializer: 
    :param tensor_regularizer: 
    :param tensor_constraint: 
    :return: 
    """

    def fun(model, units, input_dim, name):
        q = add_weight(model=model,
                       initializer=tensor_initializer,
                       regularizer=tensor_regularizer,
                       constraint=tensor_constraint,
                       shape=(units, input_dim, q),
                       name="{}_Q{}".format(name, i))  # units, input_dim, q
        tmp = K.batch_dot(self.Q, self.Q, axes=[[2], [2]])  # p,m,q + p,m,q = p,m,m
        V = beta * ((T.eye(input_dim, input_dim).dimshuffle(['x', 0, 1]) * alpha) + tmp)  # m,p,p
        return [q], V

    return fun
