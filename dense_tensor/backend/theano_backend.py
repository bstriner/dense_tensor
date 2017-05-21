import keras.backend as K
import theano.tensor as T


def eye(n, m):
    return T.eye(n=n, m=m)


def quadratic_batch(x, V):
    tmp1 = K.dot(x, V)  # n,input_dim + units,input_dim,input_dim = n,units,input_dim
    tmp2 = K.batch_dot(x, tmp1, axes=[[1], [2]])  # n,input_dim + n,units,input_dim = n,units
    return tmp2
