import keras.backend as K
import tensorflow as tf


def eye(n, m):
    return tf.eye(n, m)


def quadratic_batch(x, V):
    tmp1 = K.dot(x, V)  # n,input_dim + units,input_dim,input_dim = n,units,input_dim
    xr = K.expand_dims(x, 2)  # n, 1, input_dim
    tmp2 = K.permute_dimensions(tmp1, (0, 2, 1))  # n, input_dim, units
    tmp3 = K.batch_dot(xr, tmp2, axes=[[2], [1]])  # n,1,input_dim + n,input_dim,units = n,1,units
    tmp4 = tmp3[:, 0, :]
    return tmp4
