from keras import backend as K


def keras_backend():
    if hasattr(K, 'backend'):
        return K.backend()
    else:
        return K._BACKEND


if keras_backend() == 'theano':
    from .theano_backend import *
elif keras_backend() == 'tensorflow':
    from .tensorflow_backend import *
else:
    raise ValueError("Unknown backend: {}".format(keras_backend()))
