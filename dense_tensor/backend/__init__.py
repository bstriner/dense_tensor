import keras

if keras.backend.backend() == 'theano':
    from .theano_backend import *
elif keras.backend.backend() == 'tensorflow':
    from .tensorflow_backend import *
else:
    raise ValueError("Unknown backend: {}".format(keras.backend.backend()))
