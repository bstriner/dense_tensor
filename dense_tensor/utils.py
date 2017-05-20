# Utils for Keras 1/2 compatibility

import keras
from keras import regularizers
keras_2 = int(keras.__version__.split(".")[0]) > 1  # Keras > 1


def add_activity_regularizer(layer):
    if layer.activity_regularizer and not keras_2:
        layer.activity_regularizer.set_layer(layer)
        if not hasattr(layer, 'regularizers'):
            layer.regularizers = []
            layer.regularizers.append(layer.activity_regularizer)

def l1l2(l1_weight=0, l2_weight=0):
    if keras_2:
        from keras.regularizers import L1L2
        return L1L2(l1_weight, l2_weight)
    else:
        from keras.regularizers import l1l2
        return l1l2(l1_weight, l2_weight)

def get_initializer(initializer):
    if keras_2:
        from keras import initializers
        return initializers.get(initializer)
    else:
        from keras import initializations
        return initializations.get(initializer)


def add_weight(layer,
               shape,
               name,
               initializer='random_uniform',
               regularizer=None,
               constraint=None):
    initializer = get_initializer(initializer)
    if keras_2:
        layer.add_weight(initializer=initializer,
                         shape=shape,
                         name=name,
                         regularizer=regularizer,
                         constraint=constraint)
    else:
        # create weight
        w = initializer(shape, name=name)
        # add to trainable_weights
        if not hasattr(layer, 'trainable_weights'):
            layer.trainable_weights = []
        layer.trainable_weights.append(w)
        # add to regularizers
        if not hasattr(layer, 'regularizers'):
            layer.regularizers = []
        regularizer.set_param(w)
        layer.regularizers.append(regularizer)
