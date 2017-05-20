# Utils for Keras 1/2 compatibility

import keras

keras_2 = int(keras.__version__.split(".")[0]) > 1  # Keras > 1


def add_weight(model,
               shape,
               name,
               initializer='random_uniform',
               regularizer=None,
               constraint=None):
    if keras_2:
        model.add_weight(initializer=initializer,
                         shape=shape,
                         name=name,
                         regularizer=regularizer,
                         constraint=constraint)
    else:
        # create weight
        w = init(shape, name=name)
        # add to trainable_weights
        if not hasattr(model, 'trainable_weights'):
            model.trainable_weights = []
        model.trainable_weights.append(w)
        # add to regularizers
        if not hasattr(model, 'regularizers'):
            model.regularizers = []
        regularizer.set_param(self.W)
        model.regularizers.append(self.W_regularizer)
