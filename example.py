""""Example usage of DenseTensor layer on MNIST dataset (~0.2% train/2% test error with single layer). """

import os
import logging
import logging.config
from sklearn.utils import shuffle
from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import pickle
import keras.backend as K
from tqdm import tqdm
from dense_tensor import DenseTensor
from keras.regularizers import WeightRegularizer, l1, l2

def accuracy(model, x, label_true, batch_size):
    """Calculate accuracy of a model"""
    y_pred = model.predict(x, batch_size=batch_size)
    label_pred = np.argmax(y_pred,axis=1)
    correct = np.count_nonzero(label_true == label_pred)
    return 1.0-(float(correct)/float(x.shape[0]))

def one_hot(labels, m):
    """Convert labels to one-hot representations"""
    n = labels.shape[0]
    y = np.zeros((n,m))
    y[np.arange(n),labels.ravel()]=1
    return y

def model(hidden_dim=512, input_dim=28*28, regularization=1e-9, k=10,
          #activation = lambda x: K.relu(x, 1.0 / 5.5)
          activation = "tanh"
          ):
    """Create two layer MLP with softmax output"""
    _x = Input(shape=(input_dim,))
    reg = l1(regularization)
    y = DenseTensor(k, activation='softmax', W_regularizer=reg, V_regularizer=reg)
    _y=y(_x)
    m = Model(_x, _y)
    m.summary()
    m.compile(Adam(1e-3),loss='categorical_crossentropy')
    return m

def mnist_data():
    """Rescale and reshape MNIST data"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return (x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    logging.config.fileConfig('logging.conf')
    path = "output/dense_tensor/test"
    if not os.path.exists(path):
        os.makedirs(path)
    x_train, y_train, x_test, y_test = mnist_data()
    nb_epoch = 100
    batch_size = 128
    k = 10
    decay = 0.96
    lr = 1e-3
    m=model()
    m.summary()
    log = []
    for epoch in tqdm(range(nb_epoch)):
        acc_train = accuracy(m, x_train, y_train, batch_size=batch_size)
        acc_test = accuracy(m, x_test, y_test, batch_size=batch_size)
        log.append([acc_train, acc_test])
        m.optimizer.lr.set_value(np.float32(lr))
        logging.info("Epoch: %i/%i, Train: %f, Test: %f, LR: %f"%(epoch, nb_epoch, acc_train, acc_test, lr))
        x_train, y_train = shuffle(x_train, y_train)
        m.fit(x_train, one_hot(y_train, k), nb_epoch=1, batch_size=batch_size, shuffle=True,
              validation_data=(x_test, one_hot(y_test,k)))
        lr *= decay
        if epoch%10 == 0:
            m.save_weights("%s/checkpoint-%03i.hd5"%(path, epoch))
    m.save_weights('%s/model.hd5'%path)
    with open("%s/log.pkl"%path, "w") as f:
        pickle.dump(log, f)
