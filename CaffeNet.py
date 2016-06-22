import keras.callbacks
from keras import backend as K


def Caffenet_initialization(shape, name=None):
    """
    Custom weights initialization

    From Convolution2D:

    weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.

    From train_val.prototxt

    weight_filler
    {
      type: "gaussian"
      std: 0.01
    }

    bias_filler
    {
      type: "constant"
      value: 0
    }

    Si pasamos esta funcion en el parametro init, pone este peso a las W y las b las deja a 0 (comprobado leyendo el codigo de Keras)
    """

    import numpy as np
    from keras import backend as K

    mu, sigma = 0, 0.01
    return K.variable(np.random.normal(mu, sigma, shape), name=name)



def get_Caffenet():
    """
    Caffe also allows you to choose between L2 regularization (default) and L1 regularization, by setting

    regularization_type: "L1"

    En Caffenet tenemos:     weight_decay: 0.0005
    """

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2

    weight_decay = 0.0005

    model = Sequential()

    # Conv1
    model.add(Convolution2D(nb_filter=96, nb_row=11, nb_col=11, border_mode='valid', input_shape=(3, 227, 227),
                            init=Caffenet_initialization, subsample=(4, 4),
                            W_regularizer=l2(weight_decay)))  # subsample is stride
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Conv2
    model.add(
        Convolution2D(256, 5, 5, border_mode='same', init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # Conv3
    model.add(
        Convolution2D(384, 3, 3, border_mode='same', init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))

    # Conv4
    model.add(
        Convolution2D(384, 3, 3, border_mode='same', init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))

    # Conv5
    model.add(
        Convolution2D(256, 3, 3, border_mode='same', init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # Fc6
    model.add(Dense(4096, init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fc7
    model.add(Dense(4096, init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fc8
    model.add(Dense(1000, init=Caffenet_initialization, W_regularizer=l2(weight_decay)))
    model.add(Activation('softmax'))

    return model


class Caffenet_lr_decay(keras.callbacks.Callback):
    """
    """

    def on_batch_end(self, batch, logs={}):
        lr = self.model.optimizer.lr.get_value()
        gamma = 0.1
        new_lr = lr * gamma
        K.set_value(self.model.optimizer.lr, new_lr)