# How to execute:
# THEANO_FLAGS=device=gpu,floatX=float32 python main.py


#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
# This is one of the 3 ways to tell Theano (and Keras) to use GPU. And this is included in the code, so I will not forget
# to add flags or something when I execute it -> less error prone
# Doesn't work!!!

import CaffeNet
from load_data import minibatch_4Dtensor_generator

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np


def accuracy(predictions, labels):
    """
    Expected parameters (just the format)

    predictions = [3, 4, 7] -> list of int
    labels = [np.array([0,0,0,1,0]), np.array([0,0,0,0,1]), np.array([0,0,0,0,1])]  -> list of hot-encoded np.array
    """
    return (100.0 * np.sum(np.array(predictions) == np.argmax(labels, 1))
          / np.array(predictions).shape[0])


path2train = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3_npy.txt"
path2val = r"/home/ubuntu/scripts/Keras_imagenet/splits/val3_npy.txt"
prefix = r"/mnt2/img_npy/"
path2architecture = r"/home/ubuntu/scripts/Keras_imagenet/model_architecture.json"
path2weights = r"/home/ubuntu/scripts/Keras_imagenet/model_weights.h5"
path2mean = r"/mnt2/img_npy/mean.pkl"
img_rows = 256
img_cols = 256
img_crop_rows = 227
img_crop_cols = 227
nb_classes = 1000
batch_size = 256
epochs = 36  # Max_iter in Caffe is set to 100.000 iterations, batch_size is 256 and there are 702.135 images, hence we are running the experiment 36.46 epochs -> 36
samples_per_epoch = 702135  # integer, number of samples to process before going to the next epoch.

# Prepare dataset
####

# Create generator from training data
training_images_generator = minibatch_4Dtensor_generator(path2train, path2mean, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, batch_size, infinite=True)

# Create generator from validation data
validation_images_generator = minibatch_4Dtensor_generator(path2val, path2mean, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, batch_size, infinite=False)

# Build model
####
model = CaffeNet.get_Caffenet()

sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=False)
print("Compiling model\n")
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



# Do the training
####

print("Start training ...\n")
custom_lr_scheduler = CaffeNet.Caffenet_lr_decay()
model_saver = ModelCheckpoint(filepath="weights.{epoch:02d}.h5", monitor='train_loss')
hist = model.fit_generator(training_images_generator, samples_per_epoch, nb_epoch=epochs, verbose=1, callbacks=[custom_lr_scheduler, model_saver])
print(" Training finished. Results: \n")
print(hist.history)


# Save the model
####

# Save architecture
json_string = model.to_json()
open(path2architecture, 'w').write(json_string)

# Save learnt weights
model.save_weights(path2weights)


# Predict on validation set
####

print ("Computing validation accuracy...\n")

Y_pred_ls = []
Y_val_ls = []

for X_val, Y_val in validation_images_generator:
    Y_pred_ls.append(model.predict_classes(X_val, batch_size=batch_size, verbose=0))
    Y_val_ls.append(Y_val)

predictions = [item for sublist in Y_pred_ls for item in sublist]
labels = [item for sublist in Y_val_ls for item in sublist]

acc = accuracy(predictions, labels)
print('Val accuracy: ', acc)


