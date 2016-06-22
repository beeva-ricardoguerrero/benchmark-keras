import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
# This is one of the 3 ways to tell Theano (and Keras) to use GPU. And this is included in the code, so I will not forget
# to add flags or something when I execute it -> less error prone

import CaffeNet
from compute_image_mean import compute_image_mean
from load_data import load_images_as_np_4Dtensor

from keras.optimizers import SGD
from keras.utils import np_utils


path2train = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3.txt"
path2val = r"/home/ubuntu/scripts/Keras_imagenet/splits/val3.txt"
prefix = r"/mnt/img/"
path2architecture = r"/home/ubuntu/scripts/Keras_imagenet/model_architecture.json"
path2weights = r"/home/ubuntu/scripts/Keras_imagenet/model_weights.h5"
img_rows = 256
img_cols = 256
img_crop_rows = 227
img_crop_cols = 227
nb_classes = 1000
batch_size = 256
epochs = 36 # Max_iter in Caffe is set to 10K iterations, batch_size is 256 and there are 702.135 images, hence we are running the experiment 36.46 epochs -> 36


# Prepare dataset
####

# Load Train data
print("Loading training data...\n")
X_train, Y_train = load_images_as_np_4Dtensor(path2train, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, nb_classes)
print("Finish loading training data...\n")


# Load Validation data
print("Loading validation data...\n")
X_val, Y_val = load_images_as_np_4Dtensor(path2val, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, nb_classes)
print("Finish loading validation data...\n")


# Pre process
mean_train = compute_image_mean(X_train)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

X_train[:, 0, :, :] -= mean_train['B']
X_train[:, 1, :, :] -= mean_train['G']
X_train[:, 2, :, :] -= mean_train['R']

X_val[:, 0, :, :] -= mean_train['B']
X_val[:, 1, :, :] -= mean_train['G']
X_val[:, 2, :, :] -= mean_train['R']

X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], img_crop_rows, img_crop_cols)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[3], img_crop_rows, img_crop_cols)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples\n')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_val = np_utils.to_categorical(Y_val, nb_classes)


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
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=0, callbacks=[custom_lr_scheduler])
print(" Training finished. Results: \n")
print(hist.history)

# Predict on validation set
score = model.evaluate(X_val, Y_val, verbose=0)
print('Val loss: ', score[0])
print('Val accuracy: ', score[1])

# Save the model
####

# Save architecture
json_string = model.to_json()
open(path2architecture, 'w').write(json_string)

# Save learnt weights
model.save_weights(path2weights)