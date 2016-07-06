from load_data import minibatch_4Dtensor_generator
from keras.models import model_from_json
from keras.optimizers import SGD
import CaffeNet
import numpy as np


def accuracy(predictions, labels):
    """
    Expected parameters (just the format)

    predictions = [np.array(3), np.array(4), np.array(5)] -> list of np.array
    labels = [np.array([0,0,0,1,0]), np.array([0,0,0,0,1]), np.array([0,0,0,0,1])]  -> list of hot-encoded np.array
    """
    return (100.0 * np.sum(np.squeeze(np.array(predictions)) == np.argmax(labels, 1))
          / np.array(predictions).shape[0])


path2train = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3_npy.txt"
path2val = r"/home/ubuntu/scripts/Keras_imagenet/splits/val3_npy.txt"
prefix = r"/mnt2/img_npy/"
path2architecture = r"/home/ubuntu/scripts/Keras_imagenet/model_architecture.json"
path2weights = r"/home/ubuntu/scripts/Keras_imagenet/weights.10.hdf5"
path2mean = r"/mnt2/img_npy/mean.pkl"
img_rows = 256
img_cols = 256
img_crop_rows = 227
img_crop_cols = 227
nb_classes = 1000
batch_size = 256
epochs = 36  # Max_iter in Caffe is set to 100.000 iterations, batch_size is 256 and there are 702.135 images, hence we are running the experiment 36.46 epochs -> 36
samples_per_epoch = 702135  # integer, number of samples to process before going to the next epoch.


# Create generator from validation data
validation_images_generator = minibatch_4Dtensor_generator(path2val, path2mean, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, batch_size, infinite=False)

# Load the model
#####

#model = model_from_json(open(path2architecture).read())
model = CaffeNet.get_Caffenet()
model.load_weights(path2weights)

# Finally, before it can be used, the model shall be compiled.
sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


print ("Computing validation accuracy...\n")

Y_pred_ls = []
Y_val_ls = []

for X_val, Y_val in validation_images_generator:
    Y_pred_ls.append(model.predict_classes(X_val, batch_size=batch_size, verbose=0))
    Y_val_ls.append(Y_val)

acc = accuracy(Y_pred_ls, Y_val_ls)
print('Val accuracy: ', acc)
