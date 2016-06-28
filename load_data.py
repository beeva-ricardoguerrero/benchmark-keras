import math as mt
import os
import numpy as np
import cv2
from random import shuffle
from itertools import cycle, islice
import pickle
from keras.utils import np_utils


def load_img_as_4Dtensor(path2dataset, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols):
    """

    :return:
    """

    x_ls = []
    y_ls = []

    with open(path2dataset, 'rb') as fin:
        paths = fin.readlines()

    num_total_paths = len(paths)

    for num, line in enumerate(paths):
        path, label = line.strip().split()

        if os.path.exists(prefix + path):
            try:
                image = cv2.imread(prefix + path)
                image = cv2.resize(image, (img_rows, img_cols),
                                   interpolation=cv2.INTER_AREA)  # Resize in create_caffenet.sh

                if img_crop_rows != 0 and img_crop_rows != img_rows: # We need to crop rows
                    crop_rows = img_rows - img_crop_rows
                    crop_rows_pre, crop_rows_post = int(mt.ceil(crop_rows / 2.0)), int(mt.floor(crop_rows / 2.0))
                    image = image[crop_rows_pre:-crop_rows_post, :]

                if img_crop_cols != 0 and img_crop_cols != img_cols:  # We need to crop cols
                    crop_cols = img_cols - img_crop_cols
                    crop_cols_pre, crop_cols_post = int(mt.ceil(crop_cols / 2.0)), int(mt.floor(crop_cols / 2.0))
                    image = image[:, crop_cols_pre:-crop_cols_post]  # Crop in train_val.prototxt

                x_ls.append(image)
                y_ls.append(int(label))
            except cv2.error:
                print("Exception catched. The image in path %s can't be read. Could be corrupted\n" % path)
        else:
            print("There is no image in %s" % path)

        if num % 100 == 0 and num != 0:
            print("Loaded 100 more images.. (%d/%d)\n" % (num, num_total_paths))


    print("Total images loaded: %d (remember that corrupted or not present images were discarded)" % len(x_ls))

    x_np = np.array(x_ls)
    y_np = np.array(y_ls)

    return x_np, y_np



class minibatch_4Dtensor_generator(object):
    """
    generator: a generator.
            The output of the generator must be either
            - a tuple (inputs, targets)
            - a tuple (inputs, targets, sample_weights).
            All arrays should contain the same number of samples.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `samples_per_epoch`
            samples have been seen by the model.

    Note:
    Iterator is a more general concept: any object whose class has a next method and an __iter__ method that does return self.
    A generator is a function that produces or yields a sequence of values using yield method.

    """

    def __init__(self, path2dataset, path2mean, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, batch_size, infinite, nb_classes=1000):

        self.prefix = prefix
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_crop_rows = img_crop_rows
        self.img_crop_cols = img_crop_cols
        self.batch_size = batch_size
        self.position = 0
        self.original_paths = []
        self.infinite = infinite
        self.end_reached = False
        self.nb_classes = nb_classes

        with open(path2dataset, "rb") as fin:
            for line in fin:
                self.original_paths.append(line.strip())

        with open(path2mean, "rb") as fin:
            self.mean = pickle.load(fin)

        self.iter = cycle(self.original_paths)

    def __iter__(self):

        while True:

            if self.infinite:
                # Get a slice of the whole paths
                current_paths = islice(self.iter, self.batch_size)

                self.position += self.batch_size

                # Every epoch we need to randomize the images' order to prevent the net from learning specific sequences
                if self.position > len(self.original_paths):
                    shuffle(self.original_paths)
                    self.position = 0
            else:
                offset = self.position + self.batch_size
                if offset < len(self.original_paths):
                    current_paths = self.original_paths[self.position:offset]
                else:
                    current_paths = self.original_paths[self.position:]
                    self.end_reached = True

            # Load
            X = []
            Y = []

            print("Loading data...\n")

            for line in current_paths:
                path, label = line.strip().split()

                try:  # 100% sure that is not needed, but I don't want incidentals in a multiple days experiment
                    image = np.load(self.prefix + path)  # Already resized and cropped

                    X.append(image)
                    Y.append(int(label))
                except (IOError, ValueError):
                    pass

            print("Finish loading data...\n")

            # Pre process
            print ("Pre-processing...\n")

            X = X.astype('float32')

            X[:, 0, :, :] -= self.mean['B']
            X[:, 1, :, :] -= self.mean['G']
            X[:, 2, :, :] -= self.mean['R']

            X = X.reshape(X.shape[0], X.shape[3], self.img_crop_rows, self.img_crop_cols)

            print('X_train shape:', X.shape)
            print(X.shape[0], 'train samples')

            # convert class vectors to binary class matrices
            Y = np_utils.to_categorical(Y, self.nb_classes)

            ####

            # Now we have a list with the images corresponding to a minibatch, return the 4D tensor
            yield (np.array(X), np.array(Y))

            if self.end_reached:
                break
