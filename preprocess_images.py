# As we are going to read from disk many times, the purpose of this code is to do the pre-process once
# Typical pre-process steps:
#-Filter corrupted or unreadable images
#-Resize and crop
#-Convert to double
#-Subtract the mean image
# As reading big files is less efficient than reading small files and repeat some operations (i'm talking
# about doubles), here we will end up after crop. The images will be stored as np array
#

import numpy as np
import cv2
import os
import math as mt

def preprocess_images(path2dataset_orig, prefix_orig, path2dataset_dest, prefix_dest, img_rows, img_cols, img_crop_rows, img_crop_cols):
    # Origin path = prefix + path -> /mnt/img/img393.JPEG
    # Destiny path = prefix2 + path -> /mnt/h5/img393.h5

    processed_paths = []

    with open(path2dataset_orig, 'rb') as fin:
        paths = fin.readlines()

    num_total_paths = len(paths)

    for num, line in enumerate(paths):
        path, label = line.strip().split()

        if os.path.exists(prefix_orig + path):
            try:
                image = cv2.imread(prefix_orig + path)
                image = cv2.resize(image, (img_rows, img_rows),
                                   interpolation=cv2.INTER_AREA)  # Resize in create_caffenet.sh

            except cv2.error:
                print("Exception catched. The image in path %s can't be read. Could be corrupted\n" % path)
                continue

            if img_crop_rows != 0 and img_crop_rows != img_rows:  # We need to crop rows
                crop_rows = img_rows - img_crop_rows
                crop_rows_pre, crop_rows_post = int(mt.ceil(crop_rows / 2.0)), int(mt.floor(crop_rows / 2.0))
                image = image[crop_rows_pre:-crop_rows_post, :]

            if img_crop_cols != 0 and img_crop_cols != img_cols:  # We need to crop cols
                crop_cols = img_cols - img_crop_cols
                crop_cols_pre, crop_cols_post = int(mt.ceil(crop_cols / 2.0)), int(mt.floor(crop_cols / 2.0))
                image = image[:, crop_cols_pre:-crop_cols_post]  # Crop in train_val.prototxt

            # Store the image in h5 format
            npy_path = prefix_dest + path.split(".")[0] + ".npy"
            with open(npy_path, "wb") as fout:
                np.save(fout, image)

            processed_paths.append(line.replace("JPEG", "npy"))

        else:
            print("There is no image in %s" % path)

        if num % 100 == 0 and num != 0:
            print("Pre-processed 100 more images.. (%d/%d)\n" % (num, num_total_paths))

    with open(path2dataset_dest, "wb") as fout:
        fout.writelines(processed_paths)

    print("Total images pre-processed: %d (remember that corrupted or not present images were discarded)" % len(processed_paths))


if __name__ == "__main__":

    print("Pre-processing training set")
    path2dataset_orig = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3.txt"
    prefix_orig = r"/mnt/img/"
    path2dataset_dest = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3_npy.txt"
    prefix_dest = r"/mnt/img_npy/"
    img_rows = 256
    img_cols = 256
    img_crop_rows = 227
    img_crop_cols = 227

    preprocess_images(path2dataset_orig, prefix_orig, path2dataset_dest, prefix_dest, img_rows, img_cols, img_crop_rows,
                      img_crop_cols)

    print("Pre-processing validation set")
    path2dataset_orig = r"/home/ubuntu/scripts/Keras_imagenet/splits/val3.txt"
    path2dataset_dest = r"/home/ubuntu/scripts/Keras_imagenet/splits/val3_npy.txt"

    preprocess_images(path2dataset_orig, prefix_orig, path2dataset_dest, prefix_dest, img_rows, img_cols, img_crop_rows,
                      img_crop_cols)