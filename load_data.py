def load_images_as_np_4Dtensor(path2dataset, prefix, img_rows, img_cols, img_crop_rows, img_crop_cols, nclasses):
    """

    :return:
    """

    import math as mt
    import os
    import numpy as np
    import cv2

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
