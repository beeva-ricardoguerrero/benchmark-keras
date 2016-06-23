import cv2
import os


def filter_corrupted_images(path2dataset, prefix, path2filtereddataset):

    filtered_paths = []

    with open(path2dataset, 'rb') as fin:
        paths = fin.readlines()

    num_total_paths = len(paths)

    for num, line in enumerate(paths):
        path, label = line.strip().split()

        if os.path.exists(prefix + path):
            try:
                image = cv2.imread(prefix + path)
                _ = cv2.resize(image, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_AREA)  # Some images are corrupted in a way that imread does not throw any exception.
                                                                  # Doing a small operation on it, will uncover the misbehaviour
                filtered_paths.append(line)

            except cv2.error:
                print("Exception catched. The image in path %s can't be read. Could be corrupted\n" % path)

        else:
            print("There is no image in %s" % path)

        if num % 100 == 0 and num != 0:
            print("Processed 100 more images.. (%d/%d)\n" % (num, num_total_paths))

    print("Total correct images: %d", len(filtered_paths))

    with open(path2filtereddataset, 'wb') as fout:
        fout.writelines(filtered_paths)



if __name__ == "__main__":

    path2train = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3.txt"
    path2train_clean = r"/home/ubuntu/scripts/Keras_imagenet/splits/train4.txt"
    prefix = r"/mnt/img/"

    filter_corrupted_images(path2train, prefix, path2train_clean)