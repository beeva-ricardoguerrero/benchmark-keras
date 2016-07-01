# As I was stupid, I need to recover the splits (txt) for the npy files, thus this script
# read file with jpeg splits
# check if exist a npy file
# store it in a txt

import os

path2train = r"/home/ubuntu/scripts/Keras_imagenet/splits/train3.txt"
path2train_dest = r"/mnt2/img_npy/train3_npy.txt"
path2val = r"/home/ubuntu/scripts/Keras_imagenet/splits/val3.txt"
path2val_dest = r"/mnt2/img_npy/val3_npy.txt"
prefix = r"/mnt2/img_npy/"



path2dataset = path2train
path2dataset_dest = path2train_dest


original_paths = []
destination_paths = []

with open(path2dataset, "rb") as fin:
    for line in fin:
        if line.strip():
            original_paths.append(line.strip())

for line in original_paths:
    path, label = line.split()
    path = path.replace("JPEG", "npy")

    possible_path = prefix + path
    if os.path.exists(possible_path):
        destination_paths.append(path + " " + label + "\n")

with open(path2dataset_dest, "wb") as fout:
    fout.writelines(destination_paths)

