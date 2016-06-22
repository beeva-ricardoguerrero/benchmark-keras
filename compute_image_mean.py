def compute_image_mean(in_array):
    """
    It behaves exactly as Caffe's compute_image_mean.cpp
    Given a batch of images (BGR order expected) it computes the mean image per channel.
    E.g. if the images are single channel, it does not return a single float value that is
    the average of all images, but it will return an "image" (2D numpy array) with
    the same size as the original image and its values come from averaging all images

    in_array: np.array
        it's expected shape is (batch_size, channels, height, width) or (batch_size, depth, rows, cols)

    return: dict of np.array
        mean['B']  = shape: (height, width)
        mean['G']  = shape: (height, width)
        mean['R']  = shape: (height, width)

        in case of single channel just np.array
        shape: (height, width)
    """

    import numpy as np

    if in_array.shape[1] == 1:  # Single channel
        return np.squeeze(np.mean(in_array, 0))

    else:  # Multi channel
        mean = dict()

        res = np.mean(in_array, 0)
        mean['B'] = res[0, :, :]
        mean['G'] = res[1, :, :]
        mean['R'] = res[2, :, :]

        return mean