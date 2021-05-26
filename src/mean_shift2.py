"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
"""

import numpy as np
from os import listdir
from os.path import join, isdir, isfile
from glob import glob
import cv2
import timeit
from tqdm import tqdm

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(root):
    #cls_dirs = [root]#[d for d in listdir(root) if isdir(join(root, d))]
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    im_pths = [join(root, d) for d in listdir(root) if isfile(join(root, d))]

    for _, path in enumerate(tqdm(im_pths)):
        im = cv2.imread(path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im / 255.0
        pixel_num += (im.size / CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


# The script assumes that under train_root, there are separate directories for each class
# of training images.
train_root = "D:\Bostan\SIMD\SIMD_NEW\Dataset_for_training_blurred\SIMD\SIMD_train_HR"
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end - start))
print("mean:{}\nstd:{}".format(mean, std))