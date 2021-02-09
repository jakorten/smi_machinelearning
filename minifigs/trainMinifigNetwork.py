import os
import math
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# download the data from kaggle:
# https://www.kaggle.com/ihelon/lego-minifigures-tensorflow-tutorial
#
# somewhat reorganized...
#
BASE_DIR = 'minifigs/'
names = ["dude", "firefighter", "harrypotter", "jurassicguy", "jurassicwomen", "ronweasley", "spidermanblack", "spiderwomen"]

#tf.random.set_seed(1) obsolete now
#tf.set_random_seed(1) is deprecated so:
tf.compat.v1.set_random_seed(1)

# Step 2:

# Reorganize the folder structure:
if not os.path.isdir(BASE_DIR + 'train/'):
    for name in names:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)

# Step 3:

# move the image files to train, val and test
for folder_idx, folder in enumerate(names):
    files = os.listdir(BASE_DIR + folder)
    number_of_images = len([name for name in files])
    n_train = int((number_of_images * 0.6) + 0.5)
    n_valid = int((number_of_images*0.25) + 0.5)
    n_test = number_of_images - n_train - n_valid
    print(number_of_images, n_train, n_valid, n_test)
    for idx, file in enumerate(files):
        file_name = BASE_DIR + folder + file
        if idx < n_train:
            shutil.move(file_name, BASE_DIR + "train/" + names[folder_idx])
        elif idx < n_train + n_valid:
            shutil.move(file_name, BASE_DIR + "val/" + names[folder_idx])
        else:
            shutil.move(file_name, BASE_DIR + "test/" + names[folder_idx])
