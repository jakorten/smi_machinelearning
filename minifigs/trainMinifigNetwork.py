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

tf.random.set_seed(1)

# Step 2:

# Reorganize the folder structure:
if not os.path.isdir(BASE_DIR + 'train/'):
    for name in names:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)

# Step 3:
