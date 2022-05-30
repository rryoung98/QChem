import collections
import matplotlib as plt
import numpy as np
import tensorflow as tf


# CONSTANTS
NUM_EXAMPLES=500
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

