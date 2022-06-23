from tkinter import E
import numpy as np

import tensorflow as tf

from scipy.ndimage import interpolation

from qosf.data_loader.encoder import Encoder
from qosf.data_loader.preprocess import PreProcess


class DataCircuitLoader(Encoder,PreProcess):
    def __init__(self, dataset,preprocessing=True,split=True):
        self.preprocessing = preprocessing
        self.split = split
        if (dataset is not None):
            self.dataset = dataset
        else:
            self.dataset= tf.keras.datasets.mnist.load_data()
        if (split):
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset
            self.y_train_onehot, y_test_onehot = self._one_hot()
    def _one_hot(self):
        y_train_onehot = tf.one_hot(self.y_train,10)
        y_test_onehot = tf.one_hot(self.y_test,10)   
        return y_train_onehot, y_test_onehot
    def load(self):
        if (self.preprocessing):
            x_train_deskew,x_test_deskew = self.preprocess(self.x_train,self.x_test)
        train_circuit,test_circuit = self.section_angle_encoding()
        return (train_circuit,self.y_train_onehot),(test_circuit,self.y_test_onehot)
        
    def load_amplitude(self):
        if (self.preprocessing):
            x_train_deskew,x_test_deskew = self.preprocess(self.x_train,self.x_test)
        train_circuit,test_circuit = self.cirq_amplitude_embedding()
        return train_circuit,test_circuit
