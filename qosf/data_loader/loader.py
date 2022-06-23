from tkinter import E
import numpy as np

import tensorflow as tf

from skimage.measure import block_reduce

from qosf.data_loader.encoder import Encoder
from qosf.data_loader.preprocess import PreProcess

THRESHOLD=0.3

class DataCircuitLoader(PreProcess,Encoder):
    def __init__(self, x_train,y_train,x_test,y_test ,preprocessing=True,split=True):
        super().__init__(x_train,y_train,x_test,y_test)
        self.preprocessing = preprocessing
        self.split = split
        self.x_train = x_train
        self.y_train  = y_train
        self.x_test = x_test
        self.y_test = y_test
    def _one_hot(self):
        y_train_onehot = tf.one_hot(self.y_train,10)
        y_test_onehot = tf.one_hot(self.y_test,10)   
        return y_train_onehot, y_test_onehot
    def load(self,reshape=28,crop=[5,25]):
        train_embeddings = []
        test_embeddings = []
        if (self.preprocessing):
            x_train_cleaned,x_test_cleaned = self.preprocess()
        else:
            x_train_cleaned = self.x_train
            x_test_cleaned = self.x_test
        x_train_cleaned = np.array([block_reduce(x.reshape(reshape,reshape)[crop[0]:crop[1],crop[0]:crop[1]], (2,2), np.mean) for x in x_train_cleaned])
        x_test_cleaned =np.array([block_reduce(x.reshape(reshape,reshape)[crop[0]:crop[1],crop[0]:crop[1]], (2,2), np.mean) for x in x_test_cleaned])
        x_train_cleaned = np.array(x_train_cleaned > THRESHOLD, dtype=np.float32)
        x_test_cleaned = np.array(x_test_cleaned > THRESHOLD, dtype=np.float32)
        for img in x_train_cleaned:
            train_embeddings.append(self.section_angle_encoding(img))
        for img in x_test_cleaned:
            test_embeddings.append(self.section_angle_encoding(img))
        y_train_onehot,y_test_onehot =self._one_hot()
        return (train_embeddings,y_train_onehot),(test_embeddings,y_test_onehot)
        
    def load_amplitude(self):
        if (self.preprocessing):
            x_train_deskew,x_test_deskew = self.preprocess()
        train_circuit,test_circuit = self.cirq_amplitude_embedding()
        return train_circuit,test_circuit
