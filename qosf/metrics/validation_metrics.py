import os

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import tensorflow as tf

class ConfusionMatrixCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]   
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_np = y_true.numpy()
        y_true_class = np.argmax(y_true_np, axis=1)
        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true_class, y_pred_class, ax=ax)
        tf.summary.image(f'confusion_matrix_epoch_{epoch}', fig, step=epoch)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
   


