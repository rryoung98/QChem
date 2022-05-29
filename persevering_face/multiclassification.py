import collections
import numpy as np
import tensorflow as tf

# quantum machine learning stuff
import pennylane as qml
from qcnn_layers import ccwyy_qconv_layer,mera_circuit,ttn_layer
from embeddings import angle_embed_image,angle_embed_images


