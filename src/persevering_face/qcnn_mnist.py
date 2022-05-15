from re import T
import numpy as np
import sympy

import cirq
import tensorflow as tf
# import tensorflow_quantum as tfq

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
import pennylane as qml
from qcnn_layers import ccwyy_qconv_layer,mera_circuit,ttn_layer
from embeddings import angle_embed_image
# CONSTANTS
NUM_EXAMPLES=500
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

print("Number of filtered training examples:", len(x_train))
print("Number of filtered test examples:", len(x_test))

def build_set(x, y, desired_size, t_rate):
    assert len(x) == len(y)
    r = list(range(len(x)))
    np.random.shuffle(r)
    
    if desired_size is None or t_rate is None:
        return x,y # no change
    
    n_true_max = int(desired_size * t_rate)
    n_false_max = desired_size - n_true_max
    
    n_true = 0
    n_false = 0
    
    res_x = []
    res_y = []
    
    for i in r:
        if y[i] == True and n_true < n_true_max:
            res_x.append(x[i])
            res_y.append(y[i])
            n_true += 1
        elif y[i] == False and n_false < n_false_max:
            res_x.append(x[i])
            res_y.append(y[i])
            n_false += 1
    
    assert len(res_x) == len(res_y) and len(res_x) == desired_size
    
    return np.array(res_x), np.array(res_y)

# We do a lot of computation so we want to limit the number of examples ASAP
x_train, y_train = build_set(x_train, y_train, NUM_EXAMPLES, 0.3)
x_test, y_test = build_set(x_test, y_test, NUM_EXAMPLES, 0.7)


print(np.count_nonzero(y_train) / NUM_EXAMPLES)
print(np.count_nonzero(y_test) / NUM_EXAMPLES)


print("After restricting - number of filtered training examples:", len(x_train))
print("After restricting - number of filtered test examples:", len(x_test))


# The following moment and deskew functions were taken from the works of :

# https://stackoverflow.com/questions/43577665/deskew-mnist-images
# https://fsix.github.io/mnist/Deskewing.html

from scipy.ndimage import interpolation

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix
#Deskew the training samples 
def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    img = interpolation.affine_transform(image,affine,offset=offset)
    return (img - img.min()) / (img.max() - img.min())

print("shape of x_train is " + str(x_train.shape))
print("type of x_train is " + str(type(x_train)))
print("shape of x_test is " + str(x_test.shape))
print("type of x_test is " + str(type(x_test)))


#store the deskwed x_train into a list x_train_deskew
#store the deskwed x_test into a list x_test_deskew

#training set 
x_train_deskew = [] 
for i in range(x_train.shape[0]): 
    x_train_deskew.append(deskew(x_train[i].reshape(28,28)))
x_train_deskew = np.array(x_train_deskew)
x_train_deskew = x_train_deskew[..., np.newaxis]
print("shape of x_train_deskew is " + str(np.shape(x_train_deskew)))
print("type of x_train_deskew is " + str(type(x_train_deskew)))

#test set 
x_test_deskew = [] 
for j in range(x_test.shape[0]): 
    x_test_deskew.append(deskew(x_test[j].reshape(28,28)))
x_test_deskew = np.array(x_test_deskew)
x_test_deskew = x_test_deskew[..., np.newaxis]
print("shape of x_test_deskew is " + str(np.shape(x_test_deskew)))
print("type of x_test_deskew is " + str(type(x_test_deskew)))

# 16x16 for angle embedding!
x_train_small_256 = tf.image.resize(x_train, (2,2)).numpy()
x_test_small_256 = tf.image.resize(x_test, (2,2)).numpy()


print(x_train_small_256[0],'\n testing')
# x_train_circ = [angle_embed_image(x) for x in x_train_small_256]
# x_test_circ = [angle_embed_image(x) for x in x_test_small_256]

dev1 = qml.device('default.mixed', wires = 5)
# qnode2 = qml.QNode(x_train_circ[0], dev1, interface='tf')
n_qubits = 4
n_layers = 2
weight_shapes = {"weights": (1, 3)}

n_wires = 4
n_block_wires = 4
n_params_block = 4
n_blocks = qml.MERA.get_n_blocks(range(n_wires),n_block_wires)
template_weights = [[0.1,0.1,0.1,-0.3]]*n_blocks
print(n_blocks)
@qml.qnode(dev1)
def circuit(inputs, weights):
    #inputs = tf.cast(inputs, tf.complex128)
    angle_embed_image(inputs)
    ccwyy_qconv_layer([0,1,2,3],weights)

    # just try ttn and mera
    # mera_circuit figure out how to make trainable. 
    # Create classical model.
    # Add more samples, adjust batch size, and see if it works.
    # Add more layers, and see if it works.
    # Add more qubits, and see if it works.
    # Register # of parameters.
    #  
    mera_circuit(template_weights,n_wires, n_block_wires, n_params_block)
    ttn_layer(n_wires,template_weights)

    ttn_layer(n_wires,template_weights)
    return qml.expval(qml.PauliZ(0))

    
    
    # trainiable, not able to use on MERA and TTN, 
weights = tf.Variable([0.5, 0.1,0.1,3.0], dtype=tf.float64)


# # Build the Keras model.

y_train_onehot = tf.one_hot(y_train,depth=1,dtype=tf.float64)
#layer_1 = tf.keras.layers.Dense(4, activation='relu')
layer_2 = tf.keras.layers.Dense(1, activation="softmax", input_shape=(5,3),dtype=tf.float64)
batch_norm = tf.keras.layers.BatchNormalization()

avg_pooling = tf.keras.layers.AveragePooling2D(pool_size=(1,2))
flatten=       tf.keras.layers.Flatten()
qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=n_qubits,dtype=tf.float64)
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(),batch_size=5,  dtype=tf.float64),qlayer,flatten,layer_2])
opt = tf.keras.optimizers.SGD(learning_rate=0.02)
print(x_train_small_256[0].shape)
model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()
fitting = model.fit(x_train_small_256, y_train_onehot, epochs=6, batch_size=5, validation_split=0.25, verbose=2)
