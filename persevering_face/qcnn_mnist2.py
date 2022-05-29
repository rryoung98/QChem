import collections
import numpy as np
import tensorflow as tf

# quantum machine learning stuff
import pennylane as qml
from qcnn_layers import ccwyy_qconv_layer,mera_circuit,ttn_layer
from embeddings import angle_embed_image,angle_embed_images

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
# x_train, y_train = build_set(x_train, y_train, NUM_EXAMPLES, 0.3)
# x_test, y_test = build_set(x_test, y_test, NUM_EXAMPLES, 0.7)


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

def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass
    
    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num_uniq_3)
    print("Number of unique 6s: ", num_uniq_6)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))
    
    return np.array(new_x), np.array(new_y)
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

# 3x3 for angle embedding!
x_train_small_256 = tf.image.resize(x_train, (4,4)).numpy()
x_test_small_256 = tf.image.resize(x_test, (4,4)).numpy()

# Remove the images that are contradicting.
x_train_nocon, y_train_nocon = remove_contradicting(x_train_small_256, y_train)
# Angle embedding
#x_train_circ = angle_embed_images(x_train_small_256)
#x_test_circ = angle_embed_images(x_test_small_256)

# CONSTANTS
NUM_EPOCHS = 10
BATCH_SIZE = 64
INPUT_SHAPE = (4,4,1)
WEIGHT_SHAPES = {'weights':(1,4)} # (1,qubits)
N_QUBITS = 4


dev1 = qml.device('default.mixed', wires = [0,1,2,3])

@qml.qnode(dev1)
def circuit(inputs, weights):
    print(inputs,weights,'inputs 3x3, 4x4')
    inputs = tf.cast(inputs, tf.complex128)
    angle_embed_image(inputs)
    # ccwyy_qconv_layer([0,1,2,3],weights)
    # ccwyy_qconv_layer([0,1,2,3],weights)

    # just try ttn and mera
    # mera_circuit figure out how to make trainable. 
    # Create classical model.
    # Add more samples, adjust batch size, and see if it works.
    # Add more layers, and see if it works.
    # Add more qubits, and see if it works.
    # Register # of parameters.
    #  
    #mera_circuit(template_weights,n_wires, n_block_wires, n_params_block)
    ttn_layer(N_QUBITS,weights)
    # 
    ttn_layer(N_QUBITS,weights)
    ttn_layer(N_QUBITS,weights)

    # ttn_layer(n_wires,template_weights)
    return qml.expval(qml.PauliZ(0))

    

# TRAIN
y_train_onehot = tf.one_hot(y_train,depth=1,dtype=tf.float64)
y_test_onehot = tf.one_hot(y_test,depth=1,dtype=tf.float64)

# LAYERS
input_layer =   tf.keras.layers.Input(shape=INPUT_SHAPE, dtype=tf.float64)
flatten_input = tf.keras.layers.Flatten()(input_layer)
qlayer = qml.qnn.KerasLayer(circuit, WEIGHT_SHAPES, output_dim=N_QUBITS**2,dtype=tf.float64)(flatten_input)
flatten= tf.keras.layers.Flatten()(qlayer)
layer_2 = tf.keras.layers.Dense(1, activation="softmax", input_shape=(BATCH_SIZE,N_QUBITS**2),dtype=tf.float64)(flatten)

# MODEL
model = tf.keras.Model(inputs=input_layer, outputs=layer_2,name="mnist_quantum")
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()

# FIT
x_flattened = tf.reshape(x_train_nocon,(x_train_nocon.shape[0],-1))
x_test_flattened = tf.reshape(x_test_small_256,(x_test_small_256.shape[0],-1))
fitting = model.fit(x_train_small_256, y_train,  epochs=6, batch_size=BATCH_SIZE,  verbose=2, validation_data=(x_test_small_256, y_test))



# CLASSICAL MODEL

def create_fair_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


model = create_fair_classical_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

#model.fit(x_train_small_256, y_train, 
#   batch_size=BATCH_SIZE,
#   epochs=100,
#   verbose=2,
#   validation_data=(x_test_small_256, y_test))
