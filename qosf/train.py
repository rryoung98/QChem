import datetime
import numpy as np
import cirq
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_quantum as tfq

from qosf.layers.circuit_layer_builder import CircuitLayerBuilder
from qosf.data_loader.loader import DataCircuitLoader
from qosf.metrics.validation_metrics import ConfusionMatrixCallback
def create_quantum_model(qubits=10):

    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits =  cirq.LineQubit.range(qubits)  # a 4x4 grid.
    readout =cirq.LineQubit.range(qubits)     # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)
    print("adding layers")
    # # Then add layers (experiment by adding more).
    builder.add_layer(circuit,cirq.Y, cirq.CNOT, "cnot1")
    builder.add_layer(circuit, cirq.Y, cirq.CNOT, "cnot2")
    # builder.add_layer(circuit, cirq.Z, cirq.CNOT, "xx1")

    # Finally, prepare the readout qubit.
    measurements = [cirq.X, cirq.Y,cirq.Z]
    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high=2, size=1)[0]
    readout_measures = []
    for i, qubit in enumerate(readout):
      for gate in measurements:
        readout_measures.append(gate(qubit))

    return circuit, readout_measures

def create_fair_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


# Build the Keras model.

def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    data_circuit = DataCircuitLoader(x_train[:500], y_train[:500],x_test[:500], y_test[:500])
    (x_train_circ, y_train_onehot), (x_test_circ, y_test_onehot)  = data_circuit.load()
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    model_circuit, model_readout = create_quantum_model()

    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print("compiling model")
    model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),],
    optimizer=tf.keras.optimizers.Adam(0.2))

    current_time = str(datetime.datetime.now().timestamp())
    train_log_dir = './logs/tensorboard/' + current_time
    test_log_dir = './logs/tensorboard/test/' + current_time
    path_for_checkpoint_callback = 'logs/summary/'+current_time

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                patience=5, min_delta=1e-7, verbose=1)
    tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=train_log_dir, histogram_freq=1, profile_batch=3
                )
    model_ckpt =  tf.keras.callbacks.ModelCheckpoint(
                    path_for_checkpoint_callback, save_weights_only=True
                )
    cm_callback= ConfusionMatrixCallback(
                                       model=model,
                                       validation_data=(x_test_tfcirc, y_test_onehot),
                                       image_dir=f'./logs/images/' + current_time )
    print("fitting model")
 
    model.fit(x_train_tfcirc,
          y_train_onehot,
          batch_size=128,
          epochs=35,
          verbose=1,
          validation_data=(x_test_tfcirc, y_test_onehot),
          callbacks=[reduce_lr,tensorboard,model_ckpt,cm_callback]) 
    qnn_results = model.evaluate(x_test_tfcirc, y_test_onehot)

    print("qnn_results:", qnn_results[1])
    # classical_model = create_fair_classical_model()
    # classical_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
    #             optimizer=tf.keras.optimizers.Adam(0.002),
    #             metrics=['accuracy'])
    # classical_model.summary()
    # model.fit(x_train[:1000],
    #       y_train_onehot[:1000],
    #       batch_size=128,
    #       epochs=20,
    #       verbose=1,
    #       validation_data=(x_test[:1000], y_test_onehot[:1000]))

    # fair_nn_results = model.evaluate(x_test, y_test_onehot)
    qnn_accuracy = qnn_results[1]
    print(qnn_results)
    y_pred = model.predict(x_test_tfcirc)
    y_pred = np.argmax(y_pred, axis=1)

    # fair_nn_accuracy = fair_nn_results[1]
    sns.barplot(["Quantum"],
                [qnn_accuracy])
    plt.savefig(f'../logs/images/{current_time}/qnn_accuracy.png')





if __name__ == "__main__":
    train()
