import datetime
import numpy as np
import cirq
import seaborn as sns
import tensorflow as tf
import tensorflow_quantum as tfq


from qosf.layers.circuit_layer_builder import CircuitLayerBuilder
from qosf.data_loader.loader import DataCircuitLoader
def create_quantum_model():

    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits =  cirq.LineQubit.range(10)  # a 4x4 grid.
    readout =cirq.LineQubit.range(10)     # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

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
    (x_train, y_train_onehot), (x_test, y_test_onehot) = DataCircuitLoader.load()
 
    model_circuit, model_readout = create_quantum_model()

    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
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
                    path_for_checkpoint_callback, save_weights_only=False
                )
    qnn_history = model.fit(x_train,
          y_train_onehot,
          batch_size=128,
          epochs=75,
          verbose=1,
          validation_data=(x_test[:1000], y_test_onehot[:1000]),
          callbacks=[reduce_lr,tensorboard]) 
    qnn_results = model.evaluate(x_test, y_test_onehot)


    classical_model = create_fair_classical_model()
    classical_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(0.002),
                metrics=['accuracy'])
    classical_model.summary()
    model.fit(x_train[:10000],
          y_train_onehot[:10000],
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test[:1000], y_test_onehot[:1000]))

    fair_nn_results = model.evaluate(x_test, y_test_onehot)
    qnn_accuracy = qnn_results[1]
    fair_nn_accuracy = fair_nn_results[1]

    sns.barplot(["Quantum", "Classical, fair"],
                [qnn_accuracy, fair_nn_accuracy])






if __name__ == "__main__":
    train()
