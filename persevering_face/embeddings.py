import numpy as np

import pennylane as qml



def angle_embed_image(input):
    values = np.ndarray.flatten(input.numpy())

    for i, value in enumerate(values):
        x_i = np.arctan(value)
        x_i_squared = np.arctan(value**2)
        qml.RY(x_i, wires=i)
        qml.RZ(x_i_squared, wires=i)
        


def angle_embed_images(images):
    return [angle_embed_image(image) for image in images]


