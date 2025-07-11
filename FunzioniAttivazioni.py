import numpy as np

# Funzioni di attivazione
def sigmoid(Z):

    A = 1 / (1 + np.exp(-Z))

    return A

def tanh(Z):

    A = np.tanh(Z)

    return A


def relu(Z):

    A = np.maximum(0, Z)

    return A