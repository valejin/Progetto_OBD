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

def softmax(Z):

    exp_Z = np.exp(Z)
    A = exp_Z / np.sum(exp_Z)

    return A


def relu_derivative(Z):
    
    return (Z > 0).astype(float)

def tanh_derivative(Z):
    
    return 1.0 - np.tanh(Z)**2


def sigmoid_derivative(Z):
    
    s = sigmoid(Z)
    return s * (1 - s)