import numpy as np

from activationFunctions import *
from config import *

# X è la mtrice che contiene tutti i campioni
# W è un tensore che contiene le matrici dei pesi
# b è il vettore di bias
# g è la funzione di attivazione
# m è il numero di campioni
def forward_pass(X, W, b, g, m):

    phi_label = np.zeros(m, dtype=int)

    if g == 'relu':
        activation = relu
    elif g == 'tanh':
        activation = tanh
    elif g == 'sigmoid':
        activation = sigmoid
    else:
        raise ValueError("Funzione di attivazione non valida")

    for i in range(m):
        a = X[i]
        for j in range(L):
            z = np.dot(W[j], a) + b[j]
            a = activation(z)
        
        z = np.dot(W[L], a) + b[L]
        phi = softmax(z)

        phi_label[i] = np.argmax(phi)


    return phi_label

#Backpropagation is used to update the weights and biases based on the error in the predictions.
#def backward_propagation(x, y, output):