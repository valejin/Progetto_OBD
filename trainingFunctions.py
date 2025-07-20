import numpy as np

from activationFunctions import *
from config import *

# X è la mtrice che contiene tutti i campioni
# W è un tensore che contiene le matrici dei pesi
# b è il vettore di bias
# g è la funzione di attivazione
# m è il numero di campioni
def forward_pass(X, W, b, g, m):

    phi = []
    phi_label = np.zeros(m, dtype=int)
    all_activations = []

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
        activations = [a]

        for j in range(L):
            z = np.dot(W[j], a) + b[j]
            a = activation(z)
            activations.append(a)
        
        z = np.dot(W[L], a) + b[L]
        phi.append(softmax(z))

        phi_label[i] = np.argmax(phi[i])
        all_activations.append(activations)

    return phi, phi_label, all_activations

def backpropagation(phi, Y, W, b, g, all_activations, m):

    # Scegli attivazione e derivata
    if g == 'relu':
        activation_deriv = relu_derivative
    elif g == 'tanh':
        activation_deriv = tanh_derivative
    elif g == 'sigmoid':
        activation_deriv = sigmoid_derivative
    else:
        raise ValueError("Funzione di attivazione non valida")

    # Inizializza gradienti per ogni layer
    #Stiamo inizializzando i gradienti con zeri, ma con la stessa forma delle rispettive matrici/pesi
    dW = [np.zeros_like(W_l) for W_l in W] # gradiente di E rispetto ai pesi per ogni layer
    db = [np.zeros_like(b_l) for b_l in b] # gradiente di E rispetto ai bias per ogni layer

    for i in range(m):
        activations = all_activations[i]
        """ a_L = activations[L]  # output layer input (prima di softmax)
        z_output = np.dot(W[L], a_L) + b[L]
        phi = softmax(z_output) """

        # delta^{L+1}
        delta = phi[i] - Y[i]

        # Gradiente per l'output layer
        dW[L] += np.outer(delta, activations[L])
        db[L] += delta

        # Backpropagation per i layer nascosti
        for l in reversed(range(L)):
            z = np.dot(W[l], activations[l]) + b[l]  # z^l
            da_dz = activation_deriv(z)              # g'(z^l)
            delta = np.dot(W[l + 1].T, delta) * da_dz
            dW[l] += np.outer(delta, activations[l])
            db[l] += delta

    # Media sui batch
    dW = [dw / m for dw in dW]
    db = [db_l / m for db_l in db]

    # Compattiamo la derivata del rischio empirico
    grads = []

    for dw, db_l in zip(dW, db):
        grads.append(dw.flatten())
        grads.append(db_l.flatten())
    
    return np.concatenate(grads)
