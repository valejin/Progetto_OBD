import numpy as np
import random

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
    
    return dW, db

# neurons è un vettore che contiene il numero di neuroni per ciascun livello;  dim(neurons) = num_levels
def weight_initializer(neurons):

    np.random.seed(42)

    W = []
    b = []

    #costruzione delle matrici W per ogni livello, fissato vettore dei neuroni
    for l in range(len(neurons) - 1):
        W.append(np.random.randn(neurons[l+1], neurons[l]) * np.sqrt(2. / neurons[l])) #usiamo He initilization; POI DA FARE ANCHE CON XAVIER PER TANH
        b.append(np.zeros((neurons[l+1],)))

    return W, b

#Versione generale per inizializazione dei pesi
def general_weight_initializer(neurons, activation, method):
    W = []
    b = []

    for l in range(len(neurons) - 1):
        n_in = neurons[l]
        n_out = neurons[l+1]

        if method == 'he':
            W.append(np.random.randn(n_out, n_in) * np.sqrt(2. / n_in))
        elif method == 'xavier':
            W.append(np.random.randn(n_out, n_in) * np.sqrt(1. / n_in))
        elif method == 'xavier_uniform':
            limit = np.sqrt(6. / (n_in + n_out))
            W.append(np.random.uniform(-limit, limit, (n_out, n_in)))
        elif method == 'lecun':
            W.append(np.random.randn(n_out, n_in) * np.sqrt(1. / n_in))
        elif method == 'orthogonal':
            a = np.random.randn(n_out, n_in)
            q, _ = np.linalg.qr(a)
            gain = np.sqrt(2.) if activation == 'relu' else 1.
            W.append(q * gain)
        else:
            raise ValueError(f"Metodo '{method}' non supportato.")

        b.append(np.zeros((n_out,)))

    return W, b

def stochastic_gradient_with_momentum(dW, db, W, b, vW, vb):

    alfa = 0.001
    beta = 0.9

    for l in range(len(W)):
        vW[l] = beta * vW[l] - alfa * dW[l]
        vb[l] = beta * vb[l] - alfa * db[l]

        W[l] += vW[l]
        b[l] += vb[l]

    return W, b, vW, vb
