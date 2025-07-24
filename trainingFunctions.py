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
def general_weight_initializer(neurons, method):
    W = []
    b = []

    for l in range(len(neurons) - 1):
        n_in = neurons[l]
        n_out = neurons[l+1]

        if method == 'he':
            W.append(np.random.randn(n_out, n_in) * np.sqrt(2. / n_in))
        elif method == 'xavier':
            W.append(np.random.randn(n_out, n_in) * np.sqrt(1. / n_in))
        else:
            raise ValueError(f"Metodo '{method}' non supportato.")

        b.append(np.zeros((n_out,)))

    return W, b

def stochastic_gradient_with_momentum(dW, db, W, b, vW, vb, lam, k, loss_prev, loss_curr, epsilon=1e-4, alfa_init=0.001, reg_type="l2"):

    beta = 0.9

    # Confronta la variazione della loss
    if abs(loss_curr - loss_prev) < epsilon:
        alfa = 1 / k  # diminishing stepsize
    else:
        alfa = alfa_init      # costante finché loss migliora abbastanza

    for l in range(len(W)):
        
        if reg_type == "l2":
            reg_term = 2 * lam * W[l]
        elif reg_type == "l1":
            reg_term = lam * np.sign(W[l])
        else:
            raise ValueError("Tipo di regolarizzazione non supportato: usa 'l1' o 'l2'")
        
        vW[l] = beta * vW[l] - alfa * (dW[l] + reg_term)
        vb[l] = beta * vb[l] - alfa * db[l]

        W[l] += vW[l]
        b[l] += vb[l]

    return W, b, vW, vb


def compute_loss(phi, Y_true, W, lam):
    """
    Calcola la cross-entropy loss con regolarizzazione L2.
    
    phi: output softmax della rete, shape (batch_size, num_classes)
    Y_true: one-hot labels, shape (batch_size, num_classes)
    W: lista dei pesi per ciascun layer
    lam: coefficiente di regolarizzazione (lambda)
    """

    phi = np.array(phi)
    Y_true = np.array(Y_true)

    m = Y_true.shape[0]
    eps = 1e-8  # per evitare log(0)

    # Cross-entropy loss
    loss_ce = -np.sum(Y_true * np.log(phi + eps)) / m

    # L2 regularization: somma dei quadrati di tutti i pesi
    loss_reg = lam * sum(np.sum(w ** 2) for w in W)

    # Totale
    return loss_ce + loss_reg
