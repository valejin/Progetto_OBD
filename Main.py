# come funzione di attivazione usiamo ReLU e Tanh
# come funzione di regolarizzazione usiamo L1 e L2 (studiate a lezione)
import pandas as pd
import numpy as np

from preprocessing import *
from trainingFunctions import *

def main():
    print('BENVENUTO!\n\n')
    
    dataset_choice = print_menu('Dataset disponibili:\n\n[1] Wine Quality Red\n[2] Glioma Grading\n')

    print('\nHai scelto il dataset ' + dataset_choice)

    if(dataset_choice == '1'):
        dataset = pd.read_csv('./datasets/winequality-red.csv')
        X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(dataset, "quality")
    elif(dataset_choice == '2'):
        dataset = pd.read_csv('./datasets/TCGA_InfoWithGrade.csv')
        X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_data(dataset, "Grade")

    print('Forma del dataset: %s' % (str(dataset.shape)))

    m = dataset.shape[0] #numero di campioni del dataset
    
    print('MO LO PROVIAMO A CASO')

    # Numero esempi
    m = 2

    # Input: 2 esempi con 3 features ciascuno
    X = np.array([
        [1.0, 0.0, 0.0],  # esempio 1
        [0.0, 1.0, 1.0]   # esempio 2
    ])

    Y = np.array([1, 1])  # Etichette (classe 1 per entrambi)
    num_classes = 3       # Numero di classi totali

    # One-hot encoding dinamico
    Y_one_hot = np.eye(num_classes)[Y]

    print(Y_one_hot)

    # Architettura: 1 hidden layer (4 neuroni), 1 output layer (3 classi)
    # Pesi e bias
    W = [
        np.ones((4, 3)),       # Layer 0 (4 neuroni, 3 input)
        np.array([             # Layer 1 (3 classi, 4 neuroni)
            [0.1, 0.1, 0.1, 0.1],   # pesi per classe 0
            [1.0, 1.0, 1.0, 1.0],   # pesi per classe 1 (più alti)
            [0.2, 0.2, 0.2, 0.2]    # pesi per classe 2
        ])
    ]

    b = [
        np.zeros((4,)),      # Bias per hidden layer
        np.zeros((3,))       # Bias per output layer
    ]   

    phi, labels, activations = forward_pass(X, W, b, 'relu', m)
    print("Predicted labels:", labels)
    print("Phi:", phi)

    grads = backpropagation(phi, Y_one_hot, W, b, 'relu', activations, m)

    print(grads)

"""     for i, activations in enumerate(activations):
        print(f"\nEsempio {i}:")
        for l, a in enumerate(activations):
            print(f"  Layer {l} (a^{l}): {a}") """


def print_menu(message):
    print(message)
    return input("Scegliere un'opzione:")

if __name__ == "__main__":
    main()