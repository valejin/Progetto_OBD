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
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, "quality")
    elif(dataset_choice == '2'):
        dataset = pd.read_csv('./datasets/TCGA_InfoWithGrade.csv')
        num_classes = 2
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, "Grade")

    print('Forma del dataset: %s' % (str(dataset.shape)))

    m = len(X_train) #numero di campioni del training set

    # One-hot encoding dinamico
    Y_one_hot = np.eye(num_classes)[Y_train]

    neurons = [23, 10, 12, num_classes]

    W, b = weight_initializer(neurons)

    phi, labels, activations = forward_pass(X_train, W, b, 'relu', m)
    print("Predicted labels:", labels)
    #print("Phi:", phi)

    grads = backpropagation(phi, Y_one_hot, W, b, 'relu', activations, m)

    print(grads)


def print_menu(message):
    print(message)
    return input("Scegliere un'opzione:")

if __name__ == "__main__":
    main()