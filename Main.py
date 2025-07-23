# come funzione di attivazione usiamo ReLU e Tanh
# come funzione di regolarizzazione usiamo L1 e L2 (studiate a lezione)
import pandas as pd
import numpy as np

from preprocessing import *
from trainingFunctions import *
from modelEvaluation import *

def main():
    print('BENVENUTO!\n\n')
    
    dataset_choice = print_menu('Dataset disponibili:\n\n[1] Glioma Grading\n[2] TUNANDROM\n')

    print('\nHai scelto il dataset ' + dataset_choice)

    if(dataset_choice == '1'):
        dataset = pd.read_csv('./datasets/TCGA_InfoWithGrade.csv')
        num_classes = 2
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, "Grade")
    elif(dataset_choice == '2'):
        dataset = pd.read_csv('./datasets/TUANDROMD.csv')
        num_classes = 2
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, "Label")

    print('Forma del dataset: %s' % (str(dataset.shape)))

    m = len(X_train) #numero di campioni del training set
    n_features = dataset.shape[1] - 1

    # One-hot encoding dinamico
    Y_one_hot = np.eye(num_classes)[Y_train]

    # neurons = [n_features, 10, 12, num_classes]
    neurons = [n_features, 16, 32, 64, num_classes]

    W, b = weight_initializer(neurons)

    vW = [np.zeros_like(w) for w in W]
    vb = [np.zeros_like(bias) for bias in b]

    accuracy_per_epoch = []

    num_batches = int(np.ceil(m / MINI_BATCH_SIZE))
    for epoch in range(NUM_EPOCHS):
        idx = np.random.permutation(m)
        for batch in range(num_batches):
            batch_idx = idx[batch*MINI_BATCH_SIZE : (batch+1)*MINI_BATCH_SIZE]
            X_batch = X_train[batch_idx]
            Y_batch = Y_one_hot[batch_idx]

            # forward, backprop e aggiornamento
            phi, labels, activations = forward_pass(X_batch, W, b, 'relu', len(X_batch))
            
            dW, db = backpropagation(phi, Y_batch, W, b, 'relu', activations, len(X_batch))

            W, b, vW, vb = stochastic_gradient_with_momentum(dW, db, W, b, vW, vb)
            
        #Facciamo una forward pass così da poter valutare l'andamento dell'accuratezza durante le epoche
        phi, labels, activations = forward_pass(X_train, W, b, 'relu', len(X_train))
        accuracy, precision, recall, f1 = evaluate_model(labels, Y_train)
        accuracy_per_epoch.append(accuracy)

    
    plot(accuracy_per_epoch,"accuracy")
    
    # Valutazione modello sul test set
    print("PRESTAZIONI SU TEST SET\n")
    phi, labels, activations = forward_pass(X_test, W, b, 'relu', len(X_test))
    accuracy, precision, recall, f1 = evaluate_model(labels, Y_test)



def print_menu(message):
    print(message)
    return input("Scegliere un'opzione:")

if __name__ == "__main__":
    main()