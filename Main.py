# come funzione di attivazione usiamo ReLU e Tanh
# come funzione di regolarizzazione usiamo L1 e L2 (studiate a lezione)
import pandas as pd
import numpy as np

from preprocessing import *
from trainingFunctions import *
from modelEvaluation import *
from crossValidation import *

def main():
    print('BENVENUTO!\n\n')
    
    dataset_choice = print_menu('Dataset disponibili:\n\n[1] Glioma Grading (839 campioni)\n[2] DARWIN (174 campioni)\n[3] Academic Dropout (4424 campioni)\n')

    activation_choice = print_menu('Funzioni di attivazione disponibili:\n[1] ReLU\n[2] Tanh\n')

    reg_choice = print_menu('Regolarizzazioni disponibili:\n[1] L1\n[2] L2\n')

    print('\nHai scelto il dataset ' + dataset_choice)

    if(dataset_choice == '1'):
        dataset = pd.read_csv('./datasets/TCGA_InfoWithGrade.csv')
        num_classes = 2
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, dataset_choice, "Grade")
        n_features = dataset.shape[1] - 1
    elif(dataset_choice == '2'):
        dataset = pd.read_csv('./datasets/DARWIN.csv')
        num_classes = 2
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, dataset_choice, "class")
        n_features = dataset.shape[1] - 2
    elif(dataset_choice == '3'):
        dataset = pd.read_csv('./datasets/academicDropout.csv')
        num_classes = 2
        X_train, Y_train, X_test, Y_test = preprocess_data(dataset, dataset_choice, "Target")
        n_features = dataset.shape[1] - 1

    if(activation_choice == '1'):
        activation_function = "relu"
        inizialization = "he"
    elif(activation_choice == '2'):
        activation_function = "tanh"
        inizialization = "xavier"

    if(reg_choice == '1'):
        regularization = 'l1'
    elif(reg_choice == '2'):
        regularization = 'l2'

    print('Forma del dataset: %s' % (str(dataset.shape)))

    m = len(X_train) #numero di campioni del training set
    
    best_model = cross_validation(X_train, Y_train, n_features, num_classes, inizialization, activation_function, regularization)
    neurons = best_model[0]
    best_lambda = best_model[1]
    print("\nIl modello migliore è stato: ", best_model)

    # One-hot encoding dinamico
    Y_one_hot = np.eye(num_classes)[Y_train]

    W, b = general_weight_initializer(neurons, inizialization)

    vW = [np.zeros_like(w) for w in W]
    vb = [np.zeros_like(bias) for bias in b]

    accuracy_per_epoch = []

    k = 1
    loss_prev = float('inf')

    num_batches = int(np.ceil(m / MINI_BATCH_SIZE))
    for epoch in range(NUM_EPOCHS):
        idx = np.random.permutation(m)
        for batch in range(num_batches):
            batch_idx = idx[batch*MINI_BATCH_SIZE : (batch+1)*MINI_BATCH_SIZE]
            X_batch = X_train[batch_idx]
            Y_batch = Y_one_hot[batch_idx]

            # forward, backprop e aggiornamento
            phi, labels, activations = forward_pass(X_batch, W, b, activation_function, len(X_batch))

            loss_curr = compute_loss(phi, Y_batch, W, best_lambda)
            
            dW, db = backpropagation(phi, Y_batch, W, b, activation_function, activations, len(X_batch))

            W, b, vW, vb = stochastic_gradient_with_momentum(dW, db, W, b, vW, vb, best_lambda, k, loss_prev, loss_curr, reg_type = regularization)

            k += 1
            loss_prev = loss_curr
            
        #Facciamo una forward pass così da poter valutare l'andamento dell'accuratezza durante le epoche
        phi, labels, activations = forward_pass(X_train, W, b, activation_function, len(X_train))
        accuracy, precision, recall, f1 = evaluate_model(labels, Y_train)
        accuracy_per_epoch.append(accuracy)

    
    plot(accuracy_per_epoch,"accuracy")
    
    # Valutazione modello sul test set
    print("\n\n*---- PRESTAZIONI SU TEST SET ----*\n")
    phi, labels, activations = forward_pass(X_test, W, b, activation_function, len(X_test))
    accuracy, precision, recall, f1 = evaluate_model(labels, Y_test)
    show_confusion_matrix(labels, Y_test)


def print_menu(message):
    print(message)
    return input("Scegliere un'opzione:")

if __name__ == "__main__":
    main()