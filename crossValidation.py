import numpy as np
import copy

from config import *
from trainingFunctions import *
from modelEvaluation import *
from itertools import combinations

def k_fold_division(X, Y, k=5):
    
    np.random.seed(42)
    m = X.shape[0]
    idx = np.random.permutation(m)  # shuffle gli indici
    fold_size = m // k

    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i != k - 1 else m  # ultimo fold prende tutto ciò che resta
        val_idx = idx[start:end]
        train_idx = np.concatenate([idx[:start], idx[end:]])
        folds.append((train_idx, val_idx))

    return folds

def manual_stratified_k_fold(X, Y, k=5, seed=42):
    np.random.seed(seed)
    folds = []

    # Trova le classi uniche
    classes = np.unique(Y)
    class_indices = {cls: np.where(Y == cls)[0] for cls in classes}

    # Per ogni classe, shuffla e dividila in k blocchi
    class_folds = {cls: [] for cls in classes}

    for cls in classes:
        indices = class_indices[cls]
        np.random.shuffle(indices)
        fold_sizes = [len(indices) // k] * k
        for i in range(len(indices) % k):
            fold_sizes[i] += 1
        current = 0
        for size in fold_sizes:
            class_folds[cls].append(indices[current:current+size])
            current += size

    # Ora crea i fold combinando in ogni fold una porzione di ogni classe
    for i in range(k):
        val_idx = np.concatenate([class_folds[cls][i] for cls in classes])
        train_idx = np.concatenate([np.concatenate(class_folds[cls][:i] + class_folds[cls][i+1:]) for cls in classes])
        folds.append((train_idx, val_idx))

    return folds


def cross_validation(X, Y, n_features, num_classes, inizialization, activation_function):
    
    folds = manual_stratified_k_fold(X, Y, k=5)
    neurons_combinations = [list(c) for c in combinations(HIDDEN_LAYER_NEURONS_LIST, L)]

    all_models = []
    all_mean_acc = []

    for comb in neurons_combinations:

        neurons = [n_features] + comb + [num_classes]
        
        for lambd in LAMBDA_VALUES_LIST:

            val_accuracies = []

            all_models.append((copy.deepcopy(neurons), lambd))

            for train_idx, val_idx in folds:
                X_train, Y_train = X[train_idx], Y[train_idx]
                X_val, Y_val = X[val_idx], Y[val_idx]

                Y_one_hot = np.eye(num_classes)[Y_train]

                # Inizializzazione dei pesi e momenti
                W, b = general_weight_initializer(neurons, inizialization)

                vW = [np.zeros_like(w) for w in W]
                vb = [np.zeros_like(bi) for bi in b]

                num_batches = int(np.ceil(len(X_train) / MINI_BATCH_SIZE))

                k = 1
                loss_prev = float('inf')

                # === Training loop ===
                for epoch in range(NUM_EPOCHS):
                    idx = np.random.permutation(len(X_train))
                    for batch in range(num_batches):
                        batch_idx = idx[batch*MINI_BATCH_SIZE : (batch+1)*MINI_BATCH_SIZE]
                        X_batch = X_train[batch_idx]
                        Y_batch = Y_one_hot[batch_idx]

                        phi, labels, activations = forward_pass(X_batch, W, b, activation_function, len(X_batch))

                        loss_curr = compute_loss(phi, Y_batch, W, lambd)


                        dW, db = backpropagation(phi, Y_batch, W, b, activation_function, activations, len(X_batch))
                        W, b, vW, vb = stochastic_gradient_with_momentum(dW, db, W, b, vW, vb, lambd, k, loss_prev, loss_curr)

                        k += 1
                        loss_prev = loss_curr

                # === Validation ===
                phi_val, labels_val, _ = forward_pass(X_val, W, b, activation_function, len(X_val))
                acc, _, _, _ = evaluate_model(labels_val, Y_val)
                val_accuracies.append(acc)

            # Media delle performance su tutti i fold per questo lambda
            print("\nModello: ", neurons)
            print(f"λ={lambd} -> accuracy media: {np.mean(val_accuracies)}")

            all_mean_acc.append(np.mean(val_accuracies))

    #plot_lambda_vs_accuracy(LAMBDA_VALUES_LIST, all_mean_acc)

    return all_models[np.argmax(all_mean_acc)]


def plot_lambda_vs_accuracy(lambda_values, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_values, accuracies, marker='o')
    plt.xscale('log')  # scala log per vedere bene piccole variazioni
    plt.xlabel('λ (regolarizzazione L2)')
    plt.ylabel('Accuracy media (val set)')
    plt.title('Accuracy media su validation set vs λ')
    plt.grid(True)
    plt.tight_layout()
    plt.show()