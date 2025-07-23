import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


'''def preprocess_data(dataset, label_name):

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    #Separa le feature dalle label nel dataset
    X = dataset.drop(columns = [label_name], axis = 1)
    Y = dataset[label_name]

    #Elimina le righe che hanno valori mancanti
    X = X.dropna(axis = 0)

    #Standardizziamo i valori per avere media 0 e varianza 1
    if(label_name != 'Grade'):
        for column in X.columns:
            X[column] = (X[column] - X[column].mean()) / X[column].std()

    #Separiamo il dataset in training set, validation set e test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()
'''

def preprocess_data(dataset, label_name, test_size=0.2, random_state=42):

    # Rimuove righe con valori mancanti (sia in X che Y)
    dataset = dataset.dropna(axis=0).reset_index(drop=True)

    # Shuffle
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Separazione feature e label
    X = dataset.drop(columns=[label_name])
    Y = dataset[label_name]

    # Identifica solo le colonne numeriche per standardizzazione
    numeric_columns = X.select_dtypes(include=[np.number]).columns

    # Standardizzazione delle feature numeriche (media=0, std=1)
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # Divisione in train/test set con stratificazione
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        stratify=Y if len(np.unique(Y)) > 1 else None,  # evita errore se Y ha una sola classe
        random_state=random_state
    )

    # Conversione in numpy array per compatibilità
    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()
    