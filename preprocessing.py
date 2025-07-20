import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

def preprocess_data(dataset, label_name):

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

    