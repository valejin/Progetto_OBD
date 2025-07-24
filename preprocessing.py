import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


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

def preprocess_data(dataset, dataset_choice, label_name, test_size=0.2, random_state=42):

    if(dataset_choice == '2'):
        # Rimuove la colonna ID
        dataset = dataset.drop(columns=["ID"])
        # Trasforma la colonna 'class': P -> 1, altro -> 0
        dataset['class'] = dataset['class'].apply(lambda x: 1 if x == 'P' else 0)
    elif(dataset_choice == '3'):
        dataset['Target'] = dataset['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

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

    plot_istances(Y_train, Y_test)

    # Conversione in numpy array per compatibilità
    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()
    

def plot_istances(Y_train, Y_test):

    # Conta le classi nel training e test set
    train_counts = Counter(Y_train)
    test_counts = Counter(Y_test)

    # Estrai le etichette (es. 0 e 1)
    labels = sorted(set(Y_train) | set(Y_test))

    # Costruisci liste ordinate
    train_values = [train_counts.get(label, 0) for label in labels]
    test_values = [test_counts.get(label, 0) for label in labels]

    x = np.arange(len(labels))  # 0, 1 (per le due classi)
    width = 0.35

    # Crea il grafico a barre affiancate
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, train_values, width, label='Train', color='skyblue')
    bar2 = ax.bar(x + width/2, test_values, width, label='Test', color='salmon')

    # Etichette e legenda
    ax.set_xlabel('Classe')
    ax.set_ylabel('Numero di campioni')
    ax.set_title('Distribuzione delle classi in Train e Test set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Aggiungi i numeri sopra le barre
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()