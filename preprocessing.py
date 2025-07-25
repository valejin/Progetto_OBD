import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


def preprocess_data(dataset, dataset_choice, label_name, test_size=0.2, random_state=42):

    if(dataset_choice == '2'):
        # rimuove la colonna ID
        dataset = dataset.drop(columns=["ID"])
        # trasforma la colonna 'class': P -> 1, altro -> 0
        dataset['class'] = dataset['class'].apply(lambda x: 1 if x == 'P' else 0)
    elif(dataset_choice == '3'):
        dataset['Target'] = dataset['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

    # rimuove righe con valori mancanti (sia in X che Y)
    dataset = dataset.dropna(axis=0).reset_index(drop=True)

    # shuffle dei campioni
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # separazione feature e label
    X = dataset.drop(columns=[label_name])
    Y = dataset[label_name]

    # identifica solo le colonne numeriche per standardizzazione
    numeric_columns = X.select_dtypes(include=[np.number]).columns

    # standardizzazione delle feature numeriche (media=0, std=1)
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    # divisione in train/test set con stratificazione
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        stratify=Y if len(np.unique(Y)) > 1 else None,  # evita errore se Y ha una sola classe
        random_state=random_state
    )

    plot_istances(Y_train, Y_test)

    # conversione in numpy array per compatibilità
    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()
    

def plot_istances(Y_train, Y_test):

    # conta le classi nel training e test set
    train_counts = Counter(Y_train)
    test_counts = Counter(Y_test)

    # estrai le etichette (es. 0 e 1)
    labels = sorted(set(Y_train) | set(Y_test))

    # costruisci liste ordinate
    train_values = [train_counts.get(label, 0) for label in labels]
    test_values = [test_counts.get(label, 0) for label in labels]

    x = np.arange(len(labels))  # 0, 1 (per le due classi)
    width = 0.35

    # crea il grafico a barre affiancate
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, train_values, width, label='Train', color='skyblue')
    bar2 = ax.bar(x + width/2, test_values, width, label='Test', color='salmon')

    # etichette e legenda
    ax.set_xlabel('Classe')
    ax.set_ylabel('Numero di campioni')
    ax.set_title('Distribuzione delle classi in Train e Test set')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # aggiungi i numeri sopra le barre
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