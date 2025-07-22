import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#SUPPONIAMO CHE ABBIAMO SOLO CLASSIFICAZIONE BINARIA
def evaluate_model(phi_label, y):

    #Confronto elemento per elemento e conteggio degli uguali
    """ n_equal = np.sum(phi_label == y)

    accuracy = (n_equal/len(y))*100

    # precision = TP/TP+FP

    # Confronta se entrambi sono 1 nella stessa posizione
    TP = np.sum((phi_label == 1) & (y == 1))
    FP = np.sum((phi_label == 1) & (y != 1))
    FN = np.sum((phi_label != 1) & (y == 1))

    precision = TP / (TP + FP) 
    recall = TP / (TP + FN) """


    # Accuratezza generale
    accuracy = accuracy_score(y, phi_label)

    # Precision e Recall per classe
    precision = precision_score(y, phi_label, average=None)
    recall = recall_score(y, phi_label, average=None)
    f1 = f1_score(y, phi_label, average=None)

    # Stampa risultati
    print(f"Accuracy: {accuracy:.2f}\n")

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Classe {i} -> Precision: {p:.2f}, Recall: {r:.2f}, F1-score: {f:.2f}")

    # Visualizza la confusion matrix
    cm = confusion_matrix(y, phi_label)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Confusion Matrix")
    # Salvataggio prima di mostrare
    plt.savefig("./plots/evaluation/confusion_matrix.png", dpi=300, bbox_inches='tight')  
    plt.show()


    return accuracy, precision, recall, f1


# DA SISTEMARE, NON FUNZIONA
def save_model(W, b, file_name):

    # Converti in numpy array se sono liste
    W = np.array(W)
    b = np.array(b)

    np.savetxt("W.csv", W.reshape(W.shape[0], -1), delimiter=",")  # appiattisco dimensioni oltre la prima
    np.savetxt("b.csv", b, delimiter=",")

    np.savez("./models/" + file_name, W=W, b=b)
    
    print(f"Modello salvato in {file_name}")