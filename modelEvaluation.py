import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(phi_label, y):

    # accuratezza generale
    accuracy = accuracy_score(y, phi_label)

    # precision,recall e f1 per classe
    precision = precision_score(y, phi_label, average=None)
    recall = recall_score(y, phi_label, average=None)
    f1 = f1_score(y, phi_label, average=None)

    print(f"\nAccuracy: {accuracy:.2f}\n")

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Classe {i} -> Precision: {p:.2f}, Recall: {r:.2f}, F1-score: {f:.2f}")

    
    return accuracy, precision, recall, f1


def show_confusion_matrix(phi_label, y):
    # visualizza la matrice di confusione
    cm = confusion_matrix(y, phi_label)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel("Predetto")
    plt.ylabel("Reale")
    plt.title("Confusion Matrix")
    plt.show()

    
# plot della metrica rispetto all'epoca
def plot(metric_per_epoch, metric):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(metric_per_epoch) + 1), metric_per_epoch, marker='o')
    plt.xlabel("Epoche")
    plt.ylabel(metric)
    plt.title("Andamento di " + metric + " durante il training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()