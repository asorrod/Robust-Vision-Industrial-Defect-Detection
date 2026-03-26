import torch
from sklearn.metrics import(accuracy_score, 
                            precision_recall_fscore_support,
                            confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json

def compute_classification_metrics(y_true, y_pred, corruption, severity, save_dir):
    acc = accuracy_score(y_true=y_true,
                         y_pred=y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true,
                                                               y_pred=y_pred,
                                                               average="macro")
    
    cm = confusion_matrix(y_true=y_true,
                          y_pred=y_pred)
    
    metrics = {
        "corruption": corruption,
        "severity": severity,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

    metrics_to_save = metrics.copy()
    metrics_to_save["confusion_matrix"] = cm.tolist()

    with open(save_dir / "results.txt", "a") as f:
        json.dump(metrics_to_save, f)
        f.write("\n")
    
    return metrics

def plot_confusion_matrix(metrics, class_names, save_dir):

    cm = metrics["confusion_matrix"]

    cm_norm = 100 * cm.astype("float") / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm).astype(str)

    annot = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]:.1f}%)"

    plt.figure(figsize=(8,6))


    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(
        f"{metrics['corruption']} | severity {metrics['severity']} | acc {metrics['accuracy']:.3f}"
    )

    filename = f"cm_{metrics['corruption']}_sev_{metrics['severity']}_acc_{metrics['accuracy']:.3f}.png"

    plt.tight_layout()
    plt.savefig(save_dir / filename)
    plt.close()