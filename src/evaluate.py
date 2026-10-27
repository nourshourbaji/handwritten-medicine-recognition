import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
    confusion_matrix,
    classification_report
)

from src.models import build_model


def load_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = build_model(config["model"], len(checkpoint["class_to_idx"]),
                        dropout=config["dropout"], **config["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    return model, checkpoint


def load_results(results_dir):
    """Read every experiment .json in a folder into a list."""
    return [json.load(open(p)) for p in sorted(Path(results_dir).glob("*.json"))]


def results_table(results):
    rows = [{
        "name": r["name"],
        "best_val_acc": r["best_val_acc"],
        "best_epoch": r["best_epoch"],
        "final_train_acc": r["history"]["train_acc"][-1],
        "num_params": r["num_params"],
        "train_time_min": round(r["train_time_min"], 1),
    } for r in results]
    return pd.DataFrame(rows).sort_values("best_val_acc", ascending=False).reset_index(drop=True)


@torch.no_grad()
def get_predictions(model, loader, device="cuda"):
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        outputs = model(images.to(device))
        all_probs.append(torch.softmax(outputs, dim=1).cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = probs.argmax(1)

    return y_true, y_pred, probs


def compute_metrics(y_true, y_pred, probs):
    # macro = compute the metric for each class, then average (every class counts equally)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "top5_accuracy": top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(probs.shape[1])),
    }


def per_class_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T.loc[class_names]
    return df.sort_values("f1-score")


def most_confused_pairs(y_true, y_pred, class_names, n=10):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    np.fill_diagonal(cm, 0)

    pairs = [(class_names[i], class_names[j], cm[i, j])
             for i in range(len(class_names)) for j in range(len(class_names)) if cm[i, j] > 0]

    df = pd.DataFrame(pairs, columns=["true", "predicted", "count"])
    return df.sort_values("count", ascending=False).head(n).reset_index(drop=True)


def plot_history(history, title="", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_comparison(results, metric="val_acc", title="", save_path=None):
    """Plot one metric (e.g. validation accuracy) of several experiments on the same axes."""
    plt.figure(figsize=(10, 5))

    for r in results:
        values = r["history"][metric]
        plt.plot(range(1, len(values) + 1), values, label=r["name"])

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    # normalize each row so values are the fraction of each true class
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")

    plt.figure(figsize=(20, 18))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.03)
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix (Test Set)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def show_misclassified(dataset, y_true, y_pred, class_names, n=12, save_path=None):
    wrong = np.where(y_true != y_pred)[0][:n]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    for ax, idx in zip(axes.flat, wrong):
        image, _ = dataset[idx]
        image = (image - image.min()) / (image.max() - image.min())   # back to [0, 1] for display
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
