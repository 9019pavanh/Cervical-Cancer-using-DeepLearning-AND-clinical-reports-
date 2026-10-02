"""
utils.py
========
Core Utilities for Cervical Cancer Classification Project.

Includes:
- Reproducibility & seed management
- Performance evaluation metrics (Accuracy, Precision, Recall, Macro/Weighted F1)
- Inverse class frequency weighting for imbalanced datasets
- Checkpoint management (save, load best model and latest weights)
- Visual analytics: Training loss/accuracy curves and confusion matrix heatmaps
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for python random, numpy, and pytorch for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> Dict[str, float]:
    """
    Compute classification performance metrics:
    - Accuracy
    - Macro Precision, Recall, F1-Score
    - Weighted Precision, Recall, F1-Score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
    }


def compute_class_weights(labels: List[int], num_classes: int) -> torch.FloatTensor:
    """
    Compute balanced class weights inversely proportional to class frequencies:
        weight_c = total_samples / (num_classes * count_c)
    Returns a normalized torch tensor suitable for nn.CrossEntropyLoss.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total_samples / (num_classes * counts[i])
        else:
            weights[i] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Union[str, Path],
    filename: str = "latest_checkpoint.pth",
    best_filename: str = "best_model.pth",
) -> Path:
    """
    Save training state dictionary to disk. If is_best is True, also saves/overwrites best_model.pth.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = checkpoint_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_path = checkpoint_dir / best_filename
        torch.save(state, best_path)
        return best_path

    return filepath


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load weights and optimizer state from a checkpoint file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def plot_training_curves(history: Dict[str, List[float]], save_path: Union[str, Path]) -> None:
    """
    Generate and save Loss and Accuracy learning curves across training and validation epochs.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss Curve ---
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#d62728", lw=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # --- Accuracy Curve ---
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Accuracy", color="#ff7f0e", lw=2)
    if "val_f1_macro" in history:
        ax2.plot(epochs, [f * 100 for f in history["val_f1_macro"]], "^-.", label="Val Macro F1", color="#9467bd", lw=1.5)
    ax2.set_title("Training & Validation Accuracy / F1", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Score (%)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """
    Generate and save a formatted confusion matrix heatmap.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))

    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)

    # Label axes
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate counts inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight="bold"
            )

    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
