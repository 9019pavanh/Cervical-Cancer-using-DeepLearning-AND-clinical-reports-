"""Reproducible training entry point for image and clinical experiments."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG, ensure_output_directories
from dataset import (
    CervicalImageDataset,
    audit_images,
    audit_project,
    discover_images,
    grouped_stratified_split,
    load_image_split,
    prepare_clinical_data,
    save_image_splits,
    split_clinical_indices,
)
from models import ClinicalMLP, ImageClassifier
from preprocessing import build_clinical_preprocessor, build_image_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probability = torch.exp(-cross_entropy)
        return (((1.0 - probability) ** self.gamma) * cross_entropy).mean()


def image_loaders(batch_size: int) -> Tuple[Dict[str, DataLoader], list, Dict[str, int]]:
    required = [CONFIG.data.split_dir / name for name in ("train.csv", "val.csv", "test.csv")]
    if not all(path.exists() for path in required):
        clean_records, _, _ = audit_images(discover_images())
        save_image_splits(grouped_stratified_split(clean_records))
    records = {
        "train": load_image_split("train"),
        "validation": load_image_split("validation"),
        "test": load_image_split("test"),
    }
    classes = sorted({record.label for split in records.values() for record in split})
    class_to_index = {label: index for index, label in enumerate(classes)}
    train_transform, evaluation_transform = build_image_transforms()
    datasets = {
        name: CervicalImageDataset(
            split,
            class_to_index,
            train_transform if name == "train" else evaluation_transform,
        )
        for name, split in records.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=CONFIG.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
    return loaders, classes, class_to_index


def class_weights(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = [loader.dataset.class_to_index[record.label] for record in loader.dataset.records]
    counts = np.bincount(labels, minlength=num_classes)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_image_epoch(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    losses, targets, predictions, probabilities = [], [], [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(criterion(logits, labels).item() * len(labels))
        probs = torch.softmax(logits, 1)
        targets.extend(labels.cpu().numpy())
        predictions.extend(probs.argmax(1).cpu().numpy())
        probabilities.extend(probs.cpu().numpy())
    return {
        "loss": float(sum(losses) / len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")),
    }


def train_image(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_directories()
    loaders, classes, class_to_index = image_loaders(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ImageClassifier(args.model, len(classes), pretrained=True, dropout=args.dropout).to(device)
    weights = class_weights(loaders["train"], len(classes), device)
    criterion = FocalLoss(weights, args.focal_gamma, args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    best_f1, stale_epochs = -1.0, 0
    checkpoint_path = CONFIG.model_dir / f"{args.model}_best.pth"
    history_path = CONFIG.result_dir / f"{args.model}_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=[
            "epoch", "train_loss", "train_accuracy", "validation_loss",
            "validation_accuracy", "validation_macro_f1", "validation_roc_auc", "learning_rate",
        ])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels, _ in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
            validation = evaluate_image_epoch(model, loaders["validation"], criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / total,
                "train_accuracy": correct / total,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "validation_roc_auc": validation["roc_auc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            history_file.flush()
            print(json.dumps(row))
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                stale_epochs = 0
                torch.save({
                    "architecture": args.model,
                    "class_names": classes,
                    "class_to_index": class_to_index,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation,
                    "seed": args.seed,
                    "epoch": epoch,
                }, checkpoint_path)
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="Validation accuracy")
    axes[1].plot(history["epoch"], history["validation_macro_f1"], label="Validation macro-F1")
    axes[1].set_title("Performance")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(CONFIG.result_dir / f"{args.model}_training_curves.png", dpi=180)
    plt.close(figure)


def train_clinical(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_directories()
    features, target = prepare_clinical_data()
    split_indices = split_clinical_indices(target)
    preprocessor = build_clinical_preprocessor(features.iloc[split_indices["train"]])
    train_x = preprocessor.fit_transform(features.iloc[split_indices["train"]]).astype("float32")
    validation_x = preprocessor.transform(features.iloc[split_indices["validation"]]).astype("float32")
    train_y = target.iloc[split_indices["train"]].to_numpy(dtype="float32")
    validation_y = target.iloc[split_indices["validation"]].to_numpy(dtype="float32")
    joblib.dump(preprocessor, CONFIG.model_dir / "clinical_preprocessor.joblib")
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ClinicalMLP(train_x.shape[1], dropout=args.dropout).to(device)
    positives = max(train_y.sum(), 1.0)
    pos_weight = torch.tensor([(len(train_y) - positives) / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_auc, stale_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            _, logits = model(torch.from_numpy(validation_x).to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        auc = roc_auc_score(validation_y, probabilities)
        print(json.dumps({"epoch": epoch, "validation_roc_auc": float(auc)}))
        if auc > best_auc:
            best_auc, stale_epochs = auc, 0
            torch.save({
                "input_dim": train_x.shape[1],
                "feature_columns": features.columns.tolist(),
                "model_state_dict": model.state_dict(),
                "validation_roc_auc": float(auc),
            }, CONFIG.model_dir / "clinical_mlp_best.pth")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break


def refuse_unlinked_multimodal_training() -> None:
    report = audit_project(write_report=True)
    if not report.modalities_linked:
        raise RuntimeError(
            "Multimodal training refused: no verified image-to-patient linkage exists. "
            "Provide a linkage CSV containing image_path and patient_id, and a matching patient_id in clinical data."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["image", "clinical", "multimodal"], default="image")
    parser.add_argument("--model", choices=CONFIG.supported_models, default="resnet50")
    parser.add_argument("--epochs", type=int, default=CONFIG.training.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.training.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.training.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CONFIG.training.weight_decay)
    parser.add_argument("--dropout", type=float, default=CONFIG.training.dropout)
    parser.add_argument("--label-smoothing", type=float, default=CONFIG.training.label_smoothing)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=CONFIG.training.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=CONFIG.data.seed)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.task == "image":
        train_image(arguments)
    elif arguments.task == "clinical":
        train_clinical(arguments)
    else:
        refuse_unlinked_multimodal_training()
