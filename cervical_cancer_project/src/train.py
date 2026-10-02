"""
train.py
========
Phase 4: Model Building and Training Engine for Cervical Cancer Classification.

Features:
- Transfer learning architectures: ResNet-18, ResNet-50, EfficientNet-B0, MobileNetV3-Large
- Custom classification heads with Dropout for regularization
- Imbalanced class weighting option
- AdamW optimizer & Cosine Annealing learning rate schedule
- Early stopping based on validation loss / macro-F1
- Automatic checkpointing of best and latest weights to `models/`
- Automatic visualization export (loss/accuracy curves, confusion matrix) to `results/`
- Evaluation on hold-out Test split post-training
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# Ensure local imports work whether executed from root or src/
try:
    from dataset_config import DATASETS, MODELS_DIR, RESULTS_DIR
    from dataset_loader import get_dataloaders
    from utils import (
        compute_class_weights,
        compute_metrics,
        load_checkpoint,
        plot_confusion_matrix,
        plot_training_curves,
        save_checkpoint,
        set_seed,
    )
except ImportError:
    from src.dataset_config import DATASETS, MODELS_DIR, RESULTS_DIR
    from src.dataset_loader import get_dataloaders
    from src.utils import (
        compute_class_weights,
        compute_metrics,
        load_checkpoint,
        plot_confusion_matrix,
        plot_training_curves,
        save_checkpoint,
        set_seed,
    )


# ==============================================================================
# MODEL FACTORY
# ==============================================================================
def build_model(
    model_name: str = "resnet18",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Constructs a CNN backbone and adapts the classification head to num_classes.
    """
    name = model_name.lower().strip()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Choose from: "
            f"['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large']"
        )

    return model


# ==============================================================================
# TRAIN & VALIDATE SINGLE EPOCH
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """
    Execute one training epoch with gradient descent and running metric tracking.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        # Gradient clipping to stabilize deep transfer learning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2%}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    """
    Evaluate model performance on a DataLoader (Validation or Test).
    """
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    pbar = tqdm(loader, desc=f"Evaluating [{desc}]", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total
    metrics = compute_metrics(all_targets, all_preds)
    return epoch_loss, metrics, all_targets, all_preds


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_pipeline(args: argparse.Namespace) -> None:
    """
    Orchestrates data loading, model compilation, training loop, validation,
    early stopping, checkpoint saving, test evaluation, and artifact generation.
    """
    set_seed(args.seed)

    device_str = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    device = torch.device(device_str)

    print("=" * 80)
    print("CERVICAL CANCER CLASSIFICATION - MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Dataset:       {args.dataset}")
    print(f"Model:         {args.model}")
    print(f"Device:        {device} (CUDA Available: {torch.cuda.is_available()})")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Early Stop:    Patience = {args.patience}")
    print("=" * 80)

    # 1. Prepare DataLoaders
    print("\n[*] Loading dataset and creating DataLoaders...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(class_names)
    print(f"  Target Classes ({num_classes}): {class_names}")

    # 2. Build Model
    print(f"\n[*] Initializing {args.model} backbone (pretrained={args.pretrained})...")
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    model = model.to(device)

    # 3. Loss & Optimizer Setup
    if args.use_class_weights:
        print("[*] Calculating balanced inverse class weights...")
        # Gather all training labels directly from samples list (instant, no image loads)
        train_labels = [label for _, label in train_loader.dataset.samples]
        weights = compute_class_weights(train_labels, num_classes=num_classes).to(device)
        print(f"  Class Weights: {[round(w.item(), 3) for w in weights]}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 4. Tracking & Checkpointing Directories
    models_save_dir = Path(args.models_dir) / args.dataset
    results_save_dir = Path(args.results_dir) / args.dataset
    models_save_dir.mkdir(parents=True, exist_ok=True)
    results_save_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
    }

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print("\n[*] Starting model training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
        )

        # Validate
        val_loss, val_metrics, val_targets, val_preds = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Val {epoch}",
        )

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_macro"]

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1)

        # Display epoch summary
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} Macro-F1: {val_f1:.4f}"
        )

        # Check for best model based on Macro-F1
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint state
            state = {
                "epoch": epoch,
                "model_name": args.model,
                "dataset_name": args.dataset,
                "num_classes": num_classes,
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "val_accuracy": val_acc,
            }
            best_saved_path = save_checkpoint(
                state=state,
                is_best=True,
                checkpoint_dir=models_save_dir,
                filename=f"{args.model}_latest.pth",
                best_filename=f"{args.model}_best.pth",
            )
            print(f"  --> Saved new best model to: {best_saved_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping triggered: No improvement for {args.patience} epochs.")
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETED IN {total_time / 60:.2f} MINUTES")
    print(f"Best Epoch: {best_epoch} with Val Macro-F1: {best_val_f1:.4f} (Val Loss: {best_val_loss:.4f})")
    print("=" * 80)

    # 5. Export Visual Artifacts
    curves_path = results_save_dir / f"{args.model}_training_curves.png"
    plot_training_curves(history, save_path=curves_path)
    print(f"[*] Saved learning curves to: {curves_path}")

    # 6. Evaluate on Hold-out Test Set with Best Model
    best_model_file = models_save_dir / f"{args.model}_best.pth"
    if best_model_file.exists():
        print(f"\n[*] Loading best weights from {best_model_file} for Test evaluation...")
        load_checkpoint(best_model_file, model, device=device)

        test_loss, test_metrics, test_targets, test_preds = evaluate_loader(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            desc="Hold-out Test",
        )

        cm_path = results_save_dir / f"{args.model}_test_confusion_matrix.png"
        plot_confusion_matrix(
            y_true=test_targets,
            y_pred=test_preds,
            class_names=class_names,
            save_path=cm_path,
            title=f"Test Confusion Matrix ({args.dataset} - {args.model})",
        )

        print("\n" + "-" * 80)
        print("FINAL TEST EVALUATION METRICS:")
        print(f"  Test Loss:              {test_loss:.4f}")
        print(f"  Test Accuracy:          {test_metrics['accuracy']:.2%}")
        print(f"  Test Macro Precision:   {test_metrics['precision_macro']:.4f}")
        print(f"  Test Macro Recall:      {test_metrics['recall_macro']:.4f}")
        print(f"  Test Macro F1-Score:    {test_metrics['f1_macro']:.4f}")
        print(f"  Test Weighted F1-Score: {test_metrics['f1_weighted']:.4f}")
        print(f"  Saved Test Confusion Matrix to: {cm_path}")
        print("-" * 80)

        metrics_path = results_save_dir / f"{args.model}_test_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "model_name": args.model,
                    "dataset_name": args.dataset,
                    "best_epoch": best_epoch,
                    "grouped_split": args.dataset == "SIPaKMeD",
                    "test_loss": test_loss,
                    "summary_metrics": test_metrics,
                },
                metrics_file,
                indent=2,
            )
        print(f"[*] Saved test metrics to: {metrics_path}")


# ==============================================================================
# ENTRY POINT & ARGUMENT PARSER
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deep Learning model for cervical cancer diagnosis.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SIPaKMeD",
        choices=list(DATASETS.keys()),
        help="Target dataset name (default: SIPaKMeD)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"],
        help="Model architecture (default: resnet18)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay regularization (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout probability (default: 0.2)")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Cross-entropy label smoothing (default: 0.1)",
    )
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows CPU stability)")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false", help="Do not use pretrained weights")
    parser.add_argument("--use_class_weights", action="store_true", help="Apply inverse class weights in loss function")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU training even if CUDA is detected")
    parser.add_argument("--models_dir", type=str, default=str(MODELS_DIR), help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR), help="Directory to save plots & results")
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_pipeline(cli_args)
