"""
evaluate.py
===========
Phase 5: Standalone Model Evaluation & Inference Engine for Cervical Cancer Detection.

Features:
- Comprehensive evaluation of trained checkpoints on Test, Validation, or Train splits
- Full per-class classification report (Precision, Recall, F1-Score, Support)
- Confusion matrix heatmap export
- Multiclass One-vs-Rest ROC curves and Macro/Micro ROC-AUC calculation
- Single-image inference mode: Predicts cell category and class confidence for any image
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from dataset_config import (
        DATASETS,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        MODELS_DIR,
        RESULTS_DIR,
    )
    from dataset_loader import CervicalCancerDataset, get_dataloaders, get_transforms
    from train import build_model
    from utils import compute_metrics, load_checkpoint, plot_confusion_matrix
except ImportError:
    from src.dataset_config import (
        DATASETS,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        MODELS_DIR,
        RESULTS_DIR,
    )
    from src.dataset_loader import CervicalCancerDataset, get_dataloaders, get_transforms
    from src.train import build_model
    from src.utils import compute_metrics, load_checkpoint, plot_confusion_matrix


# ==============================================================================
# BATCH EVALUATION WITH DETAILED METRICS
# ==============================================================================
@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    save_dir: Optional[Path] = None,
    split_name: str = "Test",
    model_name: str = "model",
    use_tta: bool = False,
) -> Dict[str, float]:
    """
    Evaluates model performance across a full DataLoader.
    Generates:
    - Detailed per-class classification report
    - Confusion matrix heatmap
    - Multiclass ROC curves and ROC-AUC score
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        if use_tta:
            horizontal_probs = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            vertical_probs = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
            probs = (probs + horizontal_probs + vertical_probs) / 3.0
        preds = torch.argmax(probs, dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    num_classes = len(class_names)

    # 1. Summary Metrics
    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY [{split_name.upper()} SET]")
    print("=" * 80)
    print(f"Total Evaluated Samples: {len(y_true):,}")
    print(f"Top-1 Overall Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Macro Average Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Average Recall:    {metrics['recall_macro']:.4f}")
    print(f"Macro Average F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score:       {metrics['f1_weighted']:.4f}")
    print("-" * 80)

    # 2. Detailed Per-Class Classification Report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\nPER-CLASS CLASSIFICATION REPORT:")
    print(report_text)

    # 3. Multiclass ROC-AUC (One-vs-Rest)
    try:
        y_true_onehot = np.eye(num_classes)[y_true]
        roc_auc_macro = roc_auc_score(
            y_true_onehot, y_probs, multi_class="ovr", average="macro"
        )
        print(f"Multiclass ROC-AUC (Macro OvR): {roc_auc_macro:.4f}")
        metrics["roc_auc_macro"] = float(roc_auc_macro)
    except Exception as e:
        roc_auc_macro = None
        print(f"Note: Could not calculate ROC-AUC ({e})")

    # 4. Save Visual Artifacts and Reports
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Confusion Matrix
        cm_path = save_dir / f"{model_name}_{split_name.lower()}_confusion_matrix.png"
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=cm_path,
            title=f"{split_name} Confusion Matrix ({model_name})",
        )
        print(f"[*] Confusion matrix saved to: {cm_path}")

        # Save ROC Curves if probabilities valid
        if roc_auc_macro is not None:
            roc_path = save_dir / f"{model_name}_{split_name.lower()}_roc_curves.png"
            plot_multiclass_roc(
                y_true_onehot=y_true_onehot,
                y_probs=y_probs,
                class_names=class_names,
                save_path=roc_path,
                title=f"ROC Curves ({split_name} - {model_name})",
            )
            print(f"[*] ROC curves saved to: {roc_path}")

        # Save JSON Metrics Report
        report_json_path = save_dir / f"{model_name}_{split_name.lower()}_metrics.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary_metrics": metrics,
                    "per_class_report": report_dict,
                },
                f,
                indent=4,
            )
        print(f"[*] Metrics report saved to: {report_json_path}")

    print("=" * 80 + "\n")
    return metrics


def plot_multiclass_roc(
    y_true_onehot: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "One-vs-Rest ROC Curves",
) -> None:
    """Plot and save multiclass One-vs-Rest ROC curves."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        class_auc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {class_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# SINGLE IMAGE INFERENCE
# ==============================================================================
@torch.no_grad()
def predict_single_image(
    image_path: str,
    model: nn.Module,
    class_names: List[str],
    device: torch.device,
    top_k: int = 3,
    save_annotated_path: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Run diagnostic classification inference on a single cervical cytology image.
    Returns:
        List of (class_name, confidence_percentage) sorted by highest confidence.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at: {img_path}")

    _, eval_transform = get_transforms(image_size=IMAGE_SIZE)

    try:
        with Image.open(img_path) as raw_img:
            img_rgb = raw_img.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image '{img_path}': {e}")

    input_tensor = eval_transform(img_rgb).unsqueeze(0).to(device)

    model.eval()
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))

    results: List[Tuple[str, float]] = []
    print("\n" + "=" * 70)
    print("CERVICAL CYTOLOGY INFERENCE RESULT")
    print("=" * 70)
    print(f"Image File:  {img_path.name}")
    print(f"Full Path:   {img_path}")
    print("-" * 70)

    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), 1):
        c_name = class_names[idx]
        conf = prob * 100.0
        results.append((c_name, conf))
        marker = "==> TOP PREDICTION:" if rank == 1 else f"    Rank {rank}:"
        print(f"{marker} {c_name.ljust(30)} {conf:6.2f}% confidence")

    print("=" * 70 + "\n")

    if save_annotated_path is not None:
        save_out = Path(save_annotated_path)
        save_out.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_rgb)
        ax.axis("off")
        top_name, top_conf = results[0]
        ax.set_title(
            f"Prediction: {top_name}\nConfidence: {top_conf:.2f}%",
            fontsize=12,
            fontweight="bold",
            color="navy",
        )
        plt.tight_layout()
        plt.savefig(save_out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[*] Saved annotated preview to: {save_out}")

    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained cervical cancer model or run inference.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SIPaKMeD",
        choices=list(DATASETS.keys()),
        help="Target dataset (default: SIPaKMeD)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"],
        help="Model backbone (default: resnet18)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file (defaults to models/{dataset}/{model}_best.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average original, horizontal-flip, and vertical-flip predictions",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for diagnostic inference",
    )
    parser.add_argument(
        "--save_annotated",
        type=str,
        default=None,
        help="Optional path to save annotated prediction image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoint path
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(MODELS_DIR) / args.dataset / f"{args.model}_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {ckpt_path}\n"
            f"Please run 'src/train.py --dataset {args.dataset} --model {args.model}' first to train the model."
        )

    # 1. Load Checkpoint Metadata
    print(f"[*] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    class_names = state.get("class_names")
    num_classes = state.get("num_classes", len(class_names) if class_names else 5)
    model_name = state.get("model_name", args.model)

    # 2. Build and Load Model
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    load_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)
    model.eval()

    # 3. Single Image Inference Mode
    if args.image is not None:
        predict_single_image(
            image_path=args.image,
            model=model,
            class_names=class_names,
            device=device,
            save_annotated_path=args.save_annotated,
        )
        return

    # 4. Full Dataset Split Evaluation Mode
    print(f"[*] Preparing '{args.split}' DataLoader for {args.dataset}...")
    train_loader, val_loader, test_loader, resolved_classes, _ = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    split_map = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    target_loader = split_map[args.split.lower()]

    results_dir = Path(RESULTS_DIR) / args.dataset
    evaluate_dataset(
        model=model,
        loader=target_loader,
        class_names=class_names if class_names else resolved_classes,
        device=device,
        save_dir=results_dir,
        split_name=args.split.capitalize(),
        model_name=f"{model_name}_tta" if args.tta else model_name,
        use_tta=args.tta,
    )


if __name__ == "__main__":
    main()
