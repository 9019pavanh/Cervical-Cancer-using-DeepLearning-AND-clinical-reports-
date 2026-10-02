"""Evaluate ResNet-18, EfficientNet-B0, and their probability ensemble."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

try:
    from dataset_config import MODELS_DIR, RESULTS_DIR
    from dataset_loader import get_dataloaders
    from train import build_model
    from utils import load_checkpoint
except ImportError:
    from src.dataset_config import MODELS_DIR, RESULTS_DIR
    from src.dataset_loader import get_dataloaders
    from src.train import build_model
    from src.utils import load_checkpoint


@torch.no_grad()
def collect_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    for images, labels in loader:
        images = images.to(device)
        views = [images, torch.flip(images, [3]), torch.flip(images, [2])]
        view_probs = [torch.softmax(model(view), dim=1) for view in views]
        probabilities.append(torch.stack(view_probs).mean(0).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "macro_roc_auc": float(
            roc_auc_score(targets, probabilities, multi_class="ovr", average="macro")
        ),
    }


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    checkpoint = MODELS_DIR / "SIPaKMeD" / f"{name}_best.pth"
    model = build_model(name, num_classes=5, pretrained=False)
    load_checkpoint(checkpoint, model, device=device)
    return model.to(device).eval()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation_loader, test_loader, _, _ = get_dataloaders(
        "SIPaKMeD", batch_size=128, num_workers=0
    )
    resnet = load_model("resnet18", device)
    efficientnet = load_model("efficientnet_b0", device)

    res_val, val_targets = collect_probabilities(resnet, validation_loader, device)
    eff_val, _ = collect_probabilities(efficientnet, validation_loader, device)

    candidates = []
    # Require both independently trained backbones to contribute to the hybrid.
    for resnet_weight in np.arange(0.05, 0.96, 0.05):
        fused = resnet_weight * res_val + (1.0 - resnet_weight) * eff_val
        score = f1_score(val_targets, fused.argmax(1), average="macro")
        candidates.append((float(score), float(resnet_weight)))
    _, best_resnet_weight = max(candidates)

    res_test, test_targets = collect_probabilities(resnet, test_loader, device)
    eff_test, _ = collect_probabilities(efficientnet, test_loader, device)
    hybrid_test = best_resnet_weight * res_test + (1.0 - best_resnet_weight) * eff_test

    report = {
        "selection_split": "validation",
        "evaluation_split": "grouped hold-out test",
        "tta_views": 3,
        "resnet18": metrics(res_test, test_targets),
        "efficientnet_b0": metrics(eff_test, test_targets),
        "hybrid": metrics(hybrid_test, test_targets),
        "hybrid_weights": {
            "resnet18": best_resnet_weight,
            "efficientnet_b0": 1.0 - best_resnet_weight,
        },
    }
    output = RESULTS_DIR / "SIPaKMeD" / "hybrid_comparison_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
