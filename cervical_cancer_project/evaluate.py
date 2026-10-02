"""Final unseen-test evaluation with medical classification metrics and plots."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from config import CONFIG, ensure_output_directories
from dataset import CervicalImageDataset, load_image_split, prepare_clinical_data, split_clinical_indices
from models import ClinicalMLP, ImageClassifier
from preprocessing import build_image_transforms


def bootstrap_accuracy_ci(targets: np.ndarray, predictions: np.ndarray, seed: int = 42) -> List[float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(2000):
        indices = rng.integers(0, len(targets), len(targets))
        scores.append(accuracy_score(targets[indices], predictions[indices]))
    return [float(value) for value in np.percentile(scores, [2.5, 97.5])]


def specificity_per_class(targets: np.ndarray, predictions: np.ndarray, classes: List[str]) -> Dict[str, float]:
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(classes)))
    total = matrix.sum()
    values = {}
    for index, name in enumerate(classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        values[name] = float(true_negative / max(true_negative + false_positive, 1))
    return values


def multiclass_brier(targets: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes)[targets]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


@torch.no_grad()
def image_predictions(model, loader, device, use_tta: bool = True):
    model.eval()
    targets, probabilities, paths = [], [], []
    for images, labels, batch_paths in loader:
        images = images.to(device)
        views = [images]
        if use_tta:
            views.extend([torch.flip(images, [3]), torch.flip(images, [2])])
        view_probabilities = [torch.softmax(model(view), 1) for view in views]
        probabilities.append(torch.stack(view_probabilities).mean(0).cpu().numpy())
        targets.append(labels.numpy())
        paths.extend(batch_paths)
    return np.concatenate(targets), np.concatenate(probabilities), paths


def save_image_plots(targets, probabilities, classes, prefix: Path) -> None:
    predictions = probabilities.argmax(1)
    matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_confusion_matrix.png"), dpi=180)
    plt.close()

    binary_targets = label_binarize(targets, classes=np.arange(len(classes)))
    plt.figure(figsize=(9, 7))
    for index, name in enumerate(classes):
        false_positive, true_positive, _ = roc_curve(binary_targets[:, index], probabilities[:, index])
        plt.plot(false_positive, true_positive, label=name)
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False-positive rate")
    plt.ylabel("Sensitivity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_roc_curves.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 7))
    for index, name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(binary_targets[:, index], probabilities[:, index])
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_pr_curves.png"), dpi=180)
    plt.close()


def evaluate_image(architecture: str, batch_size: int = 64) -> Dict:
    ensure_output_directories()
    checkpoint_path = CONFIG.model_dir / f"{architecture}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Train {architecture} first: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["class_names"]
    class_to_index = checkpoint["class_to_index"]
    _, evaluation_transform = build_image_transforms()
    dataset = CervicalImageDataset(load_image_split("test"), class_to_index, evaluation_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(architecture, len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    targets, probabilities, paths = image_predictions(model, loader, device)
    predictions = probabilities.argmax(1)
    binary_targets = label_binarize(targets, classes=np.arange(len(classes)))
    report = classification_report(targets, predictions, target_names=classes, output_dict=True, zero_division=0)
    results = {
        "model": architecture,
        "split": "unseen grouped test",
        "samples": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "accuracy_95_ci": bootstrap_accuracy_ci(targets, predictions),
        "precision_macro": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall_macro_sensitivity": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "specificity_per_class": specificity_per_class(targets, predictions, classes),
        "roc_auc_macro": float(roc_auc_score(binary_targets, probabilities, average="macro", multi_class="ovr")),
        "pr_auc_macro": float(average_precision_score(binary_targets, probabilities, average="macro")),
        "multiclass_brier_score": multiclass_brier(targets, probabilities, len(classes)),
        "per_class": {name: report[name] for name in classes},
    }
    prefix = CONFIG.result_dir / architecture
    save_image_plots(targets, probabilities, classes, prefix)
    prediction_frame = pd.DataFrame({"image_path": paths, "actual": [classes[i] for i in targets], "predicted": [classes[i] for i in predictions]})
    for index, name in enumerate(classes):
        prediction_frame[f"probability_{name}"] = probabilities[:, index]
    prediction_frame.to_csv(prefix.with_name(prefix.name + "_test_predictions.csv"), index=False)
    prefix.with_name(prefix.name + "_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def evaluate_clinical() -> Dict:
    ensure_output_directories()
    checkpoint = torch.load(CONFIG.model_dir / "clinical_mlp_best.pth", map_location="cpu", weights_only=False)
    preprocessor = joblib.load(CONFIG.model_dir / "clinical_preprocessor.joblib")
    features, target = prepare_clinical_data()
    indices = split_clinical_indices(target)["test"]
    inputs = preprocessor.transform(features.iloc[indices]).astype("float32")
    targets = target.iloc[indices].to_numpy()
    model = ClinicalMLP(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        _, logits = model(torch.from_numpy(inputs))
        probabilities = torch.sigmoid(logits).numpy()
    predictions = (probabilities >= 0.5).astype(int)
    results = {
        "split": "unseen stratified test",
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }
    pd.DataFrame({"actual": targets, "probability": probabilities, "predicted": predictions}).to_csv(
        CONFIG.result_dir / "clinical_test_predictions.csv", index=False
    )
    (CONFIG.result_dir / "clinical_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fraction_positive, mean_predicted = calibration_curve(targets, probabilities, n_bins=8)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted, fraction_positive, marker="o", label="Clinical MLP")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG.result_dir / "clinical_calibration.png", dpi=180)
    plt.close()
    return results


def create_model_comparison() -> List[Dict]:
    rows = []
    for architecture in CONFIG.supported_models:
        path = CONFIG.result_dir / f"{architecture}_evaluation.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": architecture,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["precision_macro"],
            "macro_sensitivity": metrics["recall_macro_sensitivity"],
            "macro_f1": metrics["f1_macro"],
            "macro_roc_auc": metrics["roc_auc_macro"],
            "macro_pr_auc": metrics["pr_auc_macro"],
            "brier_score": metrics["multiclass_brier_score"],
        })
    if not rows:
        raise FileNotFoundError("Evaluate ResNet50 and EfficientNet-B0 before creating comparison")
    pd.DataFrame(rows).to_csv(CONFIG.result_dir / "model_comparison.csv", index=False)
    (CONFIG.result_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["image", "clinical", "compare"], default="image")
    parser.add_argument("--model", choices=CONFIG.supported_models, default="resnet50")
    arguments = parser.parse_args()
    if arguments.task == "image":
        output = evaluate_image(arguments.model)
    elif arguments.task == "clinical":
        output = evaluate_clinical()
    else:
        output = create_model_comparison()
    print(json.dumps(output, indent=2))
