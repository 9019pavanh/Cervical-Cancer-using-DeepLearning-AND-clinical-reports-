"""Grad-CAM for both CNNs and permutation importance for clinical risk."""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from config import CONFIG, ensure_output_directories
from dataset import prepare_clinical_data, split_clinical_indices
from models import ClinicalMLP, ImageClassifier
from preprocessing import build_image_transforms


class GradCAM:
    def __init__(self, model: ImageClassifier):
        self.model = model
        self.activations = None
        self.gradients = None
        if model.architecture == "resnet50":
            target_layer = model.backbone.layer4[-1]
        elif model.architecture == "efficientnet_b0":
            target_layer = model.backbone.features[-1]
        else:
            raise ValueError(model.architecture)
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, tensor: torch.Tensor, target_class: int = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        class_index = int(logits.argmax(1).item()) if target_class is None else target_class
        logits[0, class_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=CONFIG.data.image_size, mode="bilinear", align_corners=False)
        heatmap = cam[0, 0].detach().cpu().numpy()
        heatmap -= heatmap.min()
        heatmap /= max(float(heatmap.max()), 1e-8)
        return heatmap, class_index

    def close(self):
        self.handle.remove()


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    import cv2

    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def explain_image(
    image_path: Path,
    architecture: str,
    output_path: Path,
) -> Dict[str, str]:
    checkpoint = torch.load(
        CONFIG.model_dir / f"{architecture}_best.pth", map_location="cpu", weights_only=False
    )
    model = ImageClassifier(architecture, len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, transform = build_image_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    explainer = GradCAM(model)
    heatmap, class_index = explainer.generate(tensor)
    explainer.close()
    original = np.asarray(image.resize(CONFIG.data.image_size))
    colored = colorize_heatmap(heatmap)
    overlay = np.uint8(0.60 * original + 0.40 * colored)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return {"prediction": checkpoint["class_names"][class_index], "output": str(output_path)}


def clinical_permutation_importance(repeats: int = 20) -> List[Dict[str, float]]:
    ensure_output_directories()
    checkpoint = torch.load(CONFIG.model_dir / "clinical_mlp_best.pth", map_location="cpu", weights_only=False)
    preprocessor = joblib.load(CONFIG.model_dir / "clinical_preprocessor.joblib")
    features, target = prepare_clinical_data()
    indices = split_clinical_indices(target)["test"]
    matrix = preprocessor.transform(features.iloc[indices]).astype("float32")
    targets = target.iloc[indices].to_numpy()
    names = preprocessor.get_feature_names_out().tolist()
    model = ClinicalMLP(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict(values):
        with torch.no_grad():
            _, logits = model(torch.from_numpy(values))
            return torch.sigmoid(logits).numpy()

    baseline = roc_auc_score(targets, predict(matrix))
    rng = np.random.default_rng(CONFIG.data.seed)
    importances = []
    for column, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            permuted = matrix.copy()
            rng.shuffle(permuted[:, column])
            drops.append(baseline - roc_auc_score(targets, predict(permuted)))
        importances.append({"feature": name, "importance": float(np.mean(drops))})
    importances.sort(key=lambda item: item["importance"], reverse=True)
    top = importances[:20]
    plt.figure(figsize=(9, 7))
    plt.barh([item["feature"] for item in reversed(top)], [item["importance"] for item in reversed(top)])
    plt.xlabel("Mean decrease in ROC-AUC")
    plt.tight_layout()
    plt.savefig(CONFIG.result_dir / "clinical_feature_importance.png", dpi=180)
    plt.close()
    return importances


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIG.supported_models)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--clinical", action="store_true")
    arguments = parser.parse_args()
    if arguments.clinical:
        print(json.dumps(clinical_permutation_importance()[:20], indent=2))
    else:
        output = CONFIG.result_dir / f"{arguments.model}_gradcam.png"
        print(json.dumps(explain_image(arguments.image, arguments.model, output), indent=2))
