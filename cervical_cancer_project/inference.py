"""Checkpoint-based image and clinical inference for the FastAPI application."""

import base64
import io
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import CONFIG
from explainability import GradCAM, colorize_heatmap
from models import ClinicalMLP, ImageClassifier
from preprocessing import build_image_transforms, clean_clinical_frame


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ResearchInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, ImageClassifier] = {}
        self.class_names: Dict[str, List[str]] = {}
        self._load_image_models()
        self.clinical_model = None
        self.clinical_preprocessor = None
        self._load_clinical_model()

    def _load_image_models(self) -> None:
        for architecture in CONFIG.supported_models:
            checkpoint_path = CONFIG.model_dir / f"{architecture}_best.pth"
            if not checkpoint_path.exists():
                continue
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            classes = checkpoint["class_names"]
            model = ImageClassifier(architecture, len(classes), pretrained=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.models[architecture] = model.to(self.device).eval()
            self.class_names[architecture] = classes

    def _load_clinical_model(self) -> None:
        checkpoint_path = CONFIG.model_dir / "clinical_mlp_best.pth"
        preprocessor_path = CONFIG.model_dir / "clinical_preprocessor.joblib"
        if not checkpoint_path.exists() or not preprocessor_path.exists():
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model = ClinicalMLP(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        self.clinical_model = model.to(self.device).eval()
        self.clinical_preprocessor = joblib.load(preprocessor_path)

    def status(self) -> Dict:
        return {
            "available_image_models": sorted(self.models),
            "clinical_model_available": self.clinical_model is not None,
            "multimodal_fusion_validated": False,
            "limitation": (
                "No shared patient identifier links SIPaKMeD images to UCI clinical rows; "
                "cross-modal fusion is intentionally disabled."
            ),
        }

    def predict_image(self, image: Image.Image, architecture: str) -> Dict:
        if architecture not in self.models:
            raise FileNotFoundError(f"No trained research checkpoint for {architecture}")
        model = self.models[architecture]
        classes = self.class_names[architecture]
        _, transform = build_image_transforms()
        rgb = image.convert("RGB")
        tensor = transform(rgb).unsqueeze(0).to(self.device)
        views = [tensor, torch.flip(tensor, [3]), torch.flip(tensor, [2])]
        with torch.no_grad():
            probabilities = torch.stack([torch.softmax(model(view), 1) for view in views]).mean(0)[0]
        prediction = int(probabilities.argmax().item())
        explainer = GradCAM(model)
        heatmap, _ = explainer.generate(tensor, prediction)
        explainer.close()
        original = np.asarray(rgb.resize(CONFIG.data.image_size))
        colored = colorize_heatmap(heatmap)
        overlay = Image.fromarray(np.uint8(0.60 * original + 0.40 * colored))
        return {
            "model": architecture,
            "prediction": classes[prediction],
            "confidence": round(float(probabilities[prediction]) * 100, 2),
            "probabilities": [
                {"class_name": name, "probability": round(float(probabilities[index]) * 100, 2)}
                for index, name in enumerate(classes)
            ],
            "gradcam_overlay": image_to_data_url(overlay),
            "original": image_to_data_url(rgb),
        }

    def predict_clinical(self, values: Dict) -> Dict:
        if self.clinical_model is None:
            raise FileNotFoundError("Clinical MLP checkpoint has not been trained")
        expected_columns = list(self.clinical_preprocessor.feature_names_in_)
        provided_count = sum(
            column in values and values[column] is not None and pd.notna(values[column])
            for column in expected_columns
        )
        row = {column: values.get(column, np.nan) for column in expected_columns}
        frame = clean_clinical_frame(pd.DataFrame([row], columns=expected_columns))
        transformed = self.clinical_preprocessor.transform(frame).astype("float32")
        with torch.no_grad():
            _, logits = self.clinical_model(torch.from_numpy(transformed).to(self.device))
            probability = float(torch.sigmoid(logits).item())
        return {
            "early_risk_probability": round(probability * 100, 2),
            "provided_feature_count": provided_count,
            "imputed_feature_count": len(expected_columns) - provided_count,
            "research_only": True,
            "note": (
                "This separately trained clinical estimate is not fused with the image result. "
                "Missing inputs use values learned from the clinical training partition."
            ),
        }
