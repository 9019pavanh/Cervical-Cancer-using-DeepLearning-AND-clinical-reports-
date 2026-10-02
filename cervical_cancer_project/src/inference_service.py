"""
inference_service.py
====================
Unified Inference Engine for Cervical Cancer Diagnostics.

Provides:
1. Trained ResNet-18 cytology inference with test-time augmentation.
2. Explainable AI (Grad-CAM):
   - Heatmap generation
   - Blended overlay with diagnostic focus
   - Base64 encoding for instant browser rendering
3. Cancer Staging Engine:
   - The Bethesda System 2014 category (NILM / LSIL / HSIL)
   - CIN grade (CIN 1 / CIN 2-3 / Normal)
   - Stage 0 / Pre-invasive classification
4. Clinical Risk Predictor:
   - Tabular risk prediction on patient parameters
   - Risk drivers & ASCCP clinical guidelines
5. Multimodal synthesis report.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from cancer_staging import STAGING_DATABASE, get_staging_info
    from dataset_config import MODELS_DIR, PROJECT_ROOT
    from hybrid_model import MODEL_BENCHMARKS
    from risk_predictor import ClinicalRiskPredictor
    from train import build_model
    from utils import load_checkpoint
except ImportError:
    from src.cancer_staging import STAGING_DATABASE, get_staging_info
    from src.dataset_config import MODELS_DIR, PROJECT_ROOT
    from src.hybrid_model import MODEL_BENCHMARKS
    from src.risk_predictor import ClinicalRiskPredictor
    from src.train import build_model
    from src.utils import load_checkpoint


# Standard class ordering matching SIPaKMeD dataset training
CLASS_NAMES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DiagnosticService:
    """Singleton service maintaining initialized models for rapid inference."""

    def __init__(self, device_str: Optional[str] = None):
        self.device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[DiagnosticService] Initializing on device: {self.device}")

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # 1. Load ResNet-18 Trained Model
        self.resnet_model = None
        self.resnet_ckpt = None
        self._load_resnet18()

        self.efficientnet_model = None
        self._load_efficientnet_b0()
        self.hybrid_model = self.efficientnet_model

        # 3. Load Clinical Risk Predictor
        self.clinical_predictor = ClinicalRiskPredictor()
        print("[DiagnosticService] All models and services successfully loaded.")

    def _load_resnet18(self):
        ckpt_path = MODELS_DIR / "SIPaKMeD" / "resnet18_best.pth"
        if not ckpt_path.exists():
            print(f"[DiagnosticService] Warning: ResNet checkpoint not found at {ckpt_path}")
            return

        model = build_model("resnet18", num_classes=5, pretrained=False)
        self.resnet_ckpt = load_checkpoint(ckpt_path, model, device=self.device)
        model.to(self.device)
        model.eval()
        self.resnet_model = model
        print(f"[DiagnosticService] ResNet-18 loaded from {ckpt_path.name}")

    def _load_efficientnet_b0(self):
        ckpt_path = MODELS_DIR / "SIPaKMeD" / "efficientnet_b0_best.pth"
        if not ckpt_path.exists():
            print(f"[DiagnosticService] EfficientNet checkpoint not found at {ckpt_path}")
            return

        model = build_model("efficientnet_b0", num_classes=5, pretrained=False)
        load_checkpoint(ckpt_path, model, device=self.device)
        model.to(self.device)
        model.eval()
        self.efficientnet_model = model
        print(f"[DiagnosticService] EfficientNet-B0 loaded from {ckpt_path.name}")

    def _tta_probabilities(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
        views = [
            input_tensor,
            torch.flip(input_tensor, dims=[3]),
            torch.flip(input_tensor, dims=[2]),
        ]
        with torch.no_grad():
            logits = torch.stack([model(view) for view in views])
            return torch.softmax(logits, dim=2).mean(dim=0)[0].cpu().numpy()

    def _compute_gradcam(
        self,
        image_pil: Image.Image,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates Grad-CAM heatmap and blended overlay numpy arrays."""
        if self.resnet_model is None:
            # Fallback blank heatmap
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            return blank, blank

        # Target layer for ResNet-18 is model.layer4[-1]
        target_layer = self.resnet_model.layer4[-1]

        activations = []
        gradients = []

        def fwd_hook(mod, inp, out):
            activations.append(out.detach())

        def bwd_hook(mod, gin, gout):
            gradients.append(gout[0].detach())

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        self.resnet_model.zero_grad()
        out = self.resnet_model(input_tensor)
        score = out[0, target_class]
        score.backward()

        h1.remove()
        h2.remove()

        act = activations[0]
        grad = gradients[0]
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            norm_cam = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            norm_cam = np.zeros_like(cam_np)

        # Generate colored heatmap
        heatmap_uint8 = np.uint8(255 * norm_cam)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Overlay with original image resized to 224x224
        img_resized = np.array(image_pil.resize((224, 224)).convert("RGB"))
        overlay = np.uint8(0.55 * img_resized + 0.45 * heatmap_color)

        return heatmap_color, overlay

    @staticmethod
    def _array_to_base64_data_url(arr: np.ndarray, format: str = "PNG") -> str:
        """Converts RGB numpy array to base64 Data URL string."""
        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    @staticmethod
    def _pil_to_base64_data_url(img: Image.Image, format: str = "PNG") -> str:
        """Converts PIL Image to base64 Data URL string."""
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    def analyze_cytology_image(
        self,
        image_input: Any,  # PIL.Image or Path or BytesIO
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-model cytology analysis on an uploaded image.
        Returns:
        - Primary predicted class, confidence, and all class probabilities
        - Cancer stage and Bethesda System report
        - Metrics and prediction from the trained ResNet-18 checkpoint
        - Grad-CAM heatmap & overlay as base64 images
        """
        if isinstance(image_input, (str, Path)):
            image_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, io.BytesIO):
            image_pil = Image.open(image_input).convert("RGB")
        else:
            image_pil = image_input.convert("RGB")

        orig_w, orig_h = image_pil.size
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # -------------------------------------------------------------
        # 1. ResNet-18 Inference
        # -------------------------------------------------------------
        if self.resnet_model is None:
            raise RuntimeError("Trained ResNet-18 checkpoint is unavailable")

        # Test-time augmentation improves robustness without introducing an
        # untrained ensemble: cervical cell orientation is non-semantic.
        res_probs = self._tta_probabilities(self.resnet_model, input_tensor)

        res_pred_idx = int(np.argmax(res_probs))
        res_conf = float(res_probs[res_pred_idx] * 100.0)

        # -------------------------------------------------------------
        eff_probs = None
        if self.efficientnet_model is not None:
            eff_probs = self._tta_probabilities(self.efficientnet_model, input_tensor)
            hybrid_probs = 0.95 * res_probs + 0.05 * eff_probs
        else:
            hybrid_probs = res_probs

        primary_idx = int(np.argmax(hybrid_probs))
        primary_class = CLASS_NAMES[primary_idx]
        primary_conf = float(hybrid_probs[primary_idx] * 100.0)

        # Probabilities breakdown for primary model
        probabilities_breakdown = [
            {
                "class_name": cls,
                "label": STAGING_DATABASE[cls]["short_name"],
                "probability": round(float(hybrid_probs[i] * 100.0), 2),
                "is_top": (i == primary_idx),
            }
            for i, cls in enumerate(CLASS_NAMES)
        ]
        probabilities_breakdown.sort(key=lambda x: x["probability"], reverse=True)

        # -------------------------------------------------------------
        # 3. Cancer Staging Information
        # -------------------------------------------------------------
        staging_info = get_staging_info(primary_class)

        # -------------------------------------------------------------
        # 4. Multi-Model Accuracy Comparison Table
        # -------------------------------------------------------------
        model_comparison = [
            {
                "model_id": "resnet18",
                "model_name": MODEL_BENCHMARKS["resnet18"]["model_name"],
                "short_name": MODEL_BENCHMARKS["resnet18"]["short_name"],
                "architecture_type": MODEL_BENCHMARKS["resnet18"]["architecture_type"],
                "parameters": MODEL_BENCHMARKS["resnet18"]["parameters"],
                "test_accuracy": MODEL_BENCHMARKS["resnet18"]["test_accuracy"],
                "macro_f1": MODEL_BENCHMARKS["resnet18"]["macro_f1"],
                "roc_auc": MODEL_BENCHMARKS["resnet18"]["roc_auc"],
                "predicted_class": STAGING_DATABASE[CLASS_NAMES[res_pred_idx]]["short_name"],
                "confidence": round(res_conf, 2),
                "is_primary": True,
                "badge": "Active Cytology Backbone",
            },
        ]

        if eff_probs is not None:
            eff_pred_idx = int(np.argmax(eff_probs))
            model_comparison.extend([
                {
                    "model_id": "efficientnet_b0",
                    "model_name": "EfficientNet-B0",
                    "predicted_class": STAGING_DATABASE[CLASS_NAMES[eff_pred_idx]]["short_name"],
                    "confidence": round(float(eff_probs[eff_pred_idx] * 100.0), 2),
                    "is_primary": False,
                    "badge": "Second Trained Backbone",
                },
                {
                    "model_id": "probability_hybrid",
                    "model_name": "ResNet-18 + EfficientNet-B0",
                    "predicted_class": STAGING_DATABASE[CLASS_NAMES[primary_idx]]["short_name"],
                    "confidence": round(primary_conf, 2),
                    "is_primary": True,
                    "badge": "Probability Fusion (95:5)",
                },
            ])

        # -------------------------------------------------------------
        # 5. Explainable AI: Grad-CAM
        # -------------------------------------------------------------
        heatmap_arr, overlay_arr = self._compute_gradcam(image_pil, input_tensor, primary_idx)

        original_b64 = self._pil_to_base64_data_url(image_pil)
        heatmap_b64 = self._array_to_base64_data_url(heatmap_arr)
        overlay_b64 = self._array_to_base64_data_url(overlay_arr)

        return {
            "prediction": {
                "class_name": primary_class,
                "class_label": staging_info["name"],
                "short_label": staging_info["short_name"],
                "confidence": round(primary_conf, 2),
            },
            "probabilities": probabilities_breakdown,
            "staging": staging_info,
            "model_comparison": model_comparison,
            "visualizations": {
                "original": original_b64,
                "heatmap": heatmap_b64,
                "overlay": overlay_b64,
                "dimensions": f"{orig_w}x{orig_h}",
            },
        }

    def assess_clinical_risk(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates patient clinical factors and returns risk score, drivers, and recommendations."""
        return self.clinical_predictor.predict(patient_dict)

    def generate_multimodal_synthesis(
        self,
        image_result: Dict[str, Any],
        clinical_result: Dict[str, Any],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Combines cytology staging and clinical risk for a comprehensive pathologist report."""
        staging = image_result.get("staging", {})
        clinical_level = clinical_result.get("risk_level", "Low Risk")
        clinical_pct = clinical_result.get("risk_percentage", 0.0)

        abnormal_classes = {"im_Dyskeratotic", "im_Koilocytotic"}
        image_abnormality = sum(
            item.get("probability", 0.0)
            for item in image_result.get("probabilities", [])
            if item.get("class_name") in abnormal_classes
        )
        image_weight = 0.70
        clinical_weight = 0.30
        hybrid_risk = round(
            image_weight * float(image_abnormality)
            + clinical_weight * float(clinical_pct),
            1,
        )

        # Composite Risk Triage Matrix
        # HSIL (Dyskeratotic) + High Clinical Risk -> Critical Tier 1
        # LSIL (Koilocytotic) + Mod/High Clinical Risk -> Urgent Tier 2
        # NILM / Metaplasia + Low Clinical Risk -> Routine Tier 4
        bethesda = staging.get("bethesda_category", "")

        if "HSIL" in bethesda:
            overall_tier = "Critical Tier 1 (Urgent Surgical / Colposcopy Triage)"
            tier_color = "#ff3b30"
            urgency = "Immediate Referral (< 2 Weeks)"
            summary = (
                f"High-grade intraepithelial lesion (HSIL / CIN 2-3) identified on cytology with "
                f"{clinical_pct}% clinical oncogenic risk. Direct histopathological confirmation via colposcopically directed punch biopsy with ECC indicated."
            )
        elif "LSIL" in bethesda:
            if clinical_level in ["High Risk", "Moderate Risk"]:
                overall_tier = "Elevated Tier 2 (Colposcopic Triage Indicated)"
                tier_color = "#ff9500"
                urgency = "Colposcopy within 4-6 Weeks"
                summary = (
                    f"Low-grade HPV lesion (LSIL / CIN 1) combined with elevated clinical risk factors ({clinical_pct}%). "
                    f"Triage directly to colposcopy rather than deferred surveillance due to clinical co-morbidities."
                )
            else:
                overall_tier = "Monitored Tier 3 (Conservative Surveillance)"
                tier_color = "#ffcc00"
                urgency = "Repeat Co-Testing in 12 Months"
                summary = (
                    f"Low-grade lesion (LSIL) with low clinical background risk ({clinical_pct}%). "
                    f"Per ASCCP guidelines, expectant observational management with 12-month follow-up co-testing is appropriate."
                )
        else:
            if clinical_level == "High Risk":
                overall_tier = "Discordant Tier 3 (High Clinical Risk despite Normal Cytology)"
                tier_color = "#ff9500"
                urgency = "hrHPV Co-Testing & Surveillance in 6 Months"
                summary = (
                    f"Cytology shows normal/benign cellular morphology (NILM/Metaplasia), but patient carries high clinical risk factors ({clinical_pct}%). "
                    f"Molecular hrHPV DNA co-testing recommended to rule out sampling omission."
                )
            else:
                overall_tier = "Routine Tier 4 (Negative / Standard Population Screening)"
                tier_color = "#34c759"
                urgency = "Routine Screening (3 to 5 Years)"
                summary = (
                    f"Benign/normal cytology with standard baseline clinical risk ({clinical_pct}%). "
                    f"Routine cervical cancer screening interval per national screening schedule."
                )

        combined_report = {
            "overall_tier": overall_tier,
            "tier_color": tier_color,
            "urgency": urgency,
            "clinical_summary": summary,
            "cytology_summary": {
                "cell_class": staging.get("name", "Unknown"),
                "bethesda_stage": staging.get("bethesda_category", "NILM"),
                "cin_grade": staging.get("cin_grade", "Normal"),
                "cancer_stage": staging.get("cancer_stage", "Stage 0"),
                "confidence": image_result.get("prediction", {}).get("confidence", 0.0),
            },
            "clinical_summary_details": {
                "risk_percentage": clinical_pct,
                "risk_level": clinical_level,
                "key_drivers": clinical_result.get("key_risk_drivers", []),
            },
            "hybrid_fusion": {
                "score": hybrid_risk,
                "image_abnormality_probability": round(float(image_abnormality), 1),
                "clinical_risk_probability": round(float(clinical_pct), 1),
                "image_weight": image_weight,
                "clinical_weight": clinical_weight,
                "method": "Weighted late decision fusion",
                "explanation": (
                    "Cytology receives the larger weight because it directly measures cell morphology; "
                    "clinical factors provide contextual risk adjustment."
                ),
            },
            "patient_info": patient_metadata or {},
            "action_items": clinical_result.get("clinical_suggestions", []),
        }

        return combined_report
