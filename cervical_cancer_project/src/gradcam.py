"""
gradcam.py
==========
Phase 6: Explainable AI (XAI) with Grad-CAM for Cervical Cancer Cytology.

Visualizes model decision attention maps on cytology microscopy images.
Identifies whether the CNN focuses on diagnostic morphological markers:
- Enlarged hyperchromatic nuclei
- High Nuclear-to-Cytoplasmic (N:C) ratio
- Irregular nuclear membranes
- Cytoplasmic halos (Koilocytes)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from dataset_config import (
        DATASETS,
        GRADCAM_DIR,
        IMAGE_SIZE,
        MODELS_DIR,
        PROJECT_ROOT,
    )
    from dataset_loader import get_transforms
    from train import build_model
    from utils import load_checkpoint
except ImportError:
    from src.dataset_config import (
        DATASETS,
        GRADCAM_DIR,
        IMAGE_SIZE,
        MODELS_DIR,
        PROJECT_ROOT,
    )
    from src.dataset_loader import get_transforms
    from src.train import build_model
    from src.utils import load_checkpoint


# ==============================================================================
# TARGET LAYER RESOLUTION
# ==============================================================================
def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Identifies the final convolutional feature layer for Grad-CAM.
    """
    name = model_name.lower().strip()
    if "resnet" in name:
        # Last Bottleneck/BasicBlock in layer4
        return model.layer4[-1]
    elif "efficientnet" in name:
        # Last feature block in features
        return model.features[-1]
    elif "mobilenet" in name:
        # Last feature block in features
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported architecture for Grad-CAM target layer: '{model_name}'")


# ==============================================================================
# GRAD-CAM IMPLEMENTATION
# ==============================================================================
class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mappings (Grad-CAM).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate normalized Grad-CAM heatmap for a single input tensor.

        Args:
            input_tensor: Shape (1, C, H, W)
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            (heatmap_2d_numpy, predicted_class_idx, confidence_percentage)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[target_class].item() * 100.0)

        # Backward pass on target class score
        score = logits[0, target_class]
        score.backward()

        # Check that hooks captured features and gradients
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # Apply ReLU to isolate positive contributions towards target class
        cam = F.relu(cam)

        # Upsample to input tensor size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Min-max normalization
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence

    def remove_hooks(self) -> None:
        """Clean up hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ==============================================================================
# VISUALIZATION & OVERLAYS
# ==============================================================================
def overlay_heatmap_on_image(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Blends Grad-CAM heatmap with original image.
    """
    img_resized = image_pil.resize(IMAGE_SIZE).convert("RGB")
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    cmap = plt.colormaps[colormap_name]
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Linear alpha blend
    overlay = (1.0 - alpha) * img_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def generate_gradcam_panel(
    image_path: Union[str, Path],
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    device: torch.device,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, str, float]:
    """
    Generates a 3-panel clinical visualization:
    [Raw Image | Standalone Thermal Heatmap | Blended Grad-CAM Overlay]
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    _, eval_transform = get_transforms(image_size=IMAGE_SIZE)

    with Image.open(image_path) as raw_img:
        img_rgb = raw_img.convert("RGB")

    input_tensor = eval_transform(img_rgb).unsqueeze(0).to(device)

    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    try:
        heatmap, pred_idx, confidence = grad_cam.generate(input_tensor)
    finally:
        grad_cam.remove_hooks()

    pred_class = class_names[pred_idx]
    overlay = overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.45)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Original Image
        ax1.imshow(img_rgb.resize(IMAGE_SIZE))
        ax1.set_title(f"Original Microscopy\n({image_path.name})", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # 2. Raw Grad-CAM Heatmap
        im_heat = ax2.imshow(heatmap, cmap="jet")
        ax2.set_title("Grad-CAM Thermal Heatmap\n(Activation Intensity)", fontsize=11, fontweight="bold")
        ax2.axis("off")
        fig.colorbar(im_heat, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Clinical Diagnostic Overlay
        ax3.imshow(overlay)
        ax3.set_title(
            f"Saliency Overlay\nPredicted: {pred_class} ({confidence:.2f}%)",
            fontsize=11,
            fontweight="bold",
            color="navy",
        )
        ax3.axis("off")

        plt.suptitle(
            f"Explainable AI: Cervical Cytology Attention Map ({model_name.upper()})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"[*] Saved Grad-CAM panel to: {save_path}")

    return overlay, pred_class, confidence


def run_batch_class_comparison(
    dataset_name: str,
    model: nn.Module,
    model_name: str,
    class_names: List[str],
    device: torch.device,
    output_dir: Path,
) -> None:
    """
    Selects 1 representative test sample from each class and generates a
    comparative 5-class Explainable AI diagnostic grid.
    """
    import csv

    manifest_csv = PROJECT_ROOT / "dataset" / "splits" / dataset_name / "test.csv"
    if not manifest_csv.exists():
        print(f"[!] Test split manifest not found at: {manifest_csv}")
        return

    # Find first test sample for each class
    samples_per_class: Dict[str, str] = {}
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for path_str, c_name in reader:
            if c_name not in samples_per_class and Path(path_str).exists():
                samples_per_class[c_name] = path_str
            if len(samples_per_class) == len(class_names):
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(samples_per_class)

    fig, axes = plt.subplots(num_classes, 3, figsize=(14, 3.8 * num_classes))

    print(f"\n[*] Generating multi-class Explainable AI comparison grid for {dataset_name}...")

    for row_idx, (true_class, img_path) in enumerate(sorted(samples_per_class.items())):
        with Image.open(img_path) as raw_img:
            img_rgb = raw_img.convert("RGB")

        _, eval_transform = get_transforms(image_size=IMAGE_SIZE)
        input_tensor = eval_transform(img_rgb).unsqueeze(0).to(device)

        target_layer = get_target_layer(model, model_name)
        grad_cam = GradCAM(model, target_layer)
        try:
            heatmap, pred_idx, confidence = grad_cam.generate(input_tensor)
        finally:
            grad_cam.remove_hooks()

        pred_class = class_names[pred_idx]
        overlay = overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.45)

        # Column 1: Original
        axes[row_idx, 0].imshow(img_rgb.resize(IMAGE_SIZE))
        axes[row_idx, 0].set_title(f"True: {true_class}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Column 2: Heatmap
        axes[row_idx, 1].imshow(heatmap, cmap="jet")
        axes[row_idx, 1].set_title("Grad-CAM Activation", fontsize=11)
        axes[row_idx, 1].axis("off")

        # Column 3: Overlay
        is_correct = pred_class == true_class
        title_color = "darkgreen" if is_correct else "red"
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(
            f"Pred: {pred_class} ({confidence:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=title_color,
        )
        axes[row_idx, 2].axis("off")

    plt.suptitle(
        f"SIPaKMeD Explainable AI (Grad-CAM) Multi-Class Attention Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    grid_save_path = output_dir / f"{dataset_name}_gradcam_multiclass_comparison.png"
    plt.savefig(grid_save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved multiclass Grad-CAM comparison grid to: {grid_save_path}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visual heatmaps for cervical cytology.")
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
        help="Model architecture (default: resnet18)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (defaults to models/{dataset}/{model}_best.pth)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for Grad-CAM generation",
    )
    parser.add_argument(
        "--batch_comparison",
        action="store_true",
        help="Generate multi-class comparison across test set samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(GRADCAM_DIR),
        help="Directory to save Grad-CAM visual artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(MODELS_DIR) / args.dataset / f"{args.model}_best.pth"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {ckpt_path}\n"
            f"Please ensure model is trained via 'src/train.py --dataset {args.dataset}' first."
        )

    print(f"[*] Loading model from checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    class_names = state.get("class_names", DATASETS[args.dataset].classes)
    num_classes = state.get("num_classes", len(class_names))
    model_name = state.get("model_name", args.model)

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    load_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)
    model.eval()

    out_dir = Path(args.output_dir)

    # 1. Single Image Mode
    if args.image is not None:
        save_file = out_dir / f"gradcam_{Path(args.image).stem}.png"
        generate_gradcam_panel(
            image_path=args.image,
            model=model,
            model_name=model_name,
            class_names=class_names,
            device=device,
            save_path=save_file,
        )

    # 2. Batch Comparison Mode (Default if no single image specified)
    if args.batch_comparison or args.image is None:
        run_batch_class_comparison(
            dataset_name=args.dataset,
            model=model,
            model_name=model_name,
            class_names=class_names,
            device=device,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
