"""Image, clinical, and guarded multimodal model definitions."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


class ImageClassifier(nn.Module):
    def __init__(self, architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.architecture = architecture
        if architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            network = models.resnet50(weights=weights)
            feature_dim = network.fc.in_features
            network.fc = nn.Identity()
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            network = models.efficientnet_b0(weights=weights)
            feature_dim = network.classifier[1].in_features
            network.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.backbone = network
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(images))


class ClinicalMLP(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(embedding_dim, 1)

    def forward(self, clinical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(clinical)
        return embedding, self.risk_head(embedding).squeeze(1)


class MultimodalHybrid(nn.Module):
    """Feature-level fusion; train only with genuinely linked image/clinical records."""

    def __init__(
        self,
        architecture: str,
        num_classes: int,
        clinical_dim: int,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.image_branch = ImageClassifier(architecture, num_classes, pretrained, dropout)
        self.clinical_branch = ClinicalMLP(clinical_dim, embedding_dim=128, dropout=dropout)
        fused_dim = self.image_branch.feature_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
        )
        self.disease_head = nn.Linear(128, num_classes)
        self.risk_head = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor, clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_embedding = self.image_branch.extract_features(images)
        clinical_embedding, _ = self.clinical_branch(clinical)
        fused = self.fusion(torch.cat([image_embedding, clinical_embedding], dim=1))
        return {
            "disease_logits": self.disease_head(fused),
            "risk_logits": self.risk_head(fused).squeeze(1),
        }


def build_image_model(architecture: str, num_classes: int, pretrained: bool = True) -> ImageClassifier:
    return ImageClassifier(architecture, num_classes, pretrained, CONFIGURED_DROPOUT)


CONFIGURED_DROPOUT = 0.30
