"""Central configuration for reproducible cervical AI experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DataConfig:
    image_root: Path = Path.home() / "Downloads" / "SIPaKMeD"
    split_dir: Path = ROOT / "dataset" / "splits" / "SIPaKMeD"
    clinical_csv: Path = ROOT / "dataset" / "clinical_risk_factors" / "risk_factors_cervical_cancer.csv"
    image_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    label_smoothing: float = 0.10
    early_stopping_patience: int = 7
    num_workers: int = 0
    monitor: str = "macro_f1"


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_dir: Path = ROOT / "models" / "research"
    result_dir: Path = ROOT / "results" / "research"
    supported_models: Tuple[str, ...] = ("resnet50", "efficientnet_b0")
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    risk_target: str = "Biopsy"
    excluded_clinical_columns: Tuple[str, ...] = (
        "Biopsy",
        "Hinselmann",
        "Schiller",
        "Citology",
        "Dx:Cancer",
        "Dx:CIN",
        "Dx:HPV",
        "Dx",
    )


CONFIG = ProjectConfig()


def ensure_output_directories() -> Dict[str, Path]:
    CONFIG.model_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.result_dir.mkdir(parents=True, exist_ok=True)
    return {"models": CONFIG.model_dir, "results": CONFIG.result_dir}
