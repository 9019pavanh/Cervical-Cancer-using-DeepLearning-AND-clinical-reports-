"""
Dataset Configuration for Cervical Cancer Detection Project.

Defines directory paths, class mappings, image parameters, split ratios,
and resolution helpers for all 5 project datasets:
- Mendeley_LBC
- Cytolog
- SIPaKMeD
- CervicalCancer
- Custom_Cervical_Cancer_Cytology
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ==============================================================================
# PROJECT BASE DIRECTORIES
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
GRADCAM_DIR = PROJECT_ROOT / "gradcam"
APP_DIR = PROJECT_ROOT / "app"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Downloads directory fallback path where raw datasets are initially stored
DOWNLOADS_DIR = Path(os.path.expanduser("~")) / "Downloads"

# Supported image file extensions
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


# ==============================================================================
# DATASET CONFIGURATION SCHEMA
# ==============================================================================
@dataclass
class DatasetConfig:
    name: str
    folder_name: str
    project_path: Path
    raw_source_path: Path
    classes: List[str] = field(default_factory=list)
    description: str = ""
    subsets: List[str] = field(default_factory=list)

    @property
    def active_path(self) -> Path:
        """
        Returns project_path if it exists and contains data,
        otherwise falls back to raw_source_path if available.
        """
        if self.project_path.exists():
            # Check if directory has files or subdirectories
            try:
                if any(self.project_path.iterdir()):
                    return self.project_path
            except PermissionError:
                pass

        if self.raw_source_path.exists():
            return self.raw_source_path

        return self.project_path

    def exists(self) -> bool:
        """Returns True if either project_path or raw_source_path exists."""
        return self.active_path.exists()


# ==============================================================================
# CONFIGURED DATASETS
# ==============================================================================
DATASETS: Dict[str, DatasetConfig] = {
    "Mendeley_LBC": DatasetConfig(
        name="Mendeley_LBC",
        folder_name="Mendeley_LBC",
        project_path=DATASET_ROOT / "Mendeley_LBC",
        raw_source_path=DOWNLOADS_DIR / "Mendeley LBC Cervical Cancer",
        classes=[
            "High squamous intra-epithelial lesion",
            "Low squamous intra-epithelial lesion",
            "Negative for Intraepithelial malignancy",
            "Squamous cell carcinoma",
        ],
        description="Mendeley Liquid Based Cytology (LBC) 4-class cervical cytology dataset.",
    ),
    "Cytolog": DatasetConfig(
        name="Cytolog",
        folder_name="Cytolog",
        project_path=DATASET_ROOT / "Cytolog",
        raw_source_path=DOWNLOADS_DIR / "Cytolog Cervical Cancer" / "Image",
        classes=[
            "ASCUS",
            "HSIL",
            "LSIL",
            "NILM",
            "SCC",
        ],
        description="Cytolog Cervical Cancer 5-class cytology image dataset.",
    ),
    "SIPaKMeD": DatasetConfig(
        name="SIPaKMeD",
        folder_name="SIPaKMeD",
        project_path=DATASET_ROOT / "SIPaKMeD",
        raw_source_path=DOWNLOADS_DIR / "SIPaKMeD",
        classes=[
            "im_Dyskeratotic",
            "im_Koilocytotic",
            "im_Metaplastic",
            "im_Parabasal",
            "im_Superficial-Intermediate",
        ],
        description="SIPaKMeD 5-class pap smear / cervical cell cluster dataset.",
    ),
    "CervicalCancer": DatasetConfig(
        name="CervicalCancer",
        folder_name="CervicalCancer",
        project_path=DATASET_ROOT / "CervicalCancer",
        raw_source_path=DOWNLOADS_DIR / "CervicalCancer",
        subsets=["Herlev", "Mendeley", "sipakmed"],
        classes=[
            # Herlev subset classes
            "carcinoma_in_situ",
            "light_dysplastic",
            "moderate_dysplastic",
            "normal_columnar",
            "normal_intermediate",
            "normal_superficiel",
            "severe_dysplastic",
            # Standard LBC classes
            "HSIL",
            "LSIL",
            "NL",
            "SCC",
        ],
        description="Composite multi-source Cervical Cancer benchmark containing Herlev, Mendeley, and SIPaKMeD subsets.",
    ),
    "Custom_Cervical_Cancer_Cytology": DatasetConfig(
        name="Custom_Cervical_Cancer_Cytology",
        folder_name="Custom_Cervical_Cancer_Cytology",
        project_path=DATASET_ROOT / "Custom_Cervical_Cancer_Cytology",
        raw_source_path=(
            DOWNLOADS_DIR
            / "Custom Cervical Cancer Cytology Image"
            / "Renamed_Custom Cytology Dataset"
        ),
        classes=[
            "Abnormal_Renamed",
            "Normal_Renamed",
        ],
        description="Custom binary (Normal vs. Abnormal) cervical cancer cytology dataset.",
    ),
}


# ==============================================================================
# MODEL & TRAINING HYPERPARAMETERS
# ==============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 2
RANDOM_SEED = 42

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet normalization standards
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================
def get_dataset_config(name: str) -> DatasetConfig:
    """Retrieve configuration for a specific dataset by key."""
    if name not in DATASETS:
        valid_keys = ", ".join(DATASETS.keys())
        raise KeyError(f"Dataset '{name}' not found. Available datasets: {valid_keys}")
    return DATASETS[name]


def get_active_path(name: str) -> Path:
    """Return the currently resolved active path (project or fallback) for a dataset."""
    return get_dataset_config(name).active_path


def list_datasets() -> List[str]:
    """List all configured dataset identifiers."""
    return list(DATASETS.keys())


def print_dataset_status() -> None:
    """Prints overview of all datasets, their active paths, and availability status."""
    print("=" * 80)
    print("CERVICAL CANCER PROJECT - DATASET CONFIGURATION STATUS")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Dir:  {DATASET_ROOT}")
    print("-" * 80)

    for key, cfg in DATASETS.items():
        status = "[READY]" if cfg.exists() else "[NOT FOUND]"
        print(f"\n[{key}] {status}")
        print(f"  Description:  {cfg.description}")
        print(f"  Project Path: {cfg.project_path}")
        print(f"  Raw Fallback: {cfg.raw_source_path}")
        print(f"  Active Path:  {cfg.active_path}")
        print(f"  Classes ({len(cfg.classes)}): {', '.join(cfg.classes[:6])}")
        if len(cfg.classes) > 6:
            print(f"               ... and {len(cfg.classes) - 6} more")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_dataset_status()
