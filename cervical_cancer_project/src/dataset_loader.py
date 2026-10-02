"""
dataset_loader.py
=================
Phase 2 & Phase 3: Dataset Splitting, Preprocessing, and PyTorch DataLoaders.

Provides:
- Stratified Train / Validation / Test splitting (70% / 15% / 15%)
- CSV split manifest export for experimental reproducibility
- Data augmentation and normalization pipelines
- Custom PyTorch CervicalCancerDataset
- DataLoaders generator for Phase 4 model training
"""

import csv
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Import dataset configuration
try:
    from dataset_config import (
        DATASETS,
        IMAGE_EXTENSIONS,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        PROJECT_ROOT,
        RANDOM_SEED,
        TEST_RATIO,
        TRAIN_RATIO,
        VAL_RATIO,
    )
except ImportError:
    from src.dataset_config import (
        DATASETS,
        IMAGE_EXTENSIONS,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        PROJECT_ROOT,
        RANDOM_SEED,
        TEST_RATIO,
        TRAIN_RATIO,
        VAL_RATIO,
    )

SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"


# ==============================================================================
# DATA TRANSFORMS & AUGMENTATIONS (Phase 3)
# ==============================================================================
def get_transforms(
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns torchvision transforms for (1) training with augmentation,
    and (2) validation/testing with deterministic resizing & normalization.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.RandomAutocontrast(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


# ==============================================================================
# PYTORCH DATASET CLASS (Phase 3)
# ==============================================================================
class CervicalCancerDataset(Dataset):
    """
    PyTorch Dataset for Cervical Cytology Images.
    Loads images on-demand from file paths, converts to RGB, and applies transforms.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            samples: List of (file_path_str, class_idx) tuples.
            class_to_idx: Mapping from class name to integer index.
            transform: Optional torchvision transform to apply.
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image '{file_path}': {e}")

        if self.transform is not None:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = transforms.ToTensor()(img_rgb)

        return image_tensor, label


# ==============================================================================
# DATA DISCOVERY & STRATIFIED SPLITTING (Phase 2)
# ==============================================================================
def discover_images(
    dataset_name: str,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Discovers all valid image files and extracts their biological class labels.
    Returns:
        (samples, sorted_class_names) where samples = [(file_path, class_name), ...]
    """
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASETS.keys())}")

    cfg = DATASETS[dataset_name]
    root = cfg.active_path

    if not root.exists():
        raise FileNotFoundError(f"Active dataset directory does not exist: {root}")

    samples: List[Tuple[str, str]] = []

    for file in root.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel = file.relative_to(root)

        # Dataset-specific class resolution
        if dataset_name == "SIPaKMeD":
            # Top-level folder represents the cell type (e.g., im_Dyskeratotic)
            class_name = rel.parts[0]
        elif dataset_name == "CervicalCancer" and len(rel.parts) >= 3:
            class_name = f"{rel.parts[0]}_{file.parent.name}"
        elif len(rel.parts) >= 2:
            class_name = rel.parts[0]
        else:
            class_name = file.parent.name if file.parent != root else "ROOT"

        samples.append((str(file.resolve()), class_name))

    class_names = sorted(list({c for _, c in samples}))
    return samples, class_names


def stratified_split(
    samples: List[Tuple[str, str]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Splits samples into Train, Validation, and Test sets using stratified partitioning.
    Ensures identical class ratios across all three partitions.
    """
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-4, f"Ratios must sum to 1.0 (got {total})"

    # Keep an original image and all of its CROPPED derivatives in one split.
    # SIPaKMeD names related samples like 032.bmp, 032_01.bmp, 032_02.bmp.
    by_class = defaultdict(lambda: defaultdict(list))
    for path, cls in samples:
        stem = Path(path).stem
        group_id = re.sub(r"_\d+$", "", stem)
        by_class[cls][group_id].append((path, cls))

    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []

    for cls, grouped_items in sorted(by_class.items()):
        groups = list(grouped_items.values())
        rng.shuffle(groups)
        n = len(groups)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        # Ensure at least 1 sample in val and test if sufficient samples exist
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        n_test = n - n_train - n_val

        train_part = [sample for group in groups[:n_train] for sample in group]
        val_part = [sample for group in groups[n_train:n_train + n_val] for sample in group]
        test_part = [sample for group in groups[n_train + n_val:] for sample in group]

        train_samples.extend(train_part)
        val_samples.extend(val_part)
        test_samples.extend(test_part)

    # Shuffle combined sets
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def _has_group_overlap(*splits: List[Tuple[str, str]]) -> bool:
    """Return True when related image crops occur in more than one split."""
    group_sets = []
    for split in splits:
        groups = {
            (cls, re.sub(r"_\d+$", "", Path(path).stem))
            for path, cls in split
        }
        group_sets.append(groups)

    return any(
        group_sets[left] & group_sets[right]
        for left in range(len(group_sets))
        for right in range(left + 1, len(group_sets))
    )


def save_split_manifests(
    dataset_name: str,
    train_samples: List[Tuple[str, str]],
    val_samples: List[Tuple[str, str]],
    test_samples: List[Tuple[str, str]],
) -> Path:
    """Exports train, val, and test manifests to CSV files for reproducibility."""
    out_dir = SPLITS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.csv": train_samples,
        "val.csv": val_samples,
        "test.csv": test_samples,
    }

    for filename, data in splits.items():
        csv_path = out_dir / filename
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "class_name"])
            writer.writerows(data)

    return out_dir


# ==============================================================================
# MAIN DATALOADER FACTORY (Phase 3)
# ==============================================================================
def get_dataloaders(
    dataset_name: str = "SIPaKMeD",
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
    force_resplit: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    High-level factory function that returns ready-to-train PyTorch DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, class_names, class_to_idx)
    """
    manifest_dir = SPLITS_DIR / dataset_name
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"

    # Use cached splits if available, otherwise perform fresh stratified split
    if (
        not force_resplit
        and train_csv.exists()
        and val_csv.exists()
        and test_csv.exists()
    ):
        print(f"[*] Loading existing stratified split manifests from: {manifest_dir}")

        def load_manifest(path: Path) -> List[Tuple[str, str]]:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                return [(row[0], row[1]) for row in reader]

        train_raw = load_manifest(train_csv)
        val_raw = load_manifest(val_csv)
        test_raw = load_manifest(test_csv)

        if dataset_name == "SIPaKMeD" and _has_group_overlap(train_raw, val_raw, test_raw):
            print("[!] Existing manifests leak related SIPaKMeD crops; rebuilding grouped splits.")
            samples, class_names = discover_images(dataset_name)
            train_raw, val_raw, test_raw = stratified_split(samples, seed=seed)
            save_split_manifests(dataset_name, train_raw, val_raw, test_raw)
        else:
            class_names = sorted(list({c for _, c in train_raw + val_raw + test_raw}))
    else:
        print(f"[*] Discovering images and computing stratified split for '{dataset_name}'...")
        samples, class_names = discover_images(dataset_name)
        train_raw, val_raw, test_raw = stratified_split(
            samples,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=seed,
        )
        saved_dir = save_split_manifests(dataset_name, train_raw, val_raw, test_raw)
        print(f"[+] Split manifests successfully saved to: {saved_dir}")

    # Build class index mappings
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    # Convert class names to integer labels
    train_samples = [(p, class_to_idx[c]) for p, c in train_raw]
    val_samples = [(p, class_to_idx[c]) for p, c in val_raw]
    test_samples = [(p, class_to_idx[c]) for p, c in test_raw]

    # Get data transformations
    train_transform, eval_transform = get_transforms(image_size=image_size)

    # Instantiate PyTorch Datasets
    train_dataset = CervicalCancerDataset(
        train_samples, class_to_idx=class_to_idx, transform=train_transform
    )
    val_dataset = CervicalCancerDataset(
        val_samples, class_to_idx=class_to_idx, transform=eval_transform
    )
    test_dataset = CervicalCancerDataset(
        test_samples, class_to_idx=class_to_idx, transform=eval_transform
    )

    # Instantiate PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_names, class_to_idx


# ==============================================================================
# VERIFICATION & DISPLAY SCRIPT
# ==============================================================================
def print_split_summary(
    dataset_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
) -> None:
    """Prints a structured summary of dataset splits and sample batches."""
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    test_len = len(test_loader.dataset)
    total_len = train_len + val_len + test_len

    print("=" * 80)
    print(f"CERVICAL CANCER PROJECT - DATALOADER INSPECTION: {dataset_name}")
    print("=" * 80)
    print(f"Total Samples:      {total_len:,}")
    print(f"Training Set (70%): {train_len:,} samples ({train_len / total_len * 100:.1f}%) | {len(train_loader)} batches")
    print(f"Validation Set (15%): {val_len:,} samples ({val_len / total_len * 100:.1f}%) | {len(val_loader)} batches")
    print(f"Test Set (15%):     {test_len:,} samples ({test_len / total_len * 100:.1f}%) | {len(test_loader)} batches")
    print("-" * 80)
    print(f"Classes ({len(class_names)}):")
    for idx, name in enumerate(class_names):
        print(f"  [{idx}] {name}")
    print("-" * 80)

    # Test loading a batch from train_loader
    print("[*] Testing batch retrieval from Train DataLoader...")
    images, labels = next(iter(train_loader))
    print(f"  Images Tensor Shape: {tuple(images.shape)} -> [Batch, Channels, Height, Width]")
    print(f"  Labels Tensor Shape: {tuple(labels.shape)}")
    print(f"  Tensor Value Range:  Min={images.min().item():.3f}, Max={images.max().item():.3f}")
    print(f"  Tensor Data Type:    {images.dtype}")

    # Count classes in current batch
    batch_counts = Counter(labels.tolist())
    print("  Class distribution in test batch:")
    for lbl_idx, count in sorted(batch_counts.items()):
        print(f"    {class_names[lbl_idx]}: {count} samples")

    print("\n[OK] Phase 2 Splitting & Phase 3 Preprocessing DataLoaders successfully verified!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test and build cervical cancer DataLoaders.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SIPaKMeD",
        choices=list(DATASETS.keys()),
        help="Dataset name to split and load (default: SIPaKMeD)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()

    train_ld, val_ld, test_ld, classes, cls_map = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
    )

    print_split_summary(args.dataset, train_ld, val_ld, test_ld, classes)
