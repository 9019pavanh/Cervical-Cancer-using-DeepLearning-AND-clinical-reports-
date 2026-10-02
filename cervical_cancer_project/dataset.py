"""Dataset audit, leakage-safe splitting, and PyTorch datasets."""

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import CONFIG, ensure_output_directories
from preprocessing import clean_clinical_frame


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: str
    group_id: str


@dataclass
class AuditReport:
    image_count: int
    classes: Dict[str, int]
    corrupted: List[str]
    duplicate_groups: List[List[str]]
    clinical_rows: int
    clinical_columns: List[str]
    clinical_target_distribution: Dict[str, int]
    image_patient_id_available: bool
    clinical_patient_id_available: bool
    modalities_linked: bool
    limitation: str


def infer_source_group(path: Path, label: str) -> str:
    """Group an original SIPaKMeD cell and all numbered crops together."""
    base = re.sub(r"_\d+$", "", path.stem)
    return f"{label}:{base}"


def discover_images(root: Optional[Path] = None) -> List[ImageRecord]:
    image_root = root or CONFIG.data.image_root
    if not image_root.exists():
        raise FileNotFoundError(f"Image dataset not found: {image_root}")
    records = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(image_root)
        label = relative.parts[0]
        records.append(ImageRecord(str(path.resolve()), label, infer_source_group(path, label)))
    if not records:
        raise RuntimeError(f"No supported images found under {image_root}")
    return records


def audit_images(records: Sequence[ImageRecord]) -> Tuple[List[ImageRecord], List[str], List[List[str]]]:
    valid = []
    corrupted = []
    hashes: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        path = Path(record.path)
        try:
            raw = path.read_bytes()
            with Image.open(path) as image:
                image.verify()
            hashes[hashlib.sha256(raw).hexdigest()].append(record)
            valid.append(record)
        except Exception:
            corrupted.append(record.path)
    duplicate_groups = [[record.path for record in group] for group in hashes.values() if len(group) > 1]
    duplicate_paths = {path for group in duplicate_groups for path in group[1:]}
    deduplicated = [record for record in valid if record.path not in duplicate_paths]
    return deduplicated, corrupted, duplicate_groups


def audit_project(write_report: bool = True) -> AuditReport:
    records = discover_images()
    clean_records, corrupted, duplicates = audit_images(records)
    clinical = pd.read_csv(CONFIG.data.clinical_csv)
    clinical_columns = clinical.columns.tolist()
    id_candidates = {"patient_id", "patientid", "patient id", "id"}
    clinical_has_id = any(column.lower() in id_candidates for column in clinical_columns)
    report = AuditReport(
        image_count=len(clean_records),
        classes=dict(sorted(Counter(record.label for record in clean_records).items())),
        corrupted=corrupted,
        duplicate_groups=duplicates,
        clinical_rows=len(clinical),
        clinical_columns=clinical_columns,
        clinical_target_distribution={
            str(key): int(value)
            for key, value in pd.to_numeric(clinical[CONFIG.risk_target], errors="coerce")
            .fillna(0)
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        },
        image_patient_id_available=False,
        clinical_patient_id_available=clinical_has_id,
        modalities_linked=False,
        limitation=(
            "SIPaKMeD images and UCI clinical rows have no shared patient identifier. "
            "Feature-level multimodal fusion cannot be scientifically validated."
        ),
    )
    if write_report:
        ensure_output_directories()
        output = CONFIG.result_dir / "dataset_audit.json"
        output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def grouped_stratified_split(
    records: Sequence[ImageRecord], seed: int = CONFIG.data.seed
) -> Dict[str, List[ImageRecord]]:
    by_class: Dict[str, Dict[str, List[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record.label][record.group_id].append(record)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for groups in by_class.values():
        values = list(groups.values())
        rng.shuffle(values)
        count = len(values)
        train_end = max(1, round(count * CONFIG.data.train_ratio))
        validation_count = max(1, round(count * CONFIG.data.validation_ratio))
        validation_end = min(count - 1, train_end + validation_count)
        partitions = {
            "train": values[:train_end],
            "validation": values[train_end:validation_end],
            "test": values[validation_end:],
        }
        for name, partition in partitions.items():
            splits[name].extend(record for group in partition for record in group)
    for split in splits.values():
        rng.shuffle(split)
    validate_group_isolation(splits)
    return splits


def validate_group_isolation(splits: Dict[str, Sequence[ImageRecord]]) -> None:
    group_sets = {name: {record.group_id for record in records} for name, records in splits.items()}
    names = list(group_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise RuntimeError(f"Data leakage: {len(overlap)} source groups in {left} and {right}")


def save_image_splits(splits: Dict[str, Sequence[ImageRecord]]) -> None:
    CONFIG.data.split_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        filename = "val.csv" if name == "validation" else f"{name}.csv"
        with (CONFIG.data.split_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file_path", "class_name", "group_id"])
            writer.writerows((record.path, record.label, record.group_id) for record in records)


def load_image_split(name: str) -> List[ImageRecord]:
    filename = "val.csv" if name in {"validation", "val"} else f"{name}.csv"
    path = CONFIG.data.split_dir / filename
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group_id = row.get("group_id") or infer_source_group(Path(row["file_path"]), row["class_name"])
            records.append(ImageRecord(row["file_path"], row["class_name"], group_id))
    return records


class CervicalImageDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], class_to_index: Dict[str, int], transform=None):
        self.records = list(records)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            rgb = image.convert("RGB")
        tensor = self.transform(rgb) if self.transform else rgb
        return tensor, self.class_to_index[record.label], record.path


def prepare_clinical_data() -> Tuple[pd.DataFrame, pd.Series]:
    frame = clean_clinical_frame(pd.read_csv(CONFIG.data.clinical_csv))
    target = pd.to_numeric(frame[CONFIG.risk_target], errors="coerce").fillna(0).astype(int)
    excluded = [column for column in CONFIG.excluded_clinical_columns if column in frame.columns]
    features = frame.drop(columns=excluded)
    return features, target


def split_clinical_indices(target: pd.Series) -> Dict[str, np.ndarray]:
    indices = np.arange(len(target))
    train_indices, temporary, train_y, temporary_y = train_test_split(
        indices,
        target.to_numpy(),
        test_size=CONFIG.data.validation_ratio + CONFIG.data.test_ratio,
        stratify=target,
        random_state=CONFIG.data.seed,
    )
    relative_test = CONFIG.data.test_ratio / (CONFIG.data.validation_ratio + CONFIG.data.test_ratio)
    validation_indices, test_indices = train_test_split(
        temporary,
        test_size=relative_test,
        stratify=temporary_y,
        random_state=CONFIG.data.seed,
    )
    return {"train": train_indices, "validation": validation_indices, "test": test_indices}


if __name__ == "__main__":
    report = audit_project()
    records, _, _ = audit_images(discover_images())
    save_image_splits(grouped_stratified_split(records))
    print(json.dumps(asdict(report), indent=2))
