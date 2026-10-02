"""Training-only image augmentation and leakage-safe clinical preprocessing."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torchvision import transforms

from config import CONFIG


def build_image_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    size = CONFIG.data.image_size
    normalize = transforms.Normalize(CONFIG.imagenet_mean, CONFIG.imagenet_std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, evaluation_transform


def clean_clinical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.replace("?", np.nan).copy()
    for column in cleaned.columns:
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        if converted.notna().sum() >= cleaned[column].notna().sum() * 0.8:
            cleaned[column] = converted
    return cleaned


def split_clinical_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numerical = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numerical]
    return numerical, categorical


def build_clinical_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numerical, categorical = split_clinical_columns(frame)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])
