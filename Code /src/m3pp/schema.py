from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


TARGET_ALIASES = {
    "Yield Strength (MPa)": "Yield strength(MPa)",
    "Yield Strength": "Yield strength(MPa)",
    "屈服强度(MPa)": "Yield strength(MPa)",
    "Tensile Strength (MPa)": "Tensile strength(MPa)",
    "Tensile Strength": "Tensile strength(MPa)",
    "抗拉强度(MPa)": "Tensile strength(MPa)",
    "Elongation (%)": "Total elongation(%)",
    "Elongation": "Total elongation(%)",
    "延伸率(%)": "Total elongation(%)",
}

DEFAULT_TARGETS = [
    "Yield strength(MPa)",
    "Tensile strength(MPa)",
    "Total elongation(%)",
]


def normalize_name(value: object) -> str:
    return " ".join(str(value).strip().replace("℃", "degC").split())


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [normalize_name(column) for column in result.columns]
    return result.rename(columns=TARGET_ALIASES)


def load_feature_manifest(path: Path) -> Dict[str, List[str]]:
    manifest = pd.read_csv(path)
    required = {"feature_name", "modality"}
    if not required.issubset(manifest.columns):
        raise ValueError("Feature manifest requires feature_name and modality columns.")

    manifest = manifest.copy()
    manifest["feature_name"] = manifest["feature_name"].map(normalize_name)
    manifest["modality"] = manifest["modality"].astype(str).str.lower().str.strip()

    groups: Dict[str, List[str]] = {}
    for modality in ("composition", "processing", "microstructure"):
        subset = manifest.loc[manifest["modality"] == modality]
        if "order" in subset.columns:
            subset = subset.sort_values("order")
        groups[modality] = subset["feature_name"].tolist()

    expected = {"composition": 38, "processing": 24, "microstructure": 6}
    counts = {name: len(columns) for name, columns in groups.items()}
    if counts != expected:
        raise ValueError(f"Expected feature counts {expected}; received {counts}.")

    all_features = sum(groups.values(), [])
    if len(all_features) != len(set(all_features)):
        raise ValueError("A predictor appears in more than one modality.")
    return groups


def build_feature_sets(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    composition = list(groups["composition"])
    processing = list(groups["processing"])
    microstructure = list(groups["microstructure"])
    return {
        "C": composition,
        "C+Proc": composition + processing,
        "C+Proc+Micro": composition + processing + microstructure,
    }


def validate_dataset(
    frame: pd.DataFrame,
    groups: Dict[str, List[str]],
    targets: List[str],
    expected_rows: int = 327,
) -> pd.DataFrame:
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} records; received {len(frame)}.")

    predictors = sum(groups.values(), [])
    missing = [column for column in predictors + targets if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    target_names = {normalize_name(column).lower() for column in targets}
    for column in predictors:
        normalized = normalize_name(column).lower()
        if normalized in target_names:
            raise ValueError(f"Target leakage detected: {column}")
        if "hardness" in normalized or "硬度" in normalized:
            raise ValueError(f"Hardness is not an M3PP predictor: {column}")

    selected = frame[predictors + targets].copy()
    selected = selected.apply(pd.to_numeric, errors="raise")
    if selected.isna().any().any():
        raise ValueError("The 327-record dataset must be complete.")
    return selected.reset_index(drop=True)
