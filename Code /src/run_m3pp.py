from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from m3pp.evaluation import run_evaluation
from m3pp.schema import (
    DEFAULT_TARGETS,
    build_feature_sets,
    canonicalize_columns,
    load_feature_manifest,
    validate_dataset,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Data file must be .xlsx, .xls, or .csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-comparators", action="store_true")
    parser.add_argument("--shap", action="store_true")
    arguments = parser.parse_args()

    frame = canonicalize_columns(read_table(Path(arguments.data)))
    groups = load_feature_manifest(Path(arguments.feature_manifest))
    validated = validate_dataset(frame, groups, DEFAULT_TARGETS, expected_rows=327)
    feature_sets = build_feature_sets(groups)
    output_dir = Path(arguments.output_dir)

    run_evaluation(
        validated,
        feature_sets,
        DEFAULT_TARGETS,
        output_dir,
        seeds=(42, 2023, 2024),
        n_splits=5,
        inner_splits=5,
        include_comparators=not arguments.skip_comparators,
    )

    if arguments.shap:
        from m3pp.shap_analysis import run_shap_analysis

        modalities = {
            feature: modality
            for modality, features in groups.items()
            for feature in features
        }
        full_features = feature_sets["C+Proc+Micro"]
        for target in DEFAULT_TARGETS:
            run_shap_analysis(
                validated[full_features],
                validated[target],
                modalities,
                output_dir / "shap" / target.replace("/", "_"),
                seed=42,
            )


if __name__ == "__main__":
    main()
