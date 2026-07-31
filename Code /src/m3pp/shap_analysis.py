from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import CrossFittedStackingRegressor, model_factories


def run_shap_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    modalities: dict,
    output_dir: Path,
    seed: int = 42,
):
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    factories = model_factories(seed, include_comparators=False)
    stack = CrossFittedStackingRegressor(
        [
            ("XGBoost", factories["XGBoost"]),
            ("TabNet", factories["TabNet"]),
        ],
        inner_splits=5,
        seed=seed,
        alpha=1.0,
    )
    values = X.to_numpy(dtype=float)
    stack.fit(values, y.to_numpy(dtype=float))

    random = np.random.default_rng(seed)
    background_indices = random.choice(len(values), min(50, len(values)), replace=False)
    explain_indices = random.choice(len(values), min(100, len(values)), replace=False)
    explainer = shap.Explainer(
        stack.predict,
        values[background_indices],
        algorithm="permutation",
        feature_names=X.columns.tolist(),
    )
    explanation = explainer(
        values[explain_indices],
        max_evals=max(2 * values.shape[1] + 1, 101),
    )

    mean_absolute = np.abs(explanation.values).mean(axis=0)
    feature_table = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": mean_absolute,
        }
    )
    feature_table["modality"] = feature_table["feature"].map(modalities)
    feature_table = feature_table.sort_values("mean_abs_shap", ascending=False)
    feature_table.to_csv(output_dir / "feature_importance.csv", index=False)

    modality_table = (
        feature_table.groupby("modality", as_index=False)["mean_abs_shap"].sum()
    )
    modality_table["share_percent"] = (
        100.0 * modality_table["mean_abs_shap"] / modality_table["mean_abs_shap"].sum()
    )
    modality_table.to_csv(output_dir / "modality_contributions.csv", index=False)
