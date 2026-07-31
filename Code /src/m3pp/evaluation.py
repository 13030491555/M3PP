from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .models import CrossFittedStackingRegressor, model_factories


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100),
    }


def repeated_folds(
    row_count: int,
    seeds: Sequence[int] = (42, 2023, 2024),
    n_splits: int = 5,
):
    indices = np.arange(row_count)
    for repeat, seed in enumerate(seeds, start=1):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold, (train_index, valid_index) in enumerate(splitter.split(indices), start=1):
            yield repeat, int(seed), fold, train_index, valid_index


def run_evaluation(
    frame: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    targets: List[str],
    output_dir: Path,
    seeds: Sequence[int] = (42, 2023, 2024),
    n_splits: int = 5,
    inner_splits: int = 5,
    include_comparators: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = []
    predictions = []
    ridge_coefficients = []

    for target in targets:
        for stage, columns in feature_sets.items():
            X = frame[columns].to_numpy(dtype=float)
            y = frame[target].to_numpy(dtype=float)

            for repeat, seed, fold, train_index, valid_index in repeated_folds(
                len(frame),
                seeds,
                n_splits,
            ):
                factories = model_factories(seed, include_comparators)
                comparator_names = ["LinearRegression", "RandomForest"]
                if include_comparators:
                    comparator_names.extend(["CNN", "GNN"])

                for model_name in comparator_names:
                    model = factories[model_name]()
                    started = time.time()
                    model.fit(X[train_index], y[train_index])
                    predicted = model.predict(X[valid_index])
                    fold_metrics.append(
                        {
                            "target": target,
                            "feature_set": stage,
                            "model": model_name,
                            "repeat": repeat,
                            "seed": seed,
                            "fold": fold,
                            "fit_seconds": time.time() - started,
                            **regression_metrics(y[valid_index], predicted),
                        }
                    )
                    predictions.extend(
                        {
                            "target": target,
                            "feature_set": stage,
                            "model": model_name,
                            "repeat": repeat,
                            "seed": seed,
                            "fold": fold,
                            "row_index": int(row_index),
                            "y_true": float(observed),
                            "y_pred": float(estimate),
                        }
                        for row_index, observed, estimate in zip(
                            valid_index,
                            y[valid_index],
                            predicted,
                        )
                    )

                base_models = [
                    ("XGBoost", factories["XGBoost"]),
                    ("TabNet", factories["TabNet"]),
                ]
                stack = CrossFittedStackingRegressor(
                    base_models,
                    inner_splits=inner_splits,
                    seed=seed + fold,
                    alpha=1.0,
                )
                started = time.time()
                stack.fit(X[train_index], y[train_index])
                predicted = stack.predict(X[valid_index])
                model_name = "XGBoost-TabNet-Ridge"
                fold_metrics.append(
                    {
                        "target": target,
                        "feature_set": stage,
                        "model": model_name,
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "fit_seconds": time.time() - started,
                        **regression_metrics(y[valid_index], predicted),
                    }
                )
                predictions.extend(
                    {
                        "target": target,
                        "feature_set": stage,
                        "model": model_name,
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "row_index": int(row_index),
                        "y_true": float(observed),
                        "y_pred": float(estimate),
                    }
                    for row_index, observed, estimate in zip(
                        valid_index,
                        y[valid_index],
                        predicted,
                    )
                )
                ridge_coefficients.append(
                    {
                        "target": target,
                        "feature_set": stage,
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "intercept": float(stack.ridge_.intercept_),
                        "xgboost_coefficient": float(stack.ridge_.coef_[0]),
                        "tabnet_coefficient": float(stack.ridge_.coef_[1]),
                    }
                )

    fold_table = pd.DataFrame(fold_metrics)
    prediction_table = pd.DataFrame(predictions)
    ridge_table = pd.DataFrame(ridge_coefficients)
    fold_table.to_csv(output_dir / "fold_metrics.csv", index=False)
    prediction_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    ridge_table.to_csv(output_dir / "ridge_coefficients.csv", index=False)

    repeat_rows = []
    for keys, group in prediction_table.groupby(
        ["target", "feature_set", "model", "repeat", "seed"]
    ):
        repeat_rows.append(
            {
                **dict(zip(["target", "feature_set", "model", "repeat", "seed"], keys)),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    repeat_table = pd.DataFrame(repeat_rows)
    repeat_table.to_csv(output_dir / "repeat_oof_metrics.csv", index=False)

    pooled_rows = []
    for keys, group in prediction_table.groupby(["target", "feature_set", "model"]):
        pooled_rows.append(
            {
                **dict(zip(["target", "feature_set", "model"], keys)),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    pd.DataFrame(pooled_rows).to_csv(output_dir / "pooled_oof_metrics.csv", index=False)

    averaged = (
        prediction_table.groupby(["target", "feature_set", "model", "row_index"], as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    )
    averaged.to_csv(output_dir / "mean_oof_predictions.csv", index=False)

    averaged_rows = []
    for keys, group in averaged.groupby(["target", "feature_set", "model"]):
        averaged_rows.append(
            {
                **dict(zip(["target", "feature_set", "model"], keys)),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    pd.DataFrame(averaged_rows).to_csv(
        output_dir / "mean_oof_metrics.csv",
        index=False,
    )
