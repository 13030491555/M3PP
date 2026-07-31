from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m3pp.models import CrossFittedStackingRegressor
from m3pp.preprocessing import FoldPreprocessor
from m3pp.schema import build_feature_sets, validate_dataset


def test_preprocessor_shape():
    X = np.column_stack([np.arange(20), np.ones(20), np.arange(20) ** 2])
    transformed = FoldPreprocessor().fit_transform(X)
    assert transformed.shape == (20, 2)


def test_cross_fitted_stacking():
    from sklearn.linear_model import LinearRegression

    X = np.arange(60, dtype=float).reshape(30, 2)
    y = X[:, 0] * 0.5 + X[:, 1]
    factories = [
        ("a", lambda: LinearRegression()),
        ("b", lambda: LinearRegression()),
    ]
    model = CrossFittedStackingRegressor(factories, inner_splits=5, seed=42)
    model.fit(X, y)
    prediction = model.predict(X[:3])
    assert prediction.shape == (3,)


def test_dataset_contract():
    groups = {
        "composition": [f"c{i}" for i in range(38)],
        "processing": [f"p{i}" for i in range(24)],
        "microstructure": [f"m{i}" for i in range(6)],
    }
    targets = ["Yield strength(MPa)", "Tensile strength(MPa)", "Total elongation(%)"]
    columns = sum(groups.values(), []) + targets
    frame = pd.DataFrame(np.ones((327, len(columns))), columns=columns)
    validated = validate_dataset(frame, groups, targets)
    assert len(validated) == 327
    assert len(build_feature_sets(groups)["C+Proc+Micro"]) == 68
