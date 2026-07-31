from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import FoldPreprocessor


def make_xgboost(seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )


class TabNetAdapter(BaseEstimator, RegressorMixin):
    """TabNet regressor with early stopping inside the current training fold."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y):
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=self.seed,
        )

        self.model_ = TabNetRegressor(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 0.02},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 20, "gamma": 0.8},
            mask_type="entmax",
            verbose=0,
            seed=self.seed,
        )
        self.model_.fit(
            X_train=X_fit,
            y_train=y_fit,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["rmse"],
            max_epochs=300,
            patience=30,
            batch_size=32,
            virtual_batch_size=16,
            num_workers=0,
            drop_last=False,
        )
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float32)).ravel()


def make_pipeline(model: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", FoldPreprocessor()),
            ("model", model),
        ]
    )


def model_factories(seed: int, include_comparators: bool = True) -> Dict[str, Callable]:
    factories: Dict[str, Callable] = {
        "LinearRegression": lambda: make_pipeline(LinearRegression()),
        "RandomForest": lambda: make_pipeline(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "XGBoost": lambda: make_pipeline(make_xgboost(seed)),
        "TabNet": lambda: make_pipeline(TabNetAdapter(seed)),
    }
    if include_comparators:
        from .neural_comparators import CNN1DRegressor, FeatureGraphRegressor

        factories["CNN"] = lambda: make_pipeline(CNN1DRegressor(seed=seed))
        factories["GNN"] = lambda: make_pipeline(FeatureGraphRegressor(seed=seed))
    return factories


class CrossFittedStackingRegressor(BaseEstimator, RegressorMixin):
    """XGBoost–TabNet predictions cross-fitted before Ridge meta-learning."""

    def __init__(
        self,
        base_factories: Sequence[Tuple[str, Callable]],
        inner_splits: int = 5,
        seed: int = 42,
        alpha: float = 1.0,
    ):
        self.base_factories = list(base_factories)
        self.inner_splits = inner_splits
        self.seed = seed
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        splitter = KFold(
            n_splits=self.inner_splits,
            shuffle=True,
            random_state=self.seed,
        )

        meta_features = np.zeros((len(y), len(self.base_factories)), dtype=float)
        for model_index, (_, factory) in enumerate(self.base_factories):
            for train_index, valid_index in splitter.split(X):
                model = factory()
                model.fit(X[train_index], y[train_index])
                meta_features[valid_index, model_index] = model.predict(X[valid_index])

        self.ridge_ = Ridge(alpha=self.alpha).fit(meta_features, y)
        self.base_models_ = []
        for name, factory in self.base_factories:
            model = factory()
            model.fit(X, y)
            self.base_models_.append((name, model))
        return self

    def predict(self, X):
        meta_features = np.column_stack(
            [model.predict(X) for _, model in self.base_models_]
        )
        return self.ridge_.predict(meta_features)
