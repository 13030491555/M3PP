from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap each feature using training-fold Q1/Q3 statistics."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.q1_ = np.quantile(values, 0.25, axis=0)
        self.q3_ = np.quantile(values, 0.75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.multiplier * iqr
        self.upper_ = self.q3_ + self.multiplier * iqr
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)


class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """Training-fold variance filtering, IQR capping, and standard scaling."""

    def __init__(
        self,
        variance_threshold: float = 1e-12,
        iqr_multiplier: float = 1.5,
    ):
        self.variance_threshold = variance_threshold
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Predictors must be complete and finite.")

        self.variance_filter_ = VarianceThreshold(self.variance_threshold)
        filtered = self.variance_filter_.fit_transform(values)
        if filtered.shape[1] == 0:
            raise ValueError("All predictors were removed by variance filtering.")

        self.capper_ = IQRCapper(self.iqr_multiplier).fit(filtered)
        capped = self.capper_.transform(filtered)
        self.scaler_ = StandardScaler().fit(capped)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        filtered = self.variance_filter_.transform(values)
        capped = self.capper_.transform(filtered)
        return self.scaler_.transform(capped)
