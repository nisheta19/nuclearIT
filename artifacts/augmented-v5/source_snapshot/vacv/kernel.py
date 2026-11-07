"""Tanimoto kernel regressors with training-only target offset and fixed blends."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def tanimoto_kernel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("Tanimoto requires finite nonnegative feature matrices of matching width")
    with threadpool_limits(limits=1):
        intersection = a @ b.T
    denominator = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - intersection
    result = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)
    result[denominator == 0] = 1.0
    return result


class TanimotoRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1.0, kernel_weight=1.0, linear_alpha=10.0):
        self.alpha = alpha
        self.kernel_weight = kernel_weight
        self.linear_alpha = linear_alpha

    def fit(self, X, y, gram=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if self.alpha <= 0 or not 0 <= self.kernel_weight <= 1 or not np.isfinite(y).all() or y.shape != (len(X),):
            raise ValueError("Invalid kernel regression parameters or targets")
        self.train_features_ = X.copy()
        self.n_features_in_ = X.shape[1]
        self.target_offset_ = float(y.mean())
        matrix = tanimoto_kernel(X, X) if gram is None else np.asarray(gram, dtype=float)
        if matrix.shape != (len(X), len(X)) or not np.isfinite(matrix).all():
            raise ValueError("Training kernel matrix shape/value mismatch")
        with threadpool_limits(limits=1):
            self.kernel_model_ = KernelRidge(alpha=self.alpha, kernel="precomputed").fit(matrix, y - self.target_offset_)
            self.linear_model_ = Ridge(alpha=self.linear_alpha, solver="lsqr", tol=1e-6).fit(X, y) if self.kernel_weight < 1 else None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.predict_with_kernel(X, tanimoto_kernel(X, self.train_features_))

    def predict_with_kernel(self, X, matrix):
        with threadpool_limits(limits=1):
            prediction = self.kernel_model_.predict(matrix) + self.target_offset_
            if self.linear_model_ is not None:
                prediction = self.kernel_weight * prediction + (1 - self.kernel_weight) * self.linear_model_.predict(X)
        return prediction


def fit_kernel_candidate(x, y, spec, candidate, clip_min=0.0, gram=None):
    with threadpool_limits(limits=1):
        if candidate["family"] == "ridge":
            model = Ridge(alpha=candidate["alpha"], solver="lsqr", tol=1e-6).fit(x, y)
        else:
            model = TanimotoRidge(alpha=candidate["alpha"], kernel_weight=candidate["kernel_weight"], linear_alpha=candidate["linear_alpha"]).fit(x, y, gram=gram)
    return {"name": candidate["id"], "estimator": model, "log_target": False, "fallback_mean": float(np.mean(y)),
            "fingerprint": spec, "clip_min": clip_min, "candidate": candidate}
