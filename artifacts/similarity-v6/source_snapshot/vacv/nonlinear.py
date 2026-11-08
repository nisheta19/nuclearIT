"""Structure-only similarity models; all landmarks come from training rows."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .kernel import tanimoto_kernel


class SimilarityRegressor(RegressorMixin, BaseEstimator):
    """Powered Tanimoto regression or a nonlinear gate and conditional means.

    The gate sees similarities to training molecules only. Its threshold is a
    statistical modeling parameter and does not reinterpret uncertain labels.
    """
    def __init__(self, alpha=1., power=1, C=10., gate="logistic", threshold=10000., transform="identity"):
        self.alpha, self.power, self.C = alpha, power, C
        self.gate, self.threshold, self.transform = gate, threshold, transform

    def _features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all() or (X < 0).any():
            raise ValueError("Expected finite nonnegative structural features")
        if self.transform not in ("identity", "log1p"):
            raise ValueError("Unknown structural transform")
        return np.log1p(X) if self.transform == "log1p" else X

    def fit(self, X, y):
        X, y = self._features(X), np.asarray(y, dtype=float)
        return self._fit(X, y, tanimoto_kernel(X, X) ** self.power)

    def _fit(self, X, y, K):
        if self.alpha <= 0 or self.power not in (1, 2) or self.C <= 0 or self.gate not in ("direct", "logistic", "linear", "ridge"):
            raise ValueError("Invalid similarity model parameters")
        if y.shape != (len(X),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("Invalid training targets")
        self.train_features_ = X.copy()
        self.n_features_in_ = X.shape[1]
        self.mean_ = float(y.mean())
        high = y >= self.threshold
        self.probability_ = float(high.mean())
        self.classifier_ = None
        with threadpool_limits(limits=1):
            if self.gate == "direct":
                self.coef_ = solve(K + self.alpha*np.eye(len(K)), y-self.mean_, assume_a="pos")
            else:
                if high.any() and not high.all():
                    if self.gate in ("logistic", "linear"):
                        self.classifier_ = LogisticRegression(C=self.C, max_iter=2000, tol=1e-7).fit(X if self.gate == "linear" else K, high)
                        if self.classifier_.n_iter_.max() >= 2000:
                            raise RuntimeError("Similarity gate did not converge")
                    else:
                        self.gate_coef_ = solve(K+self.alpha*np.eye(len(K)), high.astype(float)-self.probability_, assume_a="pos")
                self.parts_ = []
                for mask in (~high, high):
                    mean = float(y[mask].mean()) if mask.any() else self.mean_
                    coef = solve(K[np.ix_(mask,mask)]+self.alpha*np.eye(mask.sum()), y[mask]-mean, assume_a="pos") if mask.any() else np.empty(0)
                    self.parts_.append((mask, mean, coef))
        return self

    def predict(self, X):
        X = self._features(X)
        return self.predict_kernel(tanimoto_kernel(X, self.train_features_) ** self.power, X)

    def predict_kernel(self, K, X=None):
        with threadpool_limits(limits=1):
            if self.gate == "direct":
                return np.maximum(self.mean_ + K@self.coef_, 0)
            if self.classifier_ is not None:
                p = self.classifier_.predict_proba(X if self.gate == "linear" else K)[:, 1]
            elif self.gate == "ridge" and 0 < self.probability_ < 1:
                p = np.clip(self.probability_ + K@self.gate_coef_, 0, 1)
            else:
                p = np.full(len(K), self.probability_)
            low, high = [np.maximum(mean + K[:,mask]@coef, 0) for mask,mean,coef in self.parts_]
            return (1-p)*low + p*high
