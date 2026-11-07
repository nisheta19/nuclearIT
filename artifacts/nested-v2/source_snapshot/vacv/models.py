from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


def make_estimator(name: str, config: dict):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name in ("ridge_si", "ridge_log1p"):
        # Binary fingerprints need no fitted imputation/scaling/feature selection.
        return Ridge(alpha=config["ridge_alpha"], solver="lsqr", tol=1e-6)
    if name in ("extra_trees_si", "extra_trees_log1p"):
        return ExtraTreesRegressor(random_state=config["model_seed"], **config["extra_trees"])
    raise ValueError(f"Unknown candidate {name}")


def fit_model(name: str, features, y, config: dict) -> dict:
    if not np.isfinite(features).all():
        raise ValueError("Invalid structure reached model training")
    log_target = name.endswith("_log1p")
    estimator = make_estimator(name, config)
    estimator.fit(features, np.log1p(y) if log_target else y)
    return {"name": name, "estimator": estimator, "log_target": log_target, "fallback_mean": float(np.mean(y)), "fingerprint": config["fingerprint"], "clip_min": config["clip_min"]}


def predict_model(bundle: dict, features, valid) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(valid, dtype=bool)
    raw = np.full(len(features), bundle["fallback_mean"], dtype=float)
    if valid.any():
        raw[valid] = bundle["estimator"].predict(features[valid])
        if bundle["log_target"]:
            raw[valid] = np.expm1(raw[valid])
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite prediction after inverse target transform")
    prediction = np.maximum(raw, bundle["clip_min"]) if bundle["clip_min"] is not None else raw.copy()
    return prediction, raw
