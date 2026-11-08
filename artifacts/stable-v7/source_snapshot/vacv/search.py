"""Predeclared Ridge search. All fitted scaling stays inside each training split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .models import predict_model
from .validation import metrics


def inner_assignments(groups, folds, seed):
    groups = np.asarray(groups)
    k = min(folds, len(set(groups)))
    if k < 2:
        raise ValueError("Too few inner groups; outer validation is never used for tuning")
    assignments = np.full(len(groups), -1, dtype=int)
    splitter = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (fit, held) in enumerate(splitter.split(groups, groups=groups)):
        assert not set(groups[fit]) & set(groups[held])
        assignments[held] = fold
    return assignments


def fit_candidate(x, y, spec, alpha, clip_min=0.0):
    estimator = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6)
    if spec.get("descriptors"):
        split = spec["bits"]
        # Only descriptors are scaled; transformer is fitted on training rows only.
        estimator = make_pipeline(ColumnTransformer([
            ("fingerprint", "passthrough", list(range(split))),
            ("descriptors", StandardScaler(), list(range(split, x.shape[1]))),
        ]), estimator)
    with threadpool_limits(limits=1):
        estimator.fit(x, y)
    return {"name": f"ridge:{spec['name']}:alpha={alpha:g}", "estimator": estimator, "log_target": False,
            "fallback_mean": float(np.mean(y)), "fingerprint": spec, "clip_min": clip_min, "alpha": float(alpha)}


def search_candidates(matrices, y, groups, config, seed):
    assignment = inner_assignments(groups, config["search"]["inner_folds"], seed)
    rows = []
    for spec in config["search"]["feature_sets"]:
        x = matrices[spec["name"]]
        for alpha in config["search"]["alphas"]:
            prediction = np.empty(len(y))
            for fold in sorted(set(assignment)):
                held = assignment == fold
                model = fit_candidate(x[~held], y[~held], spec, alpha, config["clip_min"])
                prediction[held], _ = predict_model(model, x[held], np.ones(held.sum(), dtype=bool))
            rows.append({"feature_set": spec["name"], "alpha": float(alpha), **metrics(y, prediction)})
    ranking = pd.DataFrame(rows).sort_values(["mse", "feature_set", "alpha"], ascending=[True, True, False]).reset_index(drop=True)
    return ranking, assignment
