"""Identical dual-scheme inner selection for every outer fold and final training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kernel import fit_kernel_candidate, tanimoto_kernel
from .models import predict_model
from .search import inner_assignments
from .validation import metrics


def candidate_grid(config):
    candidates = []
    settings = config["robust_search"]
    for spec in settings["feature_sets"]:
        for alpha in settings["ridge_alphas"]:
            candidates.append({"id": f"{spec['name']}:ridge:{alpha:g}", "feature_set": spec["name"], "family": "ridge", "alpha": alpha})
        for alpha in settings["kernel_alphas"]:
            for weight in settings["kernel_weights"]:
                family = "kernel" if weight == 1 else "blend"
                candidates.append({"id": f"{spec['name']}:{family}:{alpha:g}:w{weight:g}", "feature_set": spec["name"], "family": family,
                                   "alpha": alpha, "kernel_weight": weight, "linear_alpha": settings["blend_linear_alpha"]})
    return candidates


def robust_inner_search(matrices, y, structure_groups, scaffold_groups, config, seed, kernels=None):
    settings = config["robust_search"]
    specs = {s["name"]: s for s in settings["feature_sets"]}
    kernels = kernels if kernels is not None else {name: tanimoto_kernel(x, x) for name, x in matrices.items()}
    assignments = {"structure": inner_assignments(structure_groups, settings["inner_folds"], seed),
                   "scaffold": inner_assignments(scaffold_groups, settings["inner_folds"], seed)}
    rows = []
    for candidate in candidate_grid(config):
        name = candidate["feature_set"]
        x, kernel = matrices[name], kernels[name]
        scores = {}
        for scheme, membership in assignments.items():
            prediction = np.empty(len(y))
            for fold in sorted(set(membership)):
                held = membership == fold
                fit = ~held
                model = fit_kernel_candidate(x[fit], y[fit], specs[name], candidate, config["clip_min"], gram=kernel[np.ix_(fit, fit)])
                if candidate["family"] == "ridge":
                    prediction[held], _ = predict_model(model, x[held], np.ones(held.sum(), dtype=bool))
                else:
                    raw = model["estimator"].predict_with_kernel(x[held], kernel[np.ix_(held, fit)])
                    if not np.isfinite(raw).all():
                        raise ValueError("Non-finite kernel prediction")
                    prediction[held] = np.maximum(raw, config["clip_min"]) if config["clip_min"] is not None else raw
            scores[scheme + "_mse"] = metrics(y, prediction)["mse"]
        rows.append({"candidate_id": candidate["id"], **scores, "objective": (scores["structure_mse"] + scores["scaffold_mse"]) / 2})
    return pd.DataFrame(rows).sort_values(["objective", "candidate_id"]).reset_index(drop=True), assignments
