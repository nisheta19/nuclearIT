from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_folds(structures: pd.DataFrame, config: dict) -> pd.DataFrame:
    records = []
    schemes = [("structure", seed, "structure_id") for seed in config["structure_seeds"]]
    schemes.append(("scaffold", -1, "scaffold_id"))
    for scheme, seed, group_column in schemes:
        groups = structures[group_column].to_numpy()
        n_splits = min(config["folds"], len(set(groups)))
        if n_splits < 2:
            raise ValueError(f"Too few {scheme} groups for validation")
        # Structure folds shuffle groups. Scaffold folds balance group sizes without shuffling.
        splitter = GroupKFold(n_splits=n_splits, shuffle=scheme == "structure", random_state=seed if scheme == "structure" else None)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
            if set(groups[train_idx]) & set(groups[valid_idx]):
                raise AssertionError("Group leakage")
            for row in valid_idx:
                records.append({"row_position": int(row), "scheme": scheme, "seed": seed, "fold": fold, "group_id": groups[row]})
    return pd.DataFrame(records)


def metrics(y, prediction) -> dict:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require matching, nonempty, finite vectors")
    mse = float(np.mean((y - prediction) ** 2))
    return {"n": len(y), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - prediction)))}
