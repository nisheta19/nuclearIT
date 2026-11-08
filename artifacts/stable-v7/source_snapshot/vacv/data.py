from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .target_status import target_status

TARGET = "SI VACV"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def headers(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream))
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate CSV headers in {path}")
    return columns


def load_inputs(config: dict):
    if config.get("external_data") is not False:
        raise ValueError("External data are not authorized for this pipeline")
    train_path, test_path, sample_path = [Path(config[k]) for k in ("train", "test", "sample")]
    for name, expected_hash in config.get("input_sha256", {}).items():
        if name not in ("train", "test", "sample") or sha256(config[name]) != expected_hash:
            raise ValueError(f"Input hash mismatch: {name}. Create an explicit new dataset configuration; no data repaired.")
    train_cols = headers(train_path)
    required = ["SMILES", TARGET]
    if not set(required).issubset(train_cols):
        raise ValueError("Train requires SMILES and SI VACV")
    # Read an explicit allowlist. Forbidden columns never reach feature generation.
    train = pd.read_csv(train_path, usecols=required + (["train_row_id"] if "train_row_id" in train_cols else []), dtype=str, keep_default_na=False)
    if "train_row_id" not in train:
        train.insert(0, "train_row_id", np.arange(len(train)).astype(str))
    if train.train_row_id.eq("").any() or train.train_row_id.duplicated().any():
        raise ValueError("Train row IDs must be nonempty and unique")
    train[TARGET] = pd.to_numeric(train[TARGET], errors="raise")
    y = train[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Targets must be finite and nonnegative; labels were not repaired")
    headers(test_path)
    test = pd.read_csv(test_path, usecols=["index", "SMILES"], dtype=str, keep_default_na=False)
    if headers(sample_path) != ["index", TARGET]:
        raise ValueError("Sample columns must be index,SI VACV in that order")
    # Sample numeric values are deliberately never read.
    sample_ids = pd.read_csv(sample_path, usecols=["index"], dtype=str, keep_default_na=False)["index"]
    if test["index"].eq("").any() or test["index"].duplicated().any():
        raise ValueError("Test indices must be nonempty and unique")
    if not test["index"].equals(sample_ids):
        raise ValueError("Test/sample indices or row order differ")
    observed = {"train_rows": len(train), "test_rows": len(test), "zero_targets": int((y == 0).sum()), "targets_10000": int((y == 10000).sum())}
    for key, expected in config["expected"].items():
        if observed[key] != expected:
            raise ValueError(f"Dataset version mismatch: {key}={observed[key]}, expected {expected}. No labels changed.")
    manifest = {
        "dataset_version": config["dataset_version"],
        "observed": observed,
        "inputs": {key: {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size, "columns": headers(path)} for key, path in zip(("train", "test", "sample"), (train_path, test_path, sample_path))},
        "features_from": ["SMILES"], "external_data_used": False,
        "sample_values_used": False, "target_changes": 0,
    }
    manifest['target_review'] = target_status(manifest)
    return train, test, manifest


def check_submission(submission: str | Path, sample: str | Path) -> dict:
    if headers(submission) != ["index", TARGET] or headers(sample) != ["index", TARGET]:
        raise ValueError("Columns must be exactly index,SI VACV")
    submitted = pd.read_csv(submission, dtype=str, keep_default_na=False)
    sample_ids = pd.read_csv(sample, usecols=["index"], dtype=str, keep_default_na=False)["index"]
    if submitted["index"].eq("").any() or submitted["index"].duplicated().any() or sample_ids.duplicated().any():
        raise ValueError("Empty or duplicate indices")
    if not submitted["index"].equals(sample_ids):
        raise ValueError("Submission indices/order/row count differ from sample")
    values = pd.to_numeric(submitted[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite predictions")
    return {"valid_format": True, "rows": len(submitted), "sha256": sha256(submission), "quality_verified": False}
