import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from .test_pipeline import dataset
from vacv.chemistry import featurize
from vacv.data import load_inputs, sha256
from vacv.experiment import nested_run
from vacv.pipeline import predict
from vacv.search import fit_candidate, inner_assignments


def test_hash_pin_detects_changes_even_when_row_counts_match(dataset):
    dataset["input_sha256"] = {"train": sha256(dataset["train"])}
    load_inputs(dataset)
    frame = pd.read_csv(dataset["train"])
    frame.loc[1, "SI VACV"] = 6
    frame.to_csv(dataset["train"], index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(dataset)


def test_descriptor_scaler_fits_only_training_rows():
    spec = {"name": "demo", "radius": 2, "bits": 2, "descriptors": ["MolWt"], "include_chirality": True}
    x = np.array([[0, 1, 10.0], [1, 0, 20.0], [1, 1, 30.0]])
    model = fit_candidate(x, np.array([1.0, 2.0, 3.0]), spec, 1)
    scaler = model["estimator"].steps[0][1].named_transformers_["descriptors"]
    assert scaler.mean_[0] == 20
    model["estimator"].predict([[1, 1, 100000]])
    assert scaler.mean_[0] == 20


def test_count_and_descriptor_features_are_finite_and_structure_only():
    spec = {"name": "counts", "radius": 2, "bits": 64, "kind": "count", "descriptors": ["MolWt", "TPSA"], "include_chirality": True}
    _, features = featurize(["CCCCCC", "CCCCCC"], spec)
    assert features.shape == (2, 66)
    assert features[0, :64].max() > 1
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[0], features[1])


def test_too_few_inner_groups_fails_instead_of_leaking():
    with pytest.raises(ValueError, match="inner groups"):
        inner_assignments(["same", "same"], 3, 42)


def test_nested_membership_coverage_reload_and_reports(dataset, tmp_path):
    # The nested fixture needs at least two scaffold groups in every outer training split.
    frame = pd.read_csv(dataset["train"])
    frame.loc[2:5, "SMILES"] = ["c1ccoc1", "c1ccsc1", "C1COCC1", "C1CCNC1"]
    frame.to_csv(dataset["train"], index=False)
    base = dataset["fingerprint"]
    dataset["search"] = {"inner_folds": 2, "inner_seed": 917, "alphas": [1.0, 10.0], "feature_sets": [
        {**base, "name": "r2_binary", "kind": "binary", "descriptors": []},
        {**base, "name": "r2_descriptors", "kind": "binary", "descriptors": ["MolWt", "TPSA"]},
    ]}
    output = tmp_path / "nested"
    result = nested_run(dataset, output)
    assert result["valid_format"] and result["rows"] == 3
    inner = pd.read_csv(output / "inner_folds.csv")
    outer = pd.read_csv(output / "folds.csv")
    for (scheme, seed, fold), part in inner.groupby(["scheme", "seed", "outer_fold"]):
        outer_part = outer[(outer.scheme == scheme) & (outer.seed == seed)]
        validation = outer_part[outer_part.fold == fold]
        assert set(part.train_row_id) == set(outer_part[outer_part.fold != fold].train_row_id)
        assert not set(part.group_id) & set(validation.group_id)
        assert part.groupby("group_id").inner_fold.nunique().eq(1).all()
    oof = pd.read_csv(output / "oof.csv")
    for _, part in oof.groupby(["scheme", "seed", "model"]):
        assert len(part) == 12 and part.train_row_id.nunique() == 12
    selected = result["strategy"]
    choices = pd.read_csv(output / "outer_choices.csv")
    for choice in choices.itertuples():
        part = oof[(oof.scheme == choice.scheme) & (oof.seed == choice.seed) & (oof.fold == choice.outer_fold)]
        np.testing.assert_array_equal(part[part.model == "tuned_ridge"].y_pred, part[part.model == "tuned_" + choice.feature_set].y_pred)
    predict(str(output / "model.joblib"), dataset["test"], dataset["sample"], str(tmp_path / "repeat.csv"))
    assert (tmp_path / "repeat.csv").read_bytes() == (output / "submission.csv").read_bytes()
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert diagnostics["label_changes"] == 0
    assert diagnostics["invalid_test"] == 1
    assert (output / "charts/oof_scatter.png").exists()
    assert "SYNTHETIC_TEST_FIXTURE_ONLY" in (output / "report.md").read_text(encoding="utf-8")
    from vacv.server import Workbench
    app = Workbench(output)
    response = app.predict("OCC")
    assert response["predicted_si"] == pytest.approx(pd.read_csv(output / "submission.csv")["SI VACV"].iloc[0])
    assert response["molecule_image"].startswith("data:image/svg+xml;base64,")
    assert response["max_similarity"] == 1.0 and response["training_matches"] == 2
    for value in ("", "invalid", None, "C" * 2001):
        with pytest.raises(ValueError):
            app.predict(value)
