import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from import_competition import import_archive
from vacv.chemistry import featurize
from vacv.data import TARGET, check_submission, load_inputs
from vacv.models import fit_model, predict_model
from vacv.pipeline import predict, run
from vacv.validation import make_folds, metrics


@pytest.fixture
def dataset(tmp_path):
    # Synthetic software fixture only; never part of competition training.
    smiles = ["CCO", "OCC", "CCC", "CCCC", "CCN", "CCCl", "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "C1CCC1"]
    train = pd.DataFrame({"SMILES": smiles, TARGET: [0, 5, 10, 20, 30, 10000, 50, 60, 70, 80, 90, 110], "CC50_uM": range(12), "IC50_uM_VACV": range(12), "Title": ["metadata"] * 12})
    test = pd.DataFrame({"index": ["001", "002", "003"], "SMILES": ["OCC", "CCO", "invalid"], "CC50_uM": [1, 2, 3]})
    sample = pd.DataFrame({"index": test["index"], TARGET: ["not_a_label", "never_read", "unused"]})
    config = json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))
    config.pop("input_sha256", None)
    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        config[name] = str(path)
    config.update({"dataset_version": "SYNTHETIC_TEST_FIXTURE_ONLY", "expected": {"train_rows": 12, "test_rows": 3, "zero_targets": 1, "targets_10000": 1}, "folds": 2})
    config["fingerprint"]["bits"] = 64
    config["extra_trees"].update(n_estimators=4, n_jobs=1)
    return config


def test_feature_allowlist_and_sample_values_are_ignored(dataset):
    train, test, _ = load_inputs(dataset)
    assert set(train.columns) == {"train_row_id", "SMILES", TARGET}
    assert set(test.columns) == {"index", "SMILES"}
    assert test["index"].tolist() == ["001", "002", "003"]
    _, before = featurize(train.SMILES, dataset["fingerprint"])
    original = pd.read_csv(dataset["train"])
    original["CC50_uM"] = "forbidden_concentration"
    original["Title"] = "different_metadata"
    original.to_csv(dataset["train"], index=False)
    changed, _, _ = load_inputs(dataset)
    _, after = featurize(changed.SMILES, dataset["fingerprint"])
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(train[TARGET], changed[TARGET])


def test_dataset_version_mismatch_does_not_repair_targets(dataset):
    original = Path(dataset["train"]).read_bytes()
    dataset["expected"]["zero_targets"] = 83
    with pytest.raises(ValueError, match="version mismatch"):
        load_inputs(dataset)
    assert Path(dataset["train"]).read_bytes() == original


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1])
def test_invalid_labels_rejected(dataset, bad):
    frame = pd.read_csv(dataset["train"])
    frame[TARGET] = frame[TARGET].astype(float)
    frame.loc[0, TARGET] = bad
    frame.to_csv(dataset["train"], index=False)
    with pytest.raises(ValueError):
        load_inputs(dataset)


def test_external_data_disabled(dataset):
    dataset["external_data"] = True
    with pytest.raises(ValueError, match="External data"):
        load_inputs(dataset)


def test_canonical_policy_stereo_salts_and_invalid(dataset):
    molecules = ["CCO", "OCC", "C[C@H](O)F", "C[C@@H](O)F", "CCO.[Na+]", "invalid", "", "[CH3:1][CH2:2][OH:3]"]
    table, matrix = featurize(molecules, dataset["fingerprint"])
    assert table.structure_id[0] == table.structure_id[1] == table.structure_id[7]
    assert table.structure_id[2] != table.structure_id[3]
    assert table.structure_id[0] != table.structure_id[4]
    assert table.valid.tolist() == [True] * 5 + [False, False, True]
    assert np.isnan(matrix[5:7]).all()
    np.testing.assert_array_equal(matrix[0], matrix[1])


def test_group_folds_reproducible_and_no_leakage(dataset):
    train, _, _ = load_inputs(dataset)
    table, _ = featurize(train.SMILES, dataset["fingerprint"])
    folds = make_folds(table, dataset)
    pd.testing.assert_frame_equal(folds, make_folds(table, dataset))
    for (scheme, seed), split in folds.groupby(["scheme", "seed"]):
        assert sorted(split.row_position) == list(range(len(train)))
        assert split.groupby("group_id").fold.nunique().eq(1).all()
        assert split.loc[split.row_position.isin([0, 1]), "fold"].nunique() == 1
        for fold in split.fold.unique():
            assert not set(split.loc[split.fold == fold, "group_id"]) & set(split.loc[split.fold != fold, "group_id"])


def test_log_inverse_and_invalid_fallback(dataset):
    x = np.ones((3, 64))
    bundle = fit_model("ridge_log1p", x, np.full(3, 99.0), dataset)
    pred, _ = predict_model(bundle, np.vstack([x[:1], np.full((1, 64), np.nan)]), [True, False])
    np.testing.assert_allclose(pred, [99.0, 99.0])
    assert metrics([0, 2], [0, 0])["mse"] == 2.0


@pytest.mark.parametrize("mutation", ["order", "nan", "extra", "duplicate"])
def test_submission_rejects_bad_format(dataset, tmp_path, mutation):
    frame = pd.DataFrame({"index": ["001", "002", "003"], TARGET: [1.0, 2.0, 3.0]})
    if mutation == "order":
        frame = frame.iloc[::-1]
    elif mutation == "nan":
        frame.loc[0, TARGET] = np.inf
    elif mutation == "extra":
        frame["CC50"] = 0
    else:
        frame.loc[1, "index"] = "001"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        check_submission(path, dataset["sample"])


def test_invalid_train_kept_in_audit_and_stops_training(dataset, tmp_path):
    frame = pd.read_csv(dataset["train"])
    frame.loc[0, "SMILES"] = "broken"
    frame.to_csv(dataset["train"], index=False)
    output = tmp_path / "bad_run"
    with pytest.raises(ValueError, match="Invalid train"):
        run(dataset, output)
    assert len(pd.read_csv(output / "train_for_model.csv")) == 12
    assert not (output / "model.joblib").exists()


def test_end_to_end_oof_reload_fallback_and_reproducibility(dataset, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    original_train = Path(dataset["train"]).read_bytes()
    original_test = Path(dataset["test"]).read_bytes()
    run(dataset, first)
    run(copy.deepcopy(dataset), second)
    for name in ("folds.csv", "oof.csv", "submission.csv", "leaderboard.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    oof = pd.read_csv(first / "oof.csv")
    for _, part in oof.groupby(["model", "scheme", "seed"]):
        assert len(part) == 12 and part.train_row_id.nunique() == 12
    means = oof[oof.model == "mean"]
    for _, part in means.groupby(["scheme", "seed"]):
        for fold in part.fold.unique():
            expected = part.loc[part.fold != fold, "y_true"].mean()
            np.testing.assert_allclose(part.loc[part.fold == fold, "y_pred"], expected)
    reloaded = tmp_path / "reloaded.csv"
    predict(str(first / "model.joblib"), dataset["test"], dataset["sample"], str(reloaded))
    assert reloaded.read_bytes() == (first / "submission.csv").read_bytes()
    prediction = pd.read_csv(reloaded)[TARGET]
    assert prediction[0] == prediction[1]
    assert prediction[2] == pytest.approx(pd.read_csv(dataset["train"])[TARGET].mean())
    assert Path(dataset["train"]).read_bytes() == original_train
    assert Path(dataset["test"]).read_bytes() == original_test
    assert not oof.isna().any().any()
    bundle = joblib.load(first / "model.joblib")
    assert bundle["manifest"]["external_data_used"] is False
    assert bundle["estimator"].n_features_in_ == 64


def test_archive_import_no_overwrite_or_path_traversal(tmp_path):
    from zipfile import ZipFile
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as stream:
        for name in ("train.csv", "test.csv", "sample_submit.csv"):
            stream.writestr(name, b"original bytes\n")
        stream.writestr("../escape.txt", b"not extracted")
    output = tmp_path / "raw"
    import_archive(archive, output)
    assert not (tmp_path / "escape.txt").exists()
    (output / "train.csv").write_text("do not replace", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        import_archive(archive, output)
    assert (output / "train.csv").read_text() == "do not replace"
