from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .chemistry import featurize
from .data import TARGET, check_submission, save_json, sha256
from .models import fit_model, predict_model
from .pipeline import audit
from .search import fit_candidate, search_candidates
from .validation import make_folds, metrics


def nested_run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    train, test, train_st, test_st, baseline_x, baseline_test_x, manifest, summary = audit(config, output)
    code_root = Path(__file__).resolve().parent
    (output / "source_snapshot" / "vacv").mkdir(parents=True)
    for path in code_root.glob("*.py"):
        shutil.copy2(path, output / "source_snapshot" / "vacv" / path.name)
    folds = make_folds(train_st, config)
    folds["train_row_id"] = train.train_row_id.to_numpy()[folds.row_position.to_numpy()]
    folds.to_csv(output / "folds.csv", index=False)
    specs = {spec["name"]: spec for spec in config["search"]["feature_sets"]}
    matrices = {name: featurize(train.SMILES, spec)[1] for name, spec in specs.items()}
    y = train[TARGET].to_numpy(dtype=float)
    oof_parts, inner_logs, inner_membership, choices = [], [], [], []
    for (scheme, seed), part in folds.groupby(["scheme", "seed"], sort=False):
        assignment = part.sort_values("row_position").fold.to_numpy()
        grouping = train_st["structure_id" if scheme == "structure" else "scaffold_id"].to_numpy()
        for fold in sorted(set(assignment)):
            held = assignment == fold
            fit_positions, held_positions = np.flatnonzero(~held), np.flatnonzero(held)
            inner_seed = config["search"]["inner_seed"] + int(fold)
            ranking, inner = search_candidates({key: value[~held] for key, value in matrices.items()}, y[~held], grouping[~held], config, inner_seed)
            inner_logs.append(ranking.assign(scheme=scheme, seed=seed, outer_fold=fold))
            inner_membership.append(pd.DataFrame({"scheme": scheme, "seed": seed, "outer_fold": fold,
                "row_position": fit_positions, "train_row_id": train.train_row_id.to_numpy()[~held],
                "group_id": grouping[~held], "inner_fold": inner}))
            best = ranking.iloc[0]
            choices.append({"scheme": scheme, "seed": int(seed), "outer_fold": int(fold), "feature_set": best.feature_set, "alpha": float(best.alpha), "inner_mse": float(best.mse)})
            baseline = fit_model("ridge_si", baseline_x[~held], y[~held], config)
            predictions = {"baseline_ridge": predict_model(baseline, baseline_x[held], np.ones(held.sum(), dtype=bool)),
                           "mean": (np.full(held.sum(), y[~held].mean()), np.full(held.sum(), y[~held].mean()))}
            for name in specs:
                pick = ranking[ranking.feature_set == name].iloc[0]
                candidate = fit_candidate(matrices[name][~held], y[~held], specs[name], pick.alpha, config["clip_min"])
                predictions["tuned_" + name] = predict_model(candidate, matrices[name][held], np.ones(held.sum(), dtype=bool))
            predictions["tuned_ridge"] = predictions["tuned_" + best.feature_set]
            for name, (pred, raw) in predictions.items():
                oof_parts.append(pd.DataFrame({"train_row_id": train.train_row_id.to_numpy()[held], "row_position": held_positions,
                    "scheme": scheme, "seed": seed, "fold": fold, "model": name, "y_true": y[held], "y_pred": pred,
                    "y_pred_unclipped": raw, "squared_error": (y[held] - pred) ** 2}))
            # Checkpoint completed folds without manufacturing results for unfinished folds.
            pd.concat(inner_logs).to_csv(output / "inner_scores.csv", index=False)
            pd.concat(inner_membership).to_csv(output / "inner_folds.csv", index=False)
            pd.DataFrame(choices).to_csv(output / "outer_choices.csv", index=False)
            print(f"{scheme} seed={seed} outer_fold={fold}: {best.feature_set} alpha={best.alpha:g}; inner MSE={best.mse:,.0f}; outer MSE={metrics(y[held], predictions['tuned_ridge'][0])['mse']:,.0f}", flush=True)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["scheme", "seed", "model", "row_position"])
    oof.to_csv(output / "oof.csv", index=False)
    records, fold_records = [], []
    for (scheme, seed, model), part in oof.groupby(["scheme", "seed", "model"]):
        if len(part) != len(train) or part.train_row_id.nunique() != len(train):
            raise AssertionError("OOF coverage mismatch")
        records.append({"scheme": scheme, "seed": seed, "model": model, **metrics(part.y_true, part.y_pred)})
        for fold, sub in part.groupby("fold"):
            fold_records.append({"scheme": scheme, "seed": seed, "model": model, "fold": fold, **metrics(sub.y_true, sub.y_pred)})
    scores, fold_scores = pd.DataFrame(records), pd.DataFrame(fold_records)
    scores.to_csv(output / "scores.csv", index=False)
    fold_scores.to_csv(output / "fold_scores.csv", index=False)
    leaderboard = scores[scores.scheme == "structure"].groupby("model").mse.agg(structure_mse_mean="mean", structure_mse_std="std").reset_index()
    leaderboard = leaderboard.merge(scores[scores.scheme == "scaffold"][["model", "mse"]].rename(columns={"mse": "scaffold_mse"}), on="model").sort_values("structure_mse_mean")
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    by_name = leaderboard.set_index("model")
    # Predeclared deployment gate, not tuning on external fold labels.
    promote = bool(by_name.loc["tuned_ridge", "structure_mse_mean"] < by_name.loc["baseline_ridge", "structure_mse_mean"] and by_name.loc["tuned_ridge", "scaffold_mse"] <= by_name.loc["baseline_ridge", "scaffold_mse"])
    strategy = "tuned_ridge" if promote else "baseline_ridge"
    final_ranking, final_assignment = search_candidates(matrices, y, train_st.structure_id.to_numpy(), config, config["search"]["inner_seed"])
    final_ranking.to_csv(output / "final_search.csv", index=False)
    pd.DataFrame({"train_row_id": train.train_row_id, "group_id": train_st.structure_id, "fold": final_assignment}).to_csv(output / "final_search_folds.csv", index=False)
    pick = final_ranking.iloc[0]
    if promote:
        bundle = fit_candidate(matrices[pick.feature_set], y, specs[pick.feature_set], pick.alpha, config["clip_min"])
    else:
        bundle = fit_model("ridge_si", baseline_x, y, config)
    bundle.update({"manifest": manifest, "config": config, "strategy": strategy})
    joblib.dump(bundle, output / "model.joblib", compress=3)
    final_test_st, final_test_x = featurize(test.SMILES, bundle["fingerprint"])
    pred, raw = predict_model(bundle, final_test_x, final_test_st.valid)
    pd.DataFrame({"index": test["index"], TARGET: pred}).to_csv(output / "submission.csv", index=False)
    test_st.assign(prediction=pred, prediction_unclipped=raw, used_fallback=~test_st.valid).to_csv(output / "test_predictions.csv", index=False)
    checked = check_submission(output / "submission.csv", config["sample"])
    save_json(output / "submission_check.json", checked)
    selection = {"strategy": strategy, "promoted": promote, "final_model": bundle["name"], "fingerprint": bundle["fingerprint"],
        "candidate_from_full_train_cv": {"feature_set": pick.feature_set, "alpha": float(pick.alpha), "mse": float(pick.mse)},
        "rule": "Promote the predeclared full search only if structure MSE improves AND scaffold MSE does not worsen against frozen baseline. Per-feature ablations are diagnostic only.",
        "caveat": "Nested scores evaluate the search procedure. The promotion decision uses these scores; no untouched final test exists. Repeated seeds are not independent data.", "external_data_used": False}
    save_json(output / "selection.json", selection)
    oof[oof.model == strategy].to_csv(output / "oof_selected.csv", index=False)
    from .diagnostics import build_diagnostics
    build_diagnostics(output)
    save_json(output / "artifact_hashes.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    return {"output": str(output.resolve()), **selection, **checked}
