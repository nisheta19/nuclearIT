from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .chemistry import featurize
from .data import TARGET, check_submission, save_json, sha256
from .diagnostics import build_diagnostics, table_md
from .kernel import fit_kernel_candidate, tanimoto_kernel
from .models import fit_model, predict_model
from .pipeline import audit
from .robust_search import candidate_grid, robust_inner_search
from .validation import make_folds, metrics


def benchmark_table(scores, config):
    frame = scores.copy()
    frame["phase"] = ["historical" if int(row.seed) in config["historical_seeds"][row.scheme] else "additional" for row in frame.itertuples()]
    return frame.groupby(["phase", "scheme", "model"], as_index=False).agg(mse=("mse", "mean"), seed_sd=("mse", lambda values: float(values.std(ddof=1)) if len(values) > 1 else 0.0), seed_count=("seed", "size"))


def promotion_check(benchmark, baseline_key="baseline_ridge", candidate_key="robust_search"):
    checks = []
    for phase in ("historical", "additional"):
        for scheme in ("structure", "scaffold"):
            part = benchmark[(benchmark.phase == phase) & (benchmark.scheme == scheme)].set_index("model")
            if not {baseline_key, candidate_key}.issubset(part.index):
                raise ValueError("Promotion requires both schemes and both historical/additional phases")
            before, after = float(part.loc[baseline_key, "mse"]), float(part.loc[candidate_key, "mse"])
            checks.append({"phase": phase, "scheme": scheme, "baseline_mse": before, "candidate_mse": after, "improvement_percent": (1 - after / before) * 100, "passed": after < before})
    return all(check["passed"] for check in checks), checks


def robust_run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    train, test, structures, test_structures, base_x, _, manifest, summary = audit(config, output)
    source = output / "source_snapshot" / "vacv"
    source.mkdir(parents=True)
    for path in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    folds = make_folds(structures, config)
    folds["train_row_id"] = train.train_row_id.to_numpy()[folds.row_position.to_numpy()]
    folds.to_csv(output / "folds.csv", index=False)
    specs = {spec["name"]: spec for spec in config["robust_search"]["feature_sets"]}
    candidates = {candidate["id"]: candidate for candidate in candidate_grid(config)}
    save_json(output / "candidate_grid.json", list(candidates.values()))
    matrices = {name: featurize(train.SMILES, spec)[1] for name, spec in specs.items()}
    # Pairwise features have no fitted statistics or target dependence.
    kernels = {name: tanimoto_kernel(x, x) for name, x in matrices.items()}
    y = train[TARGET].to_numpy(dtype=float)
    oof_parts, inner_scores, inner_rows, selections = [], [], [], []
    for (scheme, seed), assignment in folds.groupby(["scheme", "seed"], sort=False):
        memberships = assignment.sort_values("row_position").fold.to_numpy()
        for fold in sorted(set(memberships)):
            held, fit = memberships == fold, memberships != fold
            positions = np.flatnonzero(fit)
            ranking, inner = robust_inner_search({name: x[fit] for name, x in matrices.items()}, y[fit],
                structures.structure_id.to_numpy()[fit], structures.scaffold_id.to_numpy()[fit], config,
                config["robust_search"]["inner_seed"] + int(fold),
                kernels={name: matrix[np.ix_(fit, fit)] for name, matrix in kernels.items()})
            best = ranking.iloc[0]
            candidate = candidates[best.candidate_id]
            name = candidate["feature_set"]
            inner_scores.append(ranking.assign(scheme=scheme, seed=int(seed), outer_fold=int(fold)))
            for inner_scheme, membership in inner.items():
                grouping = structures["structure_id" if inner_scheme == "structure" else "scaffold_id"].to_numpy()
                inner_rows.append(pd.DataFrame({"scheme": scheme, "seed": int(seed), "outer_fold": int(fold), "inner_scheme": inner_scheme,
                    "row_position": positions, "train_row_id": train.train_row_id.to_numpy()[fit], "group_id": grouping[fit], "inner_fold": membership}))
            selections.append({"scheme": scheme, "seed": int(seed), "outer_fold": int(fold), **best.to_dict()})
            chosen = fit_kernel_candidate(matrices[name][fit], y[fit], specs[name], candidate, config["clip_min"], kernels[name][np.ix_(fit, fit)])
            baseline = fit_model("ridge_si", base_x[fit], y[fit], config)
            predictions = {"robust_search": predict_model(chosen, matrices[name][held], np.ones(held.sum(), dtype=bool)),
                "baseline_ridge": predict_model(baseline, base_x[held], np.ones(held.sum(), dtype=bool)),
                "mean": (np.full(held.sum(), y[fit].mean()), np.full(held.sum(), y[fit].mean()))}
            for model, (pred, raw) in predictions.items():
                oof_parts.append(pd.DataFrame({"train_row_id": train.train_row_id.to_numpy()[held], "row_position": np.flatnonzero(held), "scheme": scheme, "seed": int(seed), "fold": int(fold), "model": model, "y_true": y[held], "y_pred": pred, "y_pred_unclipped": raw, "squared_error": (y[held] - pred) ** 2}))
            pd.concat(inner_scores).to_csv(output / "inner_scores.csv", index=False)
            pd.concat(inner_rows).to_csv(output / "inner_folds.csv", index=False)
            pd.DataFrame(selections).to_csv(output / "outer_choices.csv", index=False)
            print(f"{scheme} {seed} fold {fold}: {candidate['id']} | inner {best.objective:,.0f} | outer {metrics(y[held], predictions['robust_search'][0])['mse']:,.0f}", flush=True)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["scheme", "seed", "model", "row_position"])
    oof.to_csv(output / "oof.csv", index=False)
    rows, fold_records = [], []
    for (scheme, seed, model), part in oof.groupby(["scheme", "seed", "model"]):
        if len(part) != len(train) or part.train_row_id.nunique() != len(train):
            raise AssertionError("Missing or duplicate OOF row")
        rows.append({"scheme": scheme, "seed": seed, "model": model, **metrics(part.y_true, part.y_pred)})
        for fold, sub in part.groupby("fold"):
            fold_records.append({"scheme": scheme, "seed": seed, "model": model, "fold": fold, **metrics(sub.y_true, sub.y_pred)})
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / "fold_scores.csv", index=False)
    benchmark = benchmark_table(scores, config)
    benchmark.to_csv(output / "benchmark.csv", index=False)
    promoted, checks = promotion_check(benchmark)
    strategy = "robust_search" if promoted else "baseline_ridge"
    # Main table is explicitly the old, like-for-like splits; additional splits are separate.
    historical = scores[[int(row.seed) in config["historical_seeds"][row.scheme] for row in scores.itertuples()]]
    leaderboard = historical[historical.scheme == "structure"].groupby("model").mse.agg(structure_mse_mean="mean", structure_mse_std="std").reset_index()
    leaderboard = leaderboard.merge(historical[historical.scheme == "scaffold"][["model", "mse"]].rename(columns={"mse": "scaffold_mse"}), on="model").sort_values("structure_mse_mean")
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    final_ranking, final_assignment = robust_inner_search(matrices, y, structures.structure_id, structures.scaffold_id, config, config["robust_search"]["inner_seed"], kernels=kernels)
    final_ranking.to_csv(output / "final_search.csv", index=False)
    pd.concat([pd.DataFrame({"train_row_id": train.train_row_id, "scheme": scheme, "fold": member}) for scheme, member in final_assignment.items()]).to_csv(output / "final_search_folds.csv", index=False)
    final_candidate = candidates[final_ranking.iloc[0].candidate_id]
    if promoted:
        name = final_candidate["feature_set"]
        bundle = fit_kernel_candidate(matrices[name], y, specs[name], final_candidate, config["clip_min"], kernels[name])
    else:
        bundle = fit_model("ridge_si", base_x, y, config)
    bundle.update({"manifest": manifest, "config": config, "strategy": strategy})
    joblib.dump(bundle, output / "model.joblib", compress=3)
    _, features = featurize(test.SMILES, bundle["fingerprint"])
    prediction, raw = predict_model(bundle, features, test_structures.valid)
    pd.DataFrame({"index": test["index"], TARGET: prediction}).to_csv(output / "submission.csv", index=False)
    test_structures.assign(prediction=prediction, prediction_unclipped=raw, used_fallback=~test_structures.valid).to_csv(output / "test_predictions.csv", index=False)
    checked = check_submission(output / "submission.csv", config["sample"])
    save_json(output / "submission_check.json", checked)
    selection = {"strategy": strategy, "promoted": promoted, "final_model": bundle["name"], "fingerprint": bundle["fingerprint"], "candidate_from_full_train_cv": final_candidate,
        "promotion_checks": checks, "rule": config["promotion_rule"], "external_data_used": False,
        "caveat": "Adaptive development on one dataset; extra splits are stability checks, not independent test data. Outer results used for promotion, not inner candidate tuning.",
        "model_labels": {"baseline_ridge": "Исходный Ridge", "robust_search": "Подбор по двум видам групп", "mean": "Среднее train"},
        "comparison_candidate": "robust_search"}
    save_json(output / "selection.json", selection)
    oof[oof.model == strategy].to_csv(output / "oof_selected.csv", index=False)
    build_diagnostics(output, write_report=False)
    write_report(output, config, selection, leaderboard, benchmark)
    save_json(output / "artifact_hashes.json", {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"})
    return {"output": str(output.resolve()), **selection, **checked}


def write_report(output, config, selection, leaderboard, benchmark):
    checks = pd.DataFrame(selection["promotion_checks"])
    audit_summary = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    text = f"""# VACV SI: регрессия на сходстве и подбор по двум схемам

Версия: `{config['dataset_version']}`. Модель выпуска: **{selection['final_model']}**.
Замена baseline: **{selection['promoted']}**. Источник — {config['train']}; SHA-256 проверен.
{audit_summary['train_rows']} исходных наблюдений, нулей: {audit_summary['zero_targets']}, меток 10000: {audit_summary['targets_10000']}. Все метки сохранены. Внешние данные не использованы.

## Сопоставимые результаты

Ниже только ранее использованные разбиения: structure seed {config['historical_seeds']['structure']}, scaffold seed {config['historical_seeds']['scaffold']} (-1 означает без перемешивания).

{table_md(leaderboard, [('model','Процедура'),('structure_mse_mean','MSE структуры'),('structure_mse_std','SD между seed'),('scaffold_mse','MSE каркасы')])}

## Проверка выпуска

{table_md(checks, [('phase','Набор разбиений'),('scheme','Группы'),('baseline_mse','Baseline MSE'),('candidate_mse','Новый MSE'),('improvement_percent','Улучшение, %'),('passed','Пройдено')])}

Правило задано до запуска: строгое улучшение обоих видов MSE и на исторических, и на дополнительных разбиениях.
Дополнительные seed: structure {sorted(set(config['structure_seeds']) - set(config['historical_seeds']['structure']))},
scaffold {sorted(set(config['scaffold_seeds']) - set(config['historical_seeds']['scaffold']))}.
Все разбиения используют тот же train; это проверка устойчивости, не независимый тест.
Шифры групп и все внутренние/внешние фолды сохранены.

## Что изменилось

- Kernel Ridge на Tanimoto-сходстве; среднее цели вычисляется только по train соответствующего фолда.
- Morgan r2/r3 и Feature Morgan r2; признаки только из SMILES, со стереохимией.
- Фиксированная смесь 75% kernel / 25% linear проверяется как отдельный кандидат внутри train.
- Внутри каждого внешнего фолда одинаковая процедура: групповая CV по структурам и по каркасам,
  выбор по среднему двух MSE. Та же процедура применяется при окончательном обучении.
- Число кандидатов {len(candidate_grid(config))}, inner folds {config['robust_search']['inner_folds']}.
- Попарное сходство заранее вычисляется без целей и без fitted-преобразований. В регрессор передаются только обучающие строки.
- Никакого точного переноса меток соседей или исправления специальных значений нет.

Все прогнозы оцениваются в исходной шкале SI; только нижняя граница 0, без верхней обрезки.
Внутренние оценки служат выбору; показатели обучения на всём train не выдаются за внешнюю оценку.
План следующего этапа основан на предыдущих результатах: адаптивная разработка может давать оптимизм.
Решение о выпуске также опирается на внешние результаты, поэтому окончательная точность на Kaggle неизвестна.

## Диагностика

Максимальная метка: {diagnostics['largest_target_review']['target']:g}; прогноз на первом историческом seed:
{diagnostics['largest_target_review']['primary_oof_prediction']:g}.
Медицинская достоверность метки не установлена; запись сохранена.
Срезы 0 / 10000 / остальные: slice_scores.csv. Крупнейшие ошибки: error_review.csv.
Каркасные и структурные ошибки по фолдам: fold_scores.csv.
Графики: charts/oof_scatter.png, charts/error_diagnostics.png и SVG.

## Воспроизведение

`python -m vacv improve --config configs/kernel.json --output artifacts/new-kernel-run`

Сохранены config.json, candidate_grid.json, inner_folds.csv, inner_scores.csv, outer_choices.csv,
OOF, модель, submission на {audit_summary['test_rows']} строк, проверка формата и копия исходного кода source_snapshot/vacv.
Предсказание: `python -m vacv predict --model {output.as_posix()}/model.joblib --test {config['test']} --sample {config['sample']} --output artifacts/recheck/submission.csv`.
Ничего не отправлялось на Kaggle.

Технические источники: [KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html),
[Feature Morgan](https://rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html).
"""
    (output / "report.md").write_text(text, encoding="utf-8")
