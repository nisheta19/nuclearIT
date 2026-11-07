from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .chemistry import POLICY, featurize
from .data import TARGET, check_submission, load_inputs, save_json, sha256
from .models import fit_model, predict_model
from .validation import make_folds, metrics
from .target_status import target_status


def environment() -> dict:
    return {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "rdkit", "joblib")}}


def audit(config: dict, output: Path):
    train, test, manifest = load_inputs(config)
    train_structures, train_x = featurize(train.SMILES, config["fingerprint"])
    test_structures, test_x = featurize(test.SMILES, config["fingerprint"])
    train_structures.insert(0, "train_row_id", train.train_row_id)
    test_structures.insert(0, "index", test["index"])
    train_structures.to_csv(output / "train_structures.csv", index=False)
    test_structures.to_csv(output / "test_structures.csv", index=False)
    train.to_csv(output / "train_for_model.csv", index=False)
    test.to_csv(output / "test_for_model.csv", index=False)
    manifest.update({"environment": environment(), "structure_policy": POLICY})
    source_root = Path(__file__).resolve().parent.parent
    manifest["code_sha256"] = {str(path.relative_to(source_root)): sha256(path) for path in sorted((source_root / "vacv").glob("*.py"))}
    save_json(output / "manifest.json", manifest)
    y = train[TARGET].to_numpy(dtype=float)
    overlap = train_structures.loc[train_structures.valid, ["train_row_id", "structure_id", "canonical_smiles"]].merge(test_structures.loc[test_structures.valid, ["index", "structure_id"]], on="structure_id")
    overlap.to_csv(output / "train_test_overlap.csv", index=False)
    dupes = train_structures.assign(target=y).groupby("structure_id").agg(n=("target", "size"), distinct_labels=("target", "nunique"), min_target=("target", "min"), max_target=("target", "max"))
    dupes.loc[dupes.n > 1].to_csv(output / "duplicate_train_groups.csv")
    summary = {
        **manifest["observed"],
        "target_min": float(y.min()), "target_max": float(y.max()), "target_mean": float(y.mean()), "target_median": float(np.median(y)),
        "invalid_train": int((~train_structures.valid).sum()), "invalid_test": int((~test_structures.valid).sum()),
        "unique_train_structures": train_structures.loc[train_structures.valid, "structure_id"].nunique(),
        "unique_test_structures": test_structures.loc[test_structures.valid, "structure_id"].nunique(),
        "unique_train_scaffolds": train_structures.loc[train_structures.valid, "scaffold_id"].nunique(),
        "largest_scaffold_group": int(train_structures.scaffold_id.value_counts().max()),
        "test_rows_overlapping_train": overlap["index"].nunique(),
        "train_groups_with_conflicting_labels": int(((dupes.n > 1) & (dupes.distinct_labels > 1)).sum()),
        "target_changes": 0, "external_data_used": False,
    }
    save_json(output / "audit.json", summary)
    if not train_structures.valid.all():
        raise ValueError("Invalid train SMILES: see train_structures.csv; no rows removed or labels changed")
    return train, test, train_structures, test_structures, train_x, test_x, manifest, summary


def evaluate(config, train, structures, features, folds, output):
    y = train[TARGET].to_numpy(dtype=float)
    predictions, fold_scores, scores, slices = [], [], [], []
    for (scheme, seed), assignment in folds.groupby(["scheme", "seed"], sort=False):
        assignment = assignment.sort_values("row_position")
        fold_ids = assignment.fold.to_numpy()
        for name in config["candidates"]:
            pred, raw = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for fold in sorted(set(fold_ids)):
                valid = fold_ids == fold
                bundle = fit_model(name, features[~valid], y[~valid], config)
                pred[valid], raw[valid] = predict_model(bundle, features[valid], np.ones(valid.sum(), dtype=bool))
                fold_scores.append({"scheme": scheme, "seed": seed, "model": name, "fold": int(fold), **metrics(y[valid], pred[valid]), "unclipped_mse": metrics(y[valid], raw[valid])["mse"]})
            result = {"scheme": scheme, "seed": seed, "model": name, **metrics(y, pred), "unclipped_mse": metrics(y, raw)["mse"], "clipped_predictions": int((pred != raw).sum())}
            scores.append(result)
            for label, mask in (("zero", y == 0), ("10000", y == 10000), ("other", (y != 0) & (y != 10000))):
                if mask.any():
                    slices.append({"scheme": scheme, "seed": seed, "model": name, "slice": label, **metrics(y[mask], pred[mask])})
            predictions.append(pd.DataFrame({"train_row_id": train.train_row_id, "scheme": scheme, "seed": seed, "model": name, "fold": fold_ids, "y_true": y, "y_pred": pred, "y_pred_unclipped": raw, "squared_error": (y - pred) ** 2}))
            print(f"{scheme:9s} seed={seed:4d} {name:20s} MSE={result['mse']:,.3f}", flush=True)
    oof = pd.concat(predictions, ignore_index=True)
    score_table, fold_table = pd.DataFrame(scores), pd.DataFrame(fold_scores)
    stats = fold_table.groupby(["scheme", "seed", "model"]).mse.agg(fold_mse_mean="mean", fold_mse_std="std", fold_mse_min="min", fold_mse_max="max").reset_index()
    score_table = score_table.merge(stats, on=["scheme", "seed", "model"])
    oof.to_csv(output / "oof.csv", index=False)
    score_table.to_csv(output / "scores.csv", index=False)
    fold_table.to_csv(output / "fold_scores.csv", index=False)
    pd.DataFrame(slices).to_csv(output / "slice_scores.csv", index=False)
    return oof, score_table


def write_report(output, summary, leaderboard, selected, oof, scores, submission_check, config):
    base = float(leaderboard.loc[leaderboard.model == "mean", "structure_mse_mean"].iloc[0])
    best = leaderboard.loc[leaderboard.model == selected].iloc[0]
    primary_seed = int(oof.loc[oof.scheme == "structure", "seed"].iloc[0])
    primary = oof.loc[(oof.scheme == "structure") & (oof.seed == primary_seed) & (oof.model == selected)].copy()
    primary["share_total_squared_error"] = primary.squared_error / primary.squared_error.sum() if primary.squared_error.sum() else 0.0
    primary.sort_values("squared_error", ascending=False).head(20).to_csv(output / "largest_errors.csv", index=False)
    rows = [f"| Модель | MSE по структурам, среднее {len(config['structure_seeds'])} seed | SD между seed | MSE по каркасам |", "|---|---:|---:|---:|"]
    for row in leaderboard.itertuples(index=False):
        rows.append(f"| {row.model} | {row.structure_mse_mean:,.3f} | {row.structure_mse_std:,.3f} | {row.scaffold_mse:,.3f} |")
    report = f"""# Результат baseline VACV SI

Источник: **{config['train']}**. Версия: **{config['dataset_version']}**.
Нулевых меток: **{summary['zero_targets']}**; меток со значением 10000: **{summary['targets_10000']}**.
Проверки и метрики относятся только к этой версии исходных меток; чужие очереди проверки к ней не применяются.
Цели сохранены без изменения, внешние данные не использованы.

## Данные и защита от утечки

Train: {summary['train_rows']} строк; test: {summary['test_rows']} строк. Максимум SI: {summary['target_max']:,.3f}.
Уникальных канонических структур: train {summary['unique_train_structures']}, test {summary['unique_test_structures']}.
Строк test с пересечением по структуре: {summary['test_rows_overlapping_train']}; переноса меток по совпадениям нет.
Групп train с различающимися метками: {summary['train_groups_with_conflicting_labels']}; метки не усреднялись.
Неразбираемых SMILES: train {summary['invalid_train']}, test {summary['invalid_test']}.
Каркасов train: {summary['unique_train_scaffolds']}; максимальная группа: {summary['largest_scaffold_group']} строк.

Признаки: Morgan radius={config['fingerprint']['radius']}, {config['fingerprint']['bits']} бит,
includeChirality={config['fingerprint']['include_chirality']}. В модель поступает только SMILES.
Концентрации, их производные, признаки пропусков, индексы, названия и флаги аудита исключены.
Числа из sample не используются. Канонизация сохраняет соли, заряды, изотопы и таутомеры.
Atom map numbers удаляются как аннотации. Ациклические молекулы входят в одну scaffold-группу.

## Валидация и выбор

GroupKFold по каноническим структурам: заранее фиксированные seed {config['structure_seeds']}, до {config['folds']} фолдов.
Отдельно GroupKFold по каркасам без перемешивания, с балансировкой размеров групп.
Все кандидаты используют те же строки и разбиения. Фолды сохранены до обучения.
Если канонические структуры уникальны, проверка по структурам близка к случайному разбиению;
оценка по каркасам показывает перенос на другие семейства, но не гарантирует соответствие скрытому тесту.

{chr(10).join(rows)}

Выбран **{selected}** по минимальному среднему MSE по структурам.
Изменение MSE относительно среднего: {(1 - best.structure_mse_mean / base) * 100:.2f}% улучшения.
Каркасная проверка диагностическая и не использовалась для настройки параметров.
OOF применён для выбора кандидата, поэтому это оценка разработки, а не независимый итоговый тест.
Разные seed используют один train и не являются независимыми наборами. SD не является доверительным интервалом.
MSE считается в исходной шкале SI, в том числе для log1p-моделей после expm1.
Средний прогноз каждого фолда рассчитывается только по его обучающим строкам.
Нижняя граница прогноза: {config['clip_min']}; верхней обрезки нет. Unclipped MSE и число изменений сохранены для проверки.
Ни один fitted transform не обучается на отложенных строках; дополнительных обучаемых преобразований нет.

`scores.csv` содержит общий MSE и разброс по фолдам; общий MSE взвешен по числу строк.
`fold_scores.csv` и `slice_scores.csv` содержат детали по фолдам и срезам 0 / 10000 / остальные.
`largest_errors.csv` показывает 20 крупнейших ошибок выбранной модели на первом seed ({primary_seed}).
Размер среза нулевых меток: {summary['zero_targets']}; по малому срезу нельзя делать устойчивые выводы.
Высокие SI не удалялись. Неопределённый смысл меток 0 и 10000 остаётся ограничением версии данных.

## Артефакты

Модель дообучена на всём разрешённом train: `model.joblib`.
Submission: `submission.csv`, {submission_check['rows']} строк, проверены колонки, индексы, порядок и конечность.
Это проверка формата, не проверка качества на Kaggle. Прогнозы для одинаковых канонических структур совпадают.
Неподдерживаемые test SMILES получают среднее train и отмечаются в `test_predictions.csv`.
OOF всех моделей: `oof.csv`; выбранной: `oof_selected.csv`; разбиения: `folds.csv`.
Хеши входов, кода и версии окружения: `manifest.json`; настройки: `config.json`.
Публикация и отправка на Kaggle не выполнялись.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def run(config: dict, output: str | Path, audit_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    train, test, train_st, test_st, train_x, test_x, manifest, summary = audit(config, output)
    folds = make_folds(train_st, config)
    folds["train_row_id"] = train.train_row_id.to_numpy()[folds.row_position.to_numpy()]
    folds.to_csv(output / "folds.csv", index=False)
    fold_summary = folds.merge(train.assign(row_position=np.arange(len(train))), on=["row_position", "train_row_id"]).groupby(["scheme", "seed", "fold"]).agg(rows=(TARGET, "size"), groups=("group_id", "nunique"), target_mean=(TARGET, "mean"), target_max=(TARGET, "max"), zero_targets=(TARGET, lambda y: int((y == 0).sum())), targets_10000=(TARGET, lambda y: int((y == 10000).sum())))
    fold_summary.to_csv(output / "fold_summary.csv")
    if audit_only:
        return summary
    oof, scores = evaluate(config, train, train_st, train_x, folds, output)
    leaderboard = scores.loc[scores.scheme == "structure"].groupby("model").mse.agg(structure_mse_mean="mean", structure_mse_std="std").reset_index()
    scaffold = scores.loc[scores.scheme == "scaffold", ["model", "mse"]].rename(columns={"mse": "scaffold_mse"})
    leaderboard = leaderboard.merge(scaffold, on="model").sort_values(["structure_mse_mean", "model"]).reset_index(drop=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    selected = str(leaderboard.iloc[0].model)
    save_json(output / "selection.json", {"model": selected, "criterion": "minimum mean structure-grouped OOF MSE over predeclared seeds", "independent_test": False, "external_data_used": False})
    oof.loc[oof.model == selected].to_csv(output / "oof_selected.csv", index=False)
    bundle = fit_model(selected, train_x, train[TARGET].to_numpy(dtype=float), config)
    bundle.update({"manifest": manifest, "config": config})
    joblib.dump(bundle, output / "model.joblib", compress=3)
    pred, raw = predict_model(bundle, test_x, test_st.valid)
    prediction_table = test_st.assign(prediction=pred, prediction_unclipped=raw, used_fallback=~test_st.valid)
    prediction_table.to_csv(output / "test_predictions.csv", index=False)
    if prediction_table.loc[prediction_table.valid].groupby("structure_id").prediction.nunique().gt(1).any():
        raise AssertionError("Equivalent structures received different predictions")
    pd.DataFrame({"index": test["index"], TARGET: pred}).to_csv(output / "submission.csv", index=False)
    checked = check_submission(output / "submission.csv", config["sample"])
    save_json(output / "submission_check.json", checked)
    write_report(output, summary, leaderboard, selected, oof, scores, checked, config)
    save_json(output / "artifact_hashes.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    return {"selected_model": selected, "output": str(output.resolve()), **checked}


def predict(model_path: str, test_path: str, sample_path: str, output: str):
    bundle = joblib.load(model_path)
    if environment()["packages"] != bundle["manifest"]["environment"]["packages"]:
        raise ValueError("Model/library version mismatch: install requirements.lock.txt")
    test = pd.read_csv(test_path, usecols=["index", "SMILES"], dtype=str, keep_default_na=False)
    sample_ids = pd.read_csv(sample_path, usecols=["index"], dtype=str, keep_default_na=False)["index"]
    if test["index"].duplicated().any() or not test["index"].equals(sample_ids):
        raise ValueError("Test/sample IDs differ or are duplicated")
    structures, features = featurize(test.SMILES, bundle["fingerprint"])
    pred, _ = predict_model(bundle, features, structures.valid)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": test["index"], TARGET: pred}).to_csv(destination, index=False)
    structures.assign(index=test["index"], used_fallback=~structures.valid).to_csv(destination.with_suffix(".audit.csv"), index=False)
    result = check_submission(destination, sample_path)
    result['target_review'] = target_status(bundle['manifest'])
    save_json(destination.with_suffix('.metadata.json'), result)
    return result
