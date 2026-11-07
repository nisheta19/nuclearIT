"""Read-only scientific diagnostics from saved OOF and original labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import featurize
from .data import TARGET, save_json
from .validation import metrics


def tanimoto(a, b):
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def table_md(frame, columns):
    lines = ["| " + " | ".join(title for _, title in columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        values = [f"{row[key]:,.3f}" if isinstance(row[key], (float, np.floating)) else str(row[key]) for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_diagnostics(output, write_report=True):
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    strategy = selection.get("strategy", selection.get("model"))
    oof = pd.read_csv(output / "oof.csv", dtype={"train_row_id": str})
    train = pd.read_csv(output / "train_for_model.csv", dtype={"train_row_id": str})
    test = pd.read_csv(output / "test_for_model.csv", dtype={"index": str})
    structures = pd.read_csv(output / "train_structures.csv", dtype={"train_row_id": str})
    chosen = oof[oof.model == strategy]
    primary_seed = config["structure_seeds"][0]
    primary = chosen[(chosen.scheme == "structure") & (chosen.seed == primary_seed)].sort_values("train_row_id").copy()
    # Similarity is defined once for audit, independent of selected representation.
    similarity_spec = {"radius": 2, "bits": 2048, "include_chirality": True}
    _, x = featurize(train.SMILES, similarity_spec)
    test_st, z = featurize(test.SMILES, similarity_spec)
    similarity = tanimoto(x, x)
    position = {row_id: i for i, row_id in enumerate(train.train_row_id)}
    nearest = np.empty(len(primary))
    neighbors = []
    for fold, part in primary.groupby("fold"):
        held = np.array([position[value] for value in part.train_row_id])
        fit = np.array([position[value] for value in primary.loc[primary.fold != fold, "train_row_id"]])
        for row_id, held_position in zip(part.train_row_id, held):
            order = fit[np.argsort(-similarity[held_position, fit], kind="stable")[:5]]
            nearest[primary.index.get_indexer([part.index[part.train_row_id == row_id][0]])[0]] = similarity[held_position, order[0]]
            for rank, neighbor in enumerate(order, start=1):
                neighbors.append({"train_row_id": row_id, "fold": int(fold), "rank": rank, "neighbor_row_id": train.iloc[neighbor].train_row_id,
                    "tanimoto": float(similarity[held_position, neighbor]), "neighbor_target": float(train.iloc[neighbor][TARGET]), "neighbor_smiles": train.iloc[neighbor].SMILES})
    primary["nearest_training_similarity"] = nearest
    primary["absolute_error"] = np.abs(primary.y_true - primary.y_pred)
    primary["share_total_squared_error"] = primary.squared_error / primary.squared_error.sum() if primary.squared_error.sum() else 0.0
    primary = primary.merge(train[["train_row_id", "SMILES"]], on="train_row_id")
    primary.sort_values("squared_error", ascending=False).to_csv(output / "error_review.csv", index=False)
    pd.DataFrame(neighbors).to_csv(output / "oof_neighbors.csv", index=False)
    all_slice_scores = []
    for (scheme, seed, model), part in oof.groupby(["scheme", "seed", "model"]):
        for name, mask in (("zero", part.y_true == 0), ("10000", part.y_true == 10000), ("other", (part.y_true != 0) & (part.y_true != 10000))):
            if mask.any():
                all_slice_scores.append({"scheme": scheme, "seed": seed, "model": model, "slice": name, **metrics(part.loc[mask, "y_true"], part.loc[mask, "y_pred"])})
    pd.DataFrame(all_slice_scores).to_csv(output / "slice_scores.csv", index=False)
    bins = pd.cut(primary.nearest_training_similarity, [-0.01, .3, .5, .7, .9, 1.01], labels=["<0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", ">0.9"], right=False)
    novelty_rows = [{"similarity_bin": str(name), **metrics(part.y_true, part.y_pred)} for name, part in primary.groupby(bins, observed=True)]
    pd.DataFrame(novelty_rows).to_csv(output / "similarity_scores.csv", index=False)
    target_bins = pd.cut(primary.y_true, [-1, 0, 10, 100, 1000, 9999.999, 10000, np.inf], labels=["0", "(0,10]", "(10,100]", "(100,1000]", "(1000,10000)", "10000", ">10000"])
    target_rows = [{"target_bin": str(name), **metrics(part.y_true, part.y_pred)} for name, part in primary.groupby(target_bins, observed=True)]
    pd.DataFrame(target_rows).to_csv(output / "target_range_scores.csv", index=False)
    test_similarity = np.full(len(test), np.nan)
    test_similarity[test_st.valid] = tanimoto(z[test_st.valid], x).max(axis=1)
    test.assign(nearest_training_similarity=test_similarity, valid_structure=test_st.valid).to_csv(output / "test_coverage.csv", index=False)
    joined = structures.merge(train[["train_row_id", TARGET]], on="train_row_id")
    duplicate_groups, duplicate_rows = [], []
    for key, part in joined.groupby("structure_id"):
        if len(part) < 2:
            continue
        labels = part[TARGET].to_numpy()
        status = "identical" if np.ptp(labels) == 0 else "rounding_scale_difference" if np.allclose(labels, labels[0], rtol=1e-4, atol=1e-3) else "conflicting_labels"
        duplicate_groups.append({"structure_id": key, "row_ids": ", ".join(part.train_row_id), "labels": ", ".join(f"{v:g}" for v in labels), "status": status, "canonical_smiles": part.canonical_smiles.iloc[0]})
        duplicate_rows.append(part.assign(review_status=status))
    groups = pd.DataFrame(duplicate_groups, columns=["structure_id", "row_ids", "labels", "status", "canonical_smiles"])
    groups.to_csv(output / "duplicate_review.csv", index=False)
    if duplicate_rows:
        pd.concat(duplicate_rows).to_csv(output / "duplicate_records.csv", index=False)
    extreme = primary.loc[primary.y_true.idxmax()]
    max_row = extreme.train_row_id
    # Only organizer metadata for identification; never added to feature matrices.
    metadata_columns = pd.read_csv(config["train"], nrows=0).columns
    allowed_metadata = [name for name in ("Title", "strain", "Culture") if name in metadata_columns]
    metadata = {}
    if allowed_metadata:
        source_metadata = pd.read_csv(config["train"], usecols=allowed_metadata, dtype=str, keep_default_na=False)
        metadata = source_metadata.iloc[position[max_row]].to_dict()
    review = {"row_id": str(max_row), "target": float(extreme.y_true), "primary_oof_prediction": float(extreme.y_pred),
        "primary_error_share": float(extreme.share_total_squared_error), "canonical_smiles": structures.loc[structures.train_row_id == max_row, "canonical_smiles"].iloc[0],
        "organizer_metadata": metadata, "medical_verification": "not performed; no primary publication supplied",
        "action": "Preserve label and row; inspect original experimental provenance when reviewed data arrive.",
        "all_outer_predictions": chosen[chosen.train_row_id == max_row][["scheme", "seed", "fold", "y_true", "y_pred"]].to_dict(orient="records")}
    save_json(output / "extreme_label_review.json", review)
    diagnostics = {"primary_seed": primary_seed, "strategy": strategy, "invalid_test": int((~test_st.valid).sum()),
        "test_similarity_median": float(np.nanmedian(test_similarity)), "test_below_0_3": int((test_similarity < .3).sum()), "test_below_0_5": int((test_similarity < .5).sum()),
        "duplicate_groups": len(groups), "conflicting_groups": int((groups.status == "conflicting_labels").sum()), "rounding_groups": int((groups.status == "rounding_scale_difference").sum()),
        "largest_target_review": review, "label_changes": 0, "external_data_used": False,
        "similarity_note": "Tanimoto on chiral Morgan radius=2/2048 binary bits. Descriptive coverage, not calibrated confidence. Bins are exploratory; do not select model or discard rows with them."}
    save_json(output / "diagnostics.json", diagnostics)
    make_plots(output, primary)
    if write_report:
        write_nested_report(output, config, selection, audit, diagnostics)


def make_plots(output, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#183e39", "text.color": "#183e39", "figure.facecolor": "#faf9f5", "axes.facecolor": "#faf9f5"})
    colors = np.where(primary.y_true == 10000, "#b17633", "#19756a")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    high = max(primary.y_true.max(), primary.y_pred.max()) * 1.04
    for axis in axes:
        axis.scatter(primary.y_true, primary.y_pred, c=colors, s=20, alpha=.7, edgecolor="none")
        axis.plot([0, high], [0, high], color="#9b9f96", ls="--", lw=1)
        axis.set(xlabel="Истинный SI", ylabel="Прогноз SI")
    axes[0].set_title("OOF: исходная шкала")
    axes[1].set(xscale="symlog", yscale="symlog", xlim=(0, high), ylim=(0, high))
    axes[1].set_title("Те же прогнозы: шкала symlog")
    fig.suptitle("Каждая точка предсказана без её структуры в обучении. Охра: метка 10000.", fontsize=10)
    for ext in ("png", "svg"):
        fig.savefig(charts / f"oof_scatter.{ext}", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    axes[0].scatter(primary.nearest_training_similarity, primary.absolute_error, c=colors, s=20, alpha=.65, edgecolor="none")
    axes[0].set(xlabel="Максимальное сходство с обучением своего фолда", ylabel="Абсолютная ошибка SI", title="Похожесть и ошибка")
    worst = primary.nlargest(10, "squared_error").sort_values("squared_error")
    axes[1].barh(worst.train_row_id.astype(str), worst.share_total_squared_error * 100, color="#19756a")
    axes[1].set(xlabel="Доля суммарной квадратичной ошибки, %", ylabel="train_row_id", title="Десять крупнейших ошибок")
    for ext in ("png", "svg"):
        fig.savefig(charts / f"error_diagnostics.{ext}", dpi=150)
    plt.close(fig)


def write_nested_report(output, config, selection, audit, diagnostics):
    output = Path(output)
    leaderboard = pd.read_csv(output / "leaderboard.csv")
    slices = pd.read_csv(output / "slice_scores.csv")
    selected = selection.get("strategy", selection.get("model"))
    slice_table = slices[(slices.scheme == "structure") & (slices.model == selected)].groupby("slice", as_index=False).agg(n=("n", "first"), mse=("mse", "mean"), mae=("mae", "mean"))
    review = diagnostics["largest_target_review"]
    text = f"""# VACV SI: вложенная групповая валидация

Версия данных: `{config['dataset_version']}`. Источник: `{config['train']}`.
Train: {audit['train_rows']}; test: {audit['test_rows']}; нулей: {audit['zero_targets']}; SI=10000: {audit['targets_10000']}.
Ни одна метка не изменена. Внешние данные не использовались. SHA-256 проверен перед загрузкой.

## Эксперимент

Внешние группы: одинаковые канонические структуры; seed {config['structure_seeds']}, до {config['folds']} фолдов.
Отдельная внешняя проверка по каркасам. Внутри каждого внешнего обучения — {config['search']['inner_folds']}-fold GroupKFold;
в проверке по каркасам внутренние группы тоже каркасные. Внешние строки не участвуют в выборе признаков/alpha или масштабировании.
Alpha: {config['search']['alphas']}. Представления: {', '.join(spec['name'] for spec in config['search']['feature_sets'])}.
Цель обучения и оценки — исходный SI. Порог снизу: {config['clip_min']}; верхнего clipping нет.
Все модели получают только SMILES. Дескрипторы масштабируются внутри каждого обучающего фолда.
Полный перебор внутри фолдов сохранён в inner_scores.csv, состав — в inner_folds.csv.

## Результаты

{table_md(leaderboard, [('model','Стратегия'),('structure_mse_mean','MSE структуры, среднее'),('structure_mse_std','SD между seed'),('scaffold_mse','MSE каркасы')])}

`tuned_ridge` — вся заранее объявленная процедура поиска. Остальные `tuned_*` — диагностические абляции с alpha,
выбранной внутри фолда; они не участвуют в отдельном выборе победителя по внешним результатам.
Выпуск: **{selected}**; финальная модель: **{selection.get('final_model', selected)}**.
Правило выпуска: {selection.get('rule', '')}

Вложенная оценка относится к процедуре поиска, а не одному набору параметров финальной модели.
Решение заменить baseline принято по внешним результатам. Отдельного нетронутого финального теста нет.
Три seed используют один train: SD не является доверительным интервалом. Преимущество baseline уже было известно
из предыдущего этапа, поэтому ни его повторная оценка, ни новый план не объявляются полностью независимым подтверждением.

## Ошибки и метки

{table_md(slice_table, [('slice','Срез'),('n','Строк'),('mse','Средний MSE'),('mae','Средняя MAE')])}

Разбор максимальной метки: row_id={review['row_id']}, SI={review['target']:g}; прогноз на первом внешнем seed
{review['primary_oof_prediction']:g}; доля ошибки {review['primary_error_share']:.1%}.
Организаторские метаданные: {json.dumps(review['organizer_metadata'], ensure_ascii=False)}.
SMILES разобран технически; соответствие первичной публикации не подтверждено. Автоматического исправления или удаления нет.
Соседи взяты только из обучения соответствующего фолда, их метки не переносятся в прогноз.
Разбор сохранён в extreme_label_review.json, error_review.csv и oof_neighbors.csv.

Групп дубликатов {diagnostics['duplicate_groups']}; существенные расхождения {diagnostics['conflicting_groups']};
различия масштаба округления {diagnostics['rounding_groups']} (rtol=1e-4, atol=0.001 — техническая диагностика).
Это не медицинский вердикт. Все наблюдения сохранены; duplicate_review.csv показывает строки и значения.

## Область применимости

Медиана максимального сходства test с train: {diagnostics['test_similarity_median']:.3f}.
Ниже 0.3: {diagnostics['test_below_0_3']} строк; ниже 0.5: {diagnostics['test_below_0_5']} строк.
Использован Tanimoto на Morgan radius=2, 2048 binary bits, со стереохимией.
Сходство не является вероятностью правильного ответа. Границы описательные, не пороги исключения.
Графики: charts/oof_scatter.png и charts/error_diagnostics.png (также SVG).
similarity_scores.csv и target_range_scores.csv относятся к первому seed {diagnostics['primary_seed']}.

## Воспроизведение

Команда: `python -m vacv tune --config configs/nested.json --output artifacts/new-nested-run`.
Финальная модель — model.joblib; submission.csv проверен относительно sample.
Параметры финальной модели выбраны дополнительной групповой CV на всём train, а её метрики
не выдаются за новую внешнюю оценку. Финальная модель обучена на всех строках разрешённого train.
Копия исходного кода этого запуска сохранена в source_snapshot/vacv.
Ничего не публиковалось и не отправлялось на Kaggle.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
