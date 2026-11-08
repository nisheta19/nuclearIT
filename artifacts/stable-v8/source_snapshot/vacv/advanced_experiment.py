"""Nested evaluation against the released v3 procedure, with frozen comparison folds."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .advanced import grid, fit_advanced, search
from .chemistry import featurize
from .data import TARGET, save_json, sha256, check_submission
from .diagnostics import build_diagnostics, table_md
from .models import predict_model
from .pipeline import audit
from .reliability import compare_oof
from .robust_experiment import benchmark_table, promotion_check
from .validation import make_folds, metrics
from .target_status import require_same_dataset


def advanced_run(config, output):
    reference = Path(config['reference_run'])
    if sha256(reference/'artifact_hashes.json') != config['reference_artifact_hashes_sha256']:
        raise ValueError('Reference manifest changed')
    for name, expected in json.loads((reference/'artifact_hashes.json').read_text()).items():
        if sha256(reference/name) != expected:
            raise ValueError(f'Reference artifact changed: {name}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'config.json', config)
    train, test, structures, test_st, _, _, manifest, _ = audit(config,output)
    require_same_dataset(manifest, json.loads((reference/'manifest.json').read_text(encoding='utf-8')))
    source = output/'source_snapshot'/'vacv'
    source.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source/path.name)
    folds = make_folds(structures,config)
    folds['train_row_id'] = train.train_row_id.to_numpy()[folds.row_position.to_numpy()]
    folds.to_csv(output/'folds.csv',index=False)
    old_folds = pd.read_csv(reference/'folds.csv',dtype={'train_row_id':str})
    keys = ['scheme','seed','row_position']
    pd.testing.assert_frame_equal(folds.sort_values(keys).reset_index(drop=True),old_folds.sort_values(keys).reset_index(drop=True))
    old = pd.read_csv(reference/'oof.csv',dtype={'train_row_id':str})
    old['model'] = old.model.replace({'robust_search':'incumbent_v3'})
    old_train = pd.read_csv(reference/'train_for_model.csv',dtype={'train_row_id':str})
    pd.testing.assert_frame_equal(train,old_train)
    specs = config['advanced_search']['feature_sets']
    _, X = featurize(train.SMILES,{'feature_sets':specs})
    y = train[TARGET].to_numpy(dtype=float)
    candidates = {c['id']:c for c in grid(config)}
    save_json(output/'candidate_grid.json',list(candidates.values()))
    oof_parts, inner_scores, inner_rows, choices = [old], [], [], []
    for (scheme,seed), assignment in folds.groupby(['scheme','seed'],sort=False):
        members = assignment.sort_values('row_position').fold.to_numpy()
        for fold in np.unique(members):
            held, fit = members == fold, members != fold
            ranking, inner = search(X[fit],y[fit],structures.structure_id.to_numpy()[fit],structures.scaffold_id.to_numpy()[fit],config,config['advanced_search']['inner_seed']+int(fold))
            inner_scores.append(ranking.assign(scheme=scheme,seed=int(seed),outer_fold=int(fold)))
            for inner_scheme, membership in inner.items():
                inner_rows.append(pd.DataFrame({'scheme':scheme,'seed':int(seed),'outer_fold':int(fold),'inner_scheme':inner_scheme,
                    'row_position':np.flatnonzero(fit),'train_row_id':train.train_row_id.to_numpy()[fit],
                    'group_id':structures[inner_scheme+'_id'].to_numpy()[fit],'inner_fold':membership}))
            best_by_strategy = {'advanced_search':ranking.iloc[0], 'ensemble_only':ranking[ranking.family=='ensemble'].iloc[0],
                                'two_stage_only':ranking[ranking.family=='two_stage'].iloc[0]}
            fitted = {}
            for strategy, best in best_by_strategy.items():
                candidate = candidates[best.candidate_id]
                if best.candidate_id not in fitted:
                    fitted[best.candidate_id] = fit_advanced(X[fit],y[fit],config,candidate)
                pred, raw = predict_model(fitted[best.candidate_id],X[held],np.ones(held.sum(),bool))
                choices.append({'scheme':scheme,'seed':int(seed),'outer_fold':int(fold),'strategy':strategy,**best.to_dict()})
                oof_parts.append(pd.DataFrame({'train_row_id':train.train_row_id.to_numpy()[held],'row_position':np.flatnonzero(held),
                    'scheme':scheme,'seed':int(seed),'fold':int(fold),'model':strategy,'y_true':y[held],'y_pred':pred,'y_pred_unclipped':raw,'squared_error':(y[held]-pred)**2}))
            pd.concat(inner_scores).to_csv(output/'inner_scores.csv',index=False)
            pd.concat(inner_rows).to_csv(output/'inner_folds.csv',index=False)
            pd.DataFrame(choices).to_csv(output/'outer_choices.csv',index=False)
            print(f'{scheme} {seed} fold {fold}: {ranking.iloc[0].candidate_id}',flush=True)
    oof = pd.concat(oof_parts).sort_values(['scheme','seed','model','row_position'])
    oof.to_csv(output/'oof.csv',index=False)
    records, fold_scores = [], []
    for (scheme,seed,model), part in oof.groupby(['scheme','seed','model']):
        assert len(part)==len(train)==part.train_row_id.nunique()
        records.append({'scheme':scheme,'seed':seed,'model':model,**metrics(part.y_true,part.y_pred)})
        for fold, sub in part.groupby('fold'):
            fold_scores.append({'scheme':scheme,'seed':seed,'model':model,'fold':fold,**metrics(sub.y_true,sub.y_pred)})
    scores = pd.DataFrame(records)
    scores.to_csv(output/'scores.csv',index=False)
    pd.DataFrame(fold_scores).to_csv(output/'fold_scores.csv',index=False)
    benchmark = benchmark_table(scores,config)
    benchmark.to_csv(output/'benchmark.csv',index=False)
    promoted, checks = promotion_check(benchmark,'incumbent_v3','advanced_search')
    strategy = 'advanced_search' if promoted else 'incumbent_v3'
    historical = benchmark[benchmark.phase=='historical']
    leaderboard = historical.pivot(index='model',columns='scheme',values='mse').rename(columns={'structure':'structure_mse_mean','scaffold':'scaffold_mse'}).reset_index()
    leaderboard['structure_mse_std'] = leaderboard.model.map(historical[historical.scheme=='structure'].set_index('model').seed_sd)
    leaderboard.sort_values('structure_mse_mean').to_csv(output/'leaderboard.csv',index=False)
    final_ranking, final_members = search(X,y,structures.structure_id,structures.scaffold_id,config,config['advanced_search']['inner_seed'])
    final_ranking.to_csv(output/'final_search.csv',index=False)
    pd.concat([pd.DataFrame({'train_row_id':train.train_row_id,'scheme':scheme,'fold':member}) for scheme,member in final_members.items()]).to_csv(output/'final_search_folds.csv',index=False)
    final_candidate = candidates[final_ranking.iloc[0].candidate_id]
    candidate_bundle = fit_advanced(X,y,config,final_candidate)
    candidate_bundle.update(manifest=manifest,config=config,strategy='advanced_search')
    joblib.dump(candidate_bundle,output/'candidate_model.joblib',compress=3)
    bundle = candidate_bundle if promoted else joblib.load(reference/'model.joblib')
    bundle.update(manifest=manifest,config=config,strategy=strategy)
    joblib.dump(bundle,output/'model.joblib',compress=3)
    _, test_x = featurize(test.SMILES,bundle['fingerprint'])
    pred,raw = predict_model(bundle,test_x,test_st.valid)
    pd.DataFrame({'index':test['index'],TARGET:pred}).to_csv(output/'submission.csv',index=False)
    test_st.assign(prediction=pred,prediction_unclipped=raw,used_fallback=~test_st.valid).to_csv(output/'test_predictions.csv',index=False)
    checked = check_submission(output/'submission.csv',config['sample'])
    save_json(output/'submission_check.json',checked)
    selection = {'strategy':strategy,'promoted':promoted,'final_model':bundle['name'],'fingerprint':bundle['fingerprint'],
        'candidate_from_full_train_cv':final_candidate,'promotion_checks':checks,'rule':config['promotion_rule'],
        'comparison_candidate':'advanced_search','reference_strategy':'incumbent_v3','reference_label':'Выпущенная версия v3',
        'model_labels':{'incumbent_v3':'Выпущенная версия v3','baseline_ridge':'Исходный Ridge','advanced_search':'Ансамбли и двухэтапная модель',
                        'ensemble_only':'Отдельно: ансамбли','two_stage_only':'Отдельно: двухэтапная модель','mean':'Среднее train'},'external_data_used':False}
    save_json(output/'selection.json',selection)
    oof[oof.model==strategy].to_csv(output/'oof_selected.csv',index=False)
    build_diagnostics(output,write_report=False)
    intervals = compare_oof(oof,structures,config,[('baseline_ridge','incumbent_v3'),('incumbent_v3','advanced_search')])
    intervals.to_csv(output/'reliability.csv',index=False)
    save_json(output/'reliability.json',{'rows':intervals.to_dict(orient='records'),'note':'Парный bootstrap целых групп; ошибки одной строки усреднены по seed. Интервалы условны на сохранённых OOF-прогнозах: модели не переобучаются, неопределённость подбора не учтена. Это не независимый тест и не вероятность правильности отдельного прогноза.'})
    write_report(output,selection,leaderboard,intervals)
    save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    return {'output':str(output.resolve()),**selection,**checked}


def write_report(output,selection,leaderboard,intervals):
    comparison = intervals[(intervals.reference=='incumbent_v3') & (intervals.metric=='mse')]
    text = f"""# Итерация 4: ансамбли, двухэтапная модель и устойчивость

Выпущенная модель: **{selection['final_model']}**. Замена v3: **{selection['promoted']}**.
Все исходные метки сохранены. Только SMILES; внешние данные не использованы.

## Прежние разбиения

{table_md(leaderboard,[('model','Процедура'),('structure_mse_mean','MSE структуры'),('scaffold_mse','MSE каркасы')])}

## Условия выпуска относительно v3

{table_md(pd.DataFrame(selection['promotion_checks']),[('phase','Разбиения'),('scheme','Группы'),('baseline_mse','v3 MSE'),('candidate_mse','v4 MSE'),('improvement_percent','Снижение %'),('passed','Пройдено')])}

Предварительное правило: улучшение обеих метрик в обоих наборах разбиений. Диагностические
семейства ensemble_only и two_stage_only не выбираются по внешним оценкам. В каждом train
подбор по среднему двух внутренних групповых MSE; та же процедура используется в финальном обучении.
Веса ансамбля выбираются из фиксированной сетки; каждая компонента получает своё структурное представление.
Двухэтапная модель оценивает P(SI>=10000) и две условные регрессии. Порог — статистическая
гипотеза, а не медицинская интерпретация метки 10000. Истинная группа доступна только при fit.
Все строки участвуют в общей оценке MSE; цели не исправляются и не ограничиваются сверху.

## Разброс изменения MSE

Положительная разница означает улучшение относительно v3.

{table_md(comparison,[('phase','Разбиения'),('scheme','Группы'),('improvement','Разница MSE'),('ci_low','2.5%'),('ci_high','97.5%')])}

Парный bootstrap пересэмплирует целые группы. Повторы одной строки по seed усредняются до
пересэмплирования. Это условные интервалы для сохранённых прогнозов, без переобучения моделей;
они не учитывают весь процесс выбора и не заменяют независимый тест. Все разбиения уже использованы
при разработке. Если интервал содержит ноль, устойчивое преимущество этими данными не установлено.
MAE и все интервалы: reliability.csv. Все seed: scores.csv. Срезы целей: slice_scores.csv,
target_range_scores.csv. Состав внешних/внутренних групп: folds.csv, inner_folds.csv.

## Воспроизведение

`python -m vacv advance --config configs/advanced.json --output artifacts/new-advanced-run`

candidate_model.joblib сохраняет модель нового поиска даже при отказе в выпуске.
model.joblib и submission.csv содержат только выбранную для использования модель.
Сохранены конфигурация, сетка кандидатов, снимок кода и SHA-256 артефактов. Отправки на Kaggle не было.
"""
    (output/'report.md').write_text(text,encoding='utf-8')
