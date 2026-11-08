"""Train-only external augmentation, evaluated on the frozen organizer cohort."""
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .advanced import grid, fit_advanced, search
from .augmentation import eligible_external, load_approved_external
from .chemistry import featurize
from .data import TARGET, check_submission, save_json, sha256
from .diagnostics import build_diagnostics, table_md
from .models import predict_model
from .pipeline import audit
from .reliability import compare_oof
from .robust_experiment import benchmark_table, promotion_check
from .target_status import require_same_dataset
from .validation import make_folds, metrics


def augmented_run(settings, output):
    if sha256(settings['organizer_config']) != settings['organizer_config_sha256']:
        raise ValueError('Organizer configuration changed')
    config = json.loads(Path(settings['organizer_config']).read_text(encoding='utf-8'))
    reference = Path(settings['reference_run'])
    if sha256(reference/'artifact_hashes.json') != settings['reference_artifact_hashes_sha256']:
        raise ValueError('Reference manifest changed')
    for name, expected in json.loads((reference/'artifact_hashes.json').read_text()).items():
        if sha256(reference/name) != expected:
            raise ValueError(f'Reference artifact changed: {name}')
    external, approval = load_approved_external(settings['approved_external'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'augmentation_config.json', settings)
    # This audit is deliberately the unchanged organizer evaluation cohort.
    train, test, structures, test_st, _, _, manifest, audit_summary = audit(config, output)
    require_same_dataset(manifest, json.loads((reference/'manifest.json').read_text(encoding='utf-8')))
    pd.testing.assert_frame_equal(train, pd.read_csv(reference/'train_for_model.csv', dtype={'train_row_id':str}))
    config.update(augmentation=settings, training_dataset_version=settings['training_dataset_version'])
    save_json(output/'config.json', config)
    source = output/'source_snapshot'/'vacv'
    source.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, source/path.name)
    folds = make_folds(structures, config)
    folds['train_row_id'] = train.train_row_id.to_numpy()[folds.row_position.to_numpy()]
    keys = ['scheme','seed','row_position']
    old_folds = pd.read_csv(reference/'folds.csv', dtype={'train_row_id':str})
    pd.testing.assert_frame_equal(folds.sort_values(keys).reset_index(drop=True), old_folds.sort_values(keys).reset_index(drop=True))
    folds.to_csv(output/'folds.csv', index=False)
    specs = {'feature_sets':config['advanced_search']['feature_sets']}
    _, X = featurize(train.SMILES, specs)
    ext_st, ext_x = featurize(external.SMILES, specs)
    if not ext_st.valid.all():
        raise ValueError('Invalid approved external structure')
    ext_st.insert(0, 'train_row_id', external.train_row_id)
    # Test overlap is rejected, not used for transferring or calculating labels.
    if ext_st.structure_id.isin(test_st.structure_id).any():
        raise ValueError('External structures overlap the test set; separate review is required')
    external.to_csv(output/'external_for_model.csv', index=False)
    ext_st.to_csv(output/'external_structures.csv', index=False)
    save_json(output/'external_approval.json', approval)
    y, ext_y = train[TARGET].to_numpy(float), external[TARGET].to_numpy(float)
    candidates = {c['id']:c for c in grid(config)}
    save_json(output/'candidate_grid.json', list(candidates.values()))
    prior_selection = json.loads((reference/'selection.json').read_text(encoding='utf-8'))
    prior_key = prior_selection['strategy']
    old = pd.read_csv(reference/'oof.csv', dtype={'train_row_id':str})
    old = old[old.model.isin({'mean','baseline_ridge',prior_key})].copy()
    old['model'] = old.model.replace({prior_key:'incumbent_v4'})
    oof_parts, inner_rows, scores_inner, choices, eligibility = [old], [], [], [], []
    for (scheme, seed), assignment in folds.groupby(['scheme','seed'], sort=False):
        members = assignment.sort_values('row_position').fold.to_numpy()
        for fold in np.unique(members):
            held, fit = members == fold, members != fold
            keep = eligible_external(ext_st, structures.loc[held], scheme)
            outer = {'X':ext_x[keep], 'y':ext_y[keep], 'structures':ext_st.loc[keep].reset_index(drop=True)}
            ranking, inner = search(X[fit], y[fit], structures.structure_id.to_numpy()[fit],
                                    structures.scaffold_id.to_numpy()[fit], config,
                                    config['advanced_search']['inner_seed']+int(fold), augmentation=outer)
            scores_inner.append(ranking.assign(scheme=scheme, seed=int(seed), outer_fold=int(fold)))
            context = {'scheme':scheme, 'seed':int(seed), 'outer_fold':int(fold)}
            for index, record in ext_st.iterrows():
                eligibility.append({**context,'scope':'outer','inner_scheme':'','inner_fold':-1,
                                    'external_row_id':record.train_row_id,'included':bool(keep[index])})
            fit_st = structures.loc[fit].reset_index(drop=True)
            for inner_scheme, membership in inner.items():
                inner_rows.append(pd.DataFrame({**context,'inner_scheme':inner_scheme,
                    'row_position':np.flatnonzero(fit),'train_row_id':train.train_row_id.to_numpy()[fit],
                    'group_id':fit_st[inner_scheme+'_id'],'inner_fold':membership}))
                for inner_fold in np.unique(membership):
                    inner_keep = keep & eligible_external(ext_st, fit_st.loc[membership == inner_fold], inner_scheme)
                    for index, record in ext_st.iterrows():
                        eligibility.append({**context,'scope':'inner','inner_scheme':inner_scheme,'inner_fold':int(inner_fold),
                                            'external_row_id':record.train_row_id,'included':bool(inner_keep[index])})
            best = ranking.iloc[0]
            bundle = fit_advanced(np.concatenate([X[fit],ext_x[keep]]), np.concatenate([y[fit],ext_y[keep]]), config, candidates[best.candidate_id])
            pred, raw = predict_model(bundle, X[held], np.ones(held.sum(), bool))
            choices.append({**context,'strategy':'augmented_search','external_rows':int(keep.sum()),**best.to_dict()})
            oof_parts.append(pd.DataFrame({'train_row_id':train.train_row_id.to_numpy()[held], 'row_position':np.flatnonzero(held),
                'scheme':scheme,'seed':int(seed),'fold':int(fold),'model':'augmented_search','y_true':y[held],
                'y_pred':pred,'y_pred_unclipped':raw,'squared_error':(y[held]-pred)**2}))
            pd.DataFrame(choices).to_csv(output/'outer_choices.csv', index=False)
            print(f'{scheme} {seed} fold {fold}: {best.candidate_id}; external {int(keep.sum())}', flush=True)
    pd.concat(inner_rows).to_csv(output/'inner_folds.csv', index=False)
    pd.concat(scores_inner).to_csv(output/'inner_scores.csv', index=False)
    pd.DataFrame(eligibility).to_csv(output/'external_fold_eligibility.csv', index=False)
    oof = pd.concat(oof_parts).sort_values(['scheme','seed','model','row_position'])
    oof.to_csv(output/'oof.csv', index=False)
    rows, fold_scores = [], []
    for (scheme, seed, model), part in oof.groupby(['scheme','seed','model']):
        if len(part) != len(train) or part.train_row_id.nunique() != len(train):
            raise ValueError('Incomplete organizer-only evaluation')
        rows.append({'scheme':scheme,'seed':seed,'model':model,**metrics(part.y_true,part.y_pred)})
        for fold, sub in part.groupby('fold'):
            fold_scores.append({'scheme':scheme,'seed':seed,'model':model,'fold':fold,**metrics(sub.y_true,sub.y_pred)})
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'scores.csv', index=False)
    pd.DataFrame(fold_scores).to_csv(output/'fold_scores.csv', index=False)
    benchmark = benchmark_table(scores, config)
    benchmark.to_csv(output/'benchmark.csv', index=False)
    promoted, checks = promotion_check(benchmark, 'incumbent_v4', 'augmented_search')
    strategy = 'augmented_search' if promoted else 'incumbent_v4'
    historical = benchmark[benchmark.phase == 'historical']
    leaderboard = historical.pivot(index='model',columns='scheme',values='mse').rename(columns={'structure':'structure_mse_mean','scaffold':'scaffold_mse'}).reset_index()
    leaderboard.to_csv(output/'leaderboard.csv', index=False)
    extra = {'X':ext_x,'y':ext_y,'structures':ext_st}
    final_ranking, final_members = search(X,y,structures.structure_id,structures.scaffold_id,config,config['advanced_search']['inner_seed'],augmentation=extra)
    final_ranking.to_csv(output/'final_search.csv', index=False)
    pd.concat([pd.DataFrame({'train_row_id':train.train_row_id,'scheme':s,'fold':m}) for s,m in final_members.items()]).to_csv(output/'final_search_folds.csv', index=False)
    candidate = candidates[final_ranking.iloc[0].candidate_id]
    candidate_manifest = copy.deepcopy(manifest)
    candidate_manifest.update(external_data_used=True,training_dataset_version=settings['training_dataset_version'],
        external_training_rows=len(external),training_rows=len(train)+len(external),
        approved_external=settings['approved_external'],evaluation_population='unchanged organizer rows only')
    candidate_config = {**config,'external_data':True}
    candidate_bundle = fit_advanced(np.concatenate([X,ext_x]),np.concatenate([y,ext_y]),config,candidate)
    candidate_bundle.update(manifest=candidate_manifest,config=candidate_config,strategy='augmented_search')
    joblib.dump(candidate_bundle, output/'candidate_model.joblib', compress=3)
    if promoted:
        shutil.copy2(output/'candidate_model.joblib',output/'model.joblib')
        population = pd.concat([train,external],ignore_index=True)
    else:
        shutil.copy2(reference/'model.joblib',output/'model.joblib')
        population = train
    population.to_csv(output/'training_population.csv',index=False)
    bundle = joblib.load(output/'model.joblib')
    for name, chosen_bundle in [('submission.csv',bundle),('candidate_submission.csv',candidate_bundle)]:
        _, test_x = featurize(test.SMILES,chosen_bundle['fingerprint'])
        pred, raw = predict_model(chosen_bundle,test_x,test_st.valid)
        pd.DataFrame({'index':test['index'],TARGET:pred}).to_csv(output/name,index=False)
        if name == 'submission.csv':
            test_st.assign(prediction=pred,prediction_unclipped=raw,used_fallback=~test_st.valid).to_csv(output/'test_predictions.csv',index=False)
    checked = check_submission(output/'submission.csv',config['sample'])
    save_json(output/'submission_check.json',checked)
    selection = {'strategy':strategy,'promoted':promoted,'final_model':bundle['name'],'comparison_candidate':'augmented_search',
        'reference_strategy':'incumbent_v4','reference_label':'v4: только организаторы','candidate_label':'v5: +3 проверенные строки',
        'promotion_checks':checks,'rule':settings['promotion_rule'],'external_data_used':promoted,'external_rows_used':len(external) if promoted else 0,
        'model_labels':{'incumbent_v4':'v4: только организаторы','augmented_search':'v5: +3 проверенные строки','baseline_ridge':'Исходный Ridge','mean':'Среднее train'}}
    save_json(output/'selection.json',selection)
    config['external_data'] = bool(promoted)
    save_json(output/'config.json',config)
    manifest.update(augmentation_experiment=settings,external_data_used=bool(promoted),external_training_rows=len(external) if promoted else 0,
                    external_candidate_rows=len(external),evaluation_population='unchanged organizer rows only')
    save_json(output/'manifest.json',manifest)
    audit_summary.update(external_data_used=bool(promoted),external_candidate_rows=len(external),external_training_rows=len(external) if promoted else 0,
                         training_rows=len(population),evaluation_rows=len(train))
    save_json(output/'audit.json',audit_summary)
    oof[oof.model == strategy].to_csv(output/'oof_selected.csv',index=False)
    build_diagnostics(output,write_report=False)
    intervals = compare_oof(oof,structures,config,[('incumbent_v4','augmented_search')])
    intervals.to_csv(output/'reliability.csv',index=False)
    save_json(output/'reliability.json',{'rows':intervals.to_dict(orient='records'),'note':'Парный bootstrap групп организаторского train. Повторные ошибки усреднены по seed; интервалы условны на сохранённых прогнозах, без переобучения. Разбиения ранее использовались при разработке; независимого теста нет.'})
    save_json(output/'augmentation_summary.json',{'candidate_rows':len(external),'selected_model_rows':len(external) if promoted else 0,
        'organizer_rows':len(train),'training_rows':len(population),'external_scaffolds':int(ext_st.scaffold_id.nunique()),
        'train_structure_overlaps':int(ext_st.structure_id.isin(structures.structure_id).sum()),
        'train_scaffold_overlaps':int(ext_st.scaffold_id.isin(structures.scaffold_id).sum()),
        'test_structure_overlaps':int(ext_st.structure_id.isin(test_st.structure_id).sum()),
        'bound_rows_excluded':len(approval.get('excluded_bound_compounds',[])),
        'records':[{k:r[k] for k in ('record_id','compound','si','doi')} for r in approval['records']],
        'reference_label':selection['reference_label'],'candidate_label':selection['candidate_label'],
        'reconstructed_from_final_review_log':approval.get('reconstructed_from_final_review_log',False),
        'authorization_date':approval.get('authorization_date'),'promoted':promoted})
    comparison = intervals[intervals.metric == 'mse']
    report = f"""# Проверенные дополнения: v5

Три точные записи 8i, 8j, 8m восстановлены по итоговому журналу с явным подтверждением пользователя.
Две записи с границами 8k/8l исключены из точечной регрессии. Концентрации и метаданные
использованы только для проверки происхождения целевых SI; признаки получены только из SMILES.

Добавления прошли правило выпуска: **{promoted}**. Используется **{bundle['name']}**;
внешних строк в выпущенной модели: **{len(external) if promoted else 0}**.

{table_md(pd.DataFrame(checks), [('phase','Разбиения'),('scheme','Группы'),('baseline_mse','v4 MSE'),('candidate_mse','v5 MSE'),('improvement_percent','Снижение %'),('passed','Пройдено')])}

## Как сравнивались модели

Проверочная выборка — те же {len(train)} организаторских строк с неизменёнными метками.
Внешние записи добавлялись только в обучение. Во внешних и внутренних фолдах
исключались дополнения, совпадающие с отложенными структурами, а для каркасной
проверки — с отложенными каркасами. Все принадлежности сохранены. Та же сетка
из {len(candidates)} кандидатов выбирается по среднему двух внутренних групповых MSE.
В финальном обучении выполнена та же процедура; внешние строки не входят в её метрику.

{table_md(comparison,[('phase','Разбиения'),('scheme','Группы'),('improvement','Разница MSE'),('ci_low','2.5%'),('ci_high','97.5%')])}

Интервалы условны на сохранённых OOF, без переобучения и без учёта всей истории подбора.
Это повторно используемый train, а не независимый тест. Три дополнения представляют
один каркас и одну публикацию. Они не подтверждают обобщение на новые публикации.
Спорные 81 значения 10000 и 2 нуля организаторов сохранены: их смысл не разъяснён.
Версия с 83 нулями не подменяет обучение. MSE разных версий целей не сравнивался.
Диагностические графики покрытия относятся к организаторской части обучения.

`model.joblib` и `submission.csv` относятся к выбранной модели. `candidate_model.joblib`
и `candidate_submission.csv` сохраняют эксперимент с добавлениями даже при отказе в выпуске.
Отправки на Kaggle нет. Воспроизведение:

`python -m vacv augment --config configs/augmented.json --output artifacts/new-augmented-run`
"""
    (output/'report.md').write_text(report,encoding='utf-8')
    save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    return {'output':str(output.resolve()),**selection,**checked}
