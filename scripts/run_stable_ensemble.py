"""Train/evaluate v7, then pair its folds with the independently verified v6 baseline."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from vacv.chemistry import featurize
from vacv.data import TARGET,save_json,sha256,check_submission
from vacv.diagnostics import build_diagnostics,table_md
from vacv.models import predict_model
from vacv.pipeline import audit
from vacv.reliability import compare_oof
from vacv.robust_experiment import benchmark_table,promotion_check
from vacv.stable import StageEnsemble,stable_grid,stable_search
from vacv.target_status import require_same_dataset
from vacv.validation import make_folds,metrics


def run(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_json(output/'config.json',config)
    train,test,structures,test_st,_,_,manifest,_=audit(config,output)
    reference=Path(config['reference_run'])
    assert sha256(reference/'artifact_hashes.json')==config['reference_artifact_hashes_sha256']
    for name,expected in json.loads((reference/'artifact_hashes.json').read_text()).items():
        assert sha256(reference/name)==expected,name
    require_same_dataset(manifest,json.loads((reference/'manifest.json').read_text(encoding='utf-8')))
    pd.testing.assert_frame_equal(train,pd.read_csv(reference/'train_for_model.csv',dtype={'train_row_id':str}))
    source=output/'source_snapshot'/'vacv';source.mkdir(parents=True)
    for path in Path('vacv').glob('*.py'):shutil.copy2(path,source/path.name)
    shutil.copy2(__file__,output/'source_snapshot'/'run_stable_ensemble.py')
    specs=config['stable_search']['feature_sets'];widths=[s['bits'] for s in specs]
    _,X=featurize(train.SMILES,{'feature_sets':specs});y=train[TARGET].to_numpy(float)
    all_config={**config,'structure_seeds':config['structure_seeds']+config['fresh_seeds']['structure'],
                'scaffold_seeds':config['scaffold_seeds']+config['fresh_seeds']['scaffold']}
    folds=make_folds(structures,all_config);folds['train_row_id']=train.train_row_id.to_numpy()[folds.row_position]
    folds.to_csv(output/'folds.csv',index=False)
    candidates={c['id']:c for c in stable_grid(config)};save_json(output/'candidate_grid.json',list(candidates.values()))
    parts,choices,ranks,inner_rows=[],[],[],[]
    for (scheme,seed),part in folds.groupby(['scheme','seed'],sort=False):
        members=part.sort_values('row_position').fold.to_numpy()
        for fold in np.unique(members):
            fit,held=members!=fold,members==fold
            ranking,inner=stable_search(X[fit],y[fit],structures.loc[fit],config,config['stable_search']['inner_seed']+int(fold))
            ranks.append(ranking.assign(scheme=scheme,seed=int(seed),outer_fold=int(fold)))
            for key,member in inner.items():
                inner_rows.append(pd.DataFrame({'scheme':scheme,'seed':int(seed),'outer_fold':int(fold),'inner_scheme':key,
                    'row_position':np.flatnonzero(fit),'train_row_id':train.train_row_id.to_numpy()[fit],
                    'group_id':structures[key+'_id'].to_numpy()[fit],'inner_fold':member}))
            best=ranking.iloc[0];c=candidates[best.candidate_id]
            model=StageEnsemble(widths,C=c['C'],weights=c['weights']).fit(X[fit],y[fit]);pred=model.predict(X[held])
            parts.append(pd.DataFrame({'scheme':scheme,'seed':int(seed),'fold':int(fold),'model':'stable_search','row_position':np.flatnonzero(held),
                'train_row_id':train.train_row_id.to_numpy()[held],'y_true':y[held],'y_pred':pred,'y_pred_unclipped':pred,'squared_error':(y[held]-pred)**2}))
            choices.append({'scheme':scheme,'seed':int(seed),'outer_fold':int(fold),**best.to_dict()})
            pd.concat(parts).to_csv(output/'candidate_oof.csv',index=False)
            pd.DataFrame(choices).to_csv(output/'outer_choices.csv',index=False)
            print(f'{scheme} {seed} fold {fold}: {c["id"]}',flush=True)
    pd.concat(ranks).to_csv(output/'inner_scores.csv',index=False);pd.concat(inner_rows).to_csv(output/'inner_folds.csv',index=False)
    # The sibling experiment recomputes v4 on the exact fresh folds. Its new
    # candidate is never used here. Wait for its immutable manifest, then verify.
    baseline=Path(config['fresh_baseline_run'])
    deadline=time.monotonic()+1800
    while not (baseline/'artifact_hashes.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Fresh v4 baseline is incomplete')
        time.sleep(1)
    baseline_hashes=json.loads((baseline/'artifact_hashes.json').read_text())
    for name,expected in baseline_hashes.items():assert sha256(baseline/name)==expected,name
    save_json(output/'baseline_provenance.json',{'run':str(baseline),'artifact_hashes_sha256':sha256(baseline/'artifact_hashes.json'),
        'used_model':'incumbent_v4','candidate_model_used':False})
    require_same_dataset(manifest,json.loads((baseline/'manifest.json').read_text(encoding='utf-8')))
    key=['scheme','seed','row_position']
    pd.testing.assert_frame_equal(folds.sort_values(key).reset_index(drop=True),pd.read_csv(baseline/'folds.csv',dtype={'train_row_id':str}).sort_values(key).reset_index(drop=True))
    before=pd.read_csv(baseline/'oof.csv',dtype={'train_row_id':str});before=before[before.model.isin(['incumbent_v4','baseline_ridge','mean'])]
    oof=pd.concat([before,*parts]).sort_values(['scheme','seed','model','row_position']);oof.to_csv(output/'oof.csv',index=False)
    scores=pd.DataFrame([{'scheme':scheme,'seed':seed,'model':model,**metrics(p.y_true,p.y_pred)} for (scheme,seed,model),p in oof.groupby(['scheme','seed','model'])])
    scores.to_csv(output/'scores.csv',index=False)
    fresh=[int(r.seed) in config['fresh_seeds'][r.scheme] for r in scores.itertuples()]
    frozen=benchmark_table(scores.loc[np.logical_not(fresh)],config)
    extra=scores.loc[fresh].groupby(['scheme','model'],as_index=False).agg(mse=('mse','mean'),seed_sd=('mse','std'),seed_count=('seed','size')).assign(phase='fresh')
    pd.concat([frozen,extra]).to_csv(output/'benchmark.csv',index=False)
    _,checks=promotion_check(frozen,'incumbent_v4','stable_search')
    for scheme in ('structure','scaffold'):
        sub=extra[extra.scheme==scheme].set_index('model');a=float(sub.loc['incumbent_v4','mse']);b=float(sub.loc['stable_search','mse'])
        checks.append({'phase':'fresh','scheme':scheme,'baseline_mse':a,'candidate_mse':b,'improvement_percent':(1-b/a)*100,'passed':b<a})
    promoted=all(c['passed'] for c in checks);strategy='stable_search' if promoted else 'incumbent_v4'
    ranking,membership=stable_search(X,y,structures,config,config['stable_search']['inner_seed']);ranking.to_csv(output/'final_search.csv',index=False)
    pd.concat([pd.DataFrame({'scheme':key,'train_row_id':train.train_row_id,'fold':member}) for key,member in membership.items()]).to_csv(output/'final_search_folds.csv',index=False)
    candidate=candidates[ranking.iloc[0].candidate_id]
    model=StageEnsemble(widths,C=candidate['C'],weights=candidate['weights']).fit(X,y)
    bundle={'name':candidate['id'],'estimator':model,'fingerprint':{'feature_sets':specs},'fallback_mean':float(y.mean()),'log_target':False,'clip_min':0.,
        'manifest':manifest,'config':config,'strategy':'stable_search','candidate':candidate}
    joblib.dump(bundle,output/'candidate_model.joblib',compress=3)
    _,test_x=featurize(test.SMILES,bundle['fingerprint']);pred,_=predict_model(bundle,test_x,test_st.valid)
    pd.DataFrame({'index':test['index'],TARGET:pred}).to_csv(output/'candidate_submission.csv',index=False)
    check_submission(output/'candidate_submission.csv',config['sample'])
    if promoted:
        shutil.copy2(output/'candidate_model.joblib',output/'model.joblib');shutil.copy2(output/'candidate_submission.csv',output/'submission.csv')
    else:
        shutil.copy2(reference/'model.joblib',output/'model.joblib');shutil.copy2(reference/'submission.csv',output/'submission.csv');bundle=joblib.load(reference/'model.joblib')
    checked=check_submission(output/'submission.csv',config['sample']);save_json(output/'submission_check.json',checked)
    pred=pd.read_csv(output/'submission.csv')[TARGET]
    test_st.assign(prediction=pred,prediction_unclipped=pred,used_fallback=~test_st.valid).to_csv(output/'test_predictions.csv',index=False)
    release=config.get('release_label','v7')
    selection={'strategy':strategy,'promoted':promoted,'final_model':bundle['name'],'fingerprint':bundle['fingerprint'],'candidate_from_full_train_cv':candidate,
        'promotion_checks':checks,'rule':config['promotion_rule'],'comparison_candidate':'stable_search','reference_strategy':'incumbent_v4','reference_label':'Выпущенная v4/v5',
        'model_labels':{'incumbent_v4':'Выпущенная v4/v5','stable_search':f'Усреднение двухэтапных моделей {release}','baseline_ridge':'Исходный Ridge','mean':'Среднее train'},'external_data_used':False}
    save_json(output/'selection.json',selection);oof[oof.model==strategy].to_csv(output/'oof_selected.csv',index=False)
    historical=frozen[frozen.phase=='historical'];leaderboard=historical.pivot(index='model',columns='scheme',values='mse').rename(columns={'structure':'structure_mse_mean','scaffold':'scaffold_mse'}).reset_index()
    leaderboard['structure_mse_std']=leaderboard.model.map(historical[historical.scheme=='structure'].set_index('model').seed_sd);leaderboard.to_csv(output/'leaderboard.csv',index=False)
    frozen_oof=oof[[int(r.seed) not in config['fresh_seeds'][r.scheme] for r in oof.itertuples()]]
    intervals=compare_oof(frozen_oof,structures,config,[('incumbent_v4','stable_search')]);intervals.to_csv(output/'reliability.csv',index=False)
    save_json(output/'reliability.json',{'rows':intervals.to_dict(orient='records'),'note':'Conditional paired group bootstrap; not independent test evidence; selection uncertainty excluded.'})
    build_diagnostics(output,write_report=False)
    settings=config['stable_search']
    development_note=config.get('development_note','The six-candidate grid was fixed before observing v7 OOF or fresh-fold scores, after v6 worsened additional structure MSE. Hyperparameters are selected using grouped inner validation in each outer train.')
    (output/'report.md').write_text(f'''# {release}: two-stage ensembles

Promoted: **{promoted}**. Selected model: **{bundle['name']}**.

{table_md(pd.DataFrame(checks),[('phase','Cohort'),('scheme','Grouping'),('baseline_mse','v4 MSE'),('candidate_mse',release+' MSE'),('improvement_percent','Reduction %'),('passed','Passed')])}

{len(candidates)} candidate(s): C = {settings['classifier_C']}; fixed regression alpha = 1;
weights = {settings['weights']}. Fingerprints: {[s['name'] for s in specs]}.
{development_note}
The final model uses the same procedure. No Kaggle score is used as a selection loss.
The 45 earlier folds have been reused during development; fresh seeds use the
same molecules, so neither cohort is an independent test.

All 320 original labels unchanged; only SMILES as input; no external rows.
Kaggle MSE 10,913,537.18910 is user-reported for v5; its rank and public/private
status are unknown. A lower local MSE does not guarantee a lower Kaggle MSE.

Reproduce: `python scripts/run_stable_ensemble.py --config {config.get('config_path','configs/stable.json')} --output artifacts/new-stable-run`
The v4 fresh baseline is taken from the verified similarity-v6 run; its candidate
is never used. `submission.csv` is the selected model, `candidate_submission.csv`
is experimental if promotion failed.
''',encoding='utf-8')
    save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stable.json');p.add_argument('--output',required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text(encoding='utf-8')),a.output)
