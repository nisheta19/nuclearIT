"""Reproducible nested evaluation of similarity models against frozen v4."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .advanced import fit_advanced, grid as old_grid, search as old_search
from .chemistry import featurize
from .data import TARGET, check_submission, save_json, sha256
from .diagnostics import build_diagnostics, table_md
from .models import predict_model
from .pipeline import audit
from .reliability import compare_oof
from .robust_experiment import benchmark_table, promotion_check
from .similarity_search import (SimilarityBlend, candidate_grid, choose_blend,
                                features_and_kernels, inner_search, model_for,
                                old_inner_predictions)
from .target_status import require_same_dataset
from .validation import make_folds, metrics


def select_joint(y, old, predictions, weights):
    ranked = pd.concat([choose_blend(y,old,new,weights).assign(candidate_id=key)
                        for key,new in predictions.items()],ignore_index=True)
    # Numerically equivalent models prefer the released procedure.
    ranked['_order'] = ranked.objective.round(6)
    return ranked.sort_values(['_order','new_weight','candidate_id']).drop(columns='_order').reset_index(drop=True)


def similarity_run(config, output):
    reference, output = Path(config['reference_run']), Path(output)
    if sha256(reference/'artifact_hashes.json') != config['reference_artifact_hashes_sha256']:
        raise ValueError('Reference manifest changed')
    for name, expected in json.loads((reference/'artifact_hashes.json').read_text()).items():
        if sha256(reference/name) != expected:
            raise ValueError(f'Reference artifact changed: {name}')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'config.json',config)
    train,test,structures,test_st,_,_,manifest,_ = audit(config,output)
    require_same_dataset(manifest,json.loads((reference/'manifest.json').read_text(encoding='utf-8')))
    pd.testing.assert_frame_equal(train,pd.read_csv(reference/'train_for_model.csv',dtype={'train_row_id':str}))
    source=output/'source_snapshot'/'vacv'; source.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source/path.name)
    old_config=json.loads((reference/'config.json').read_text())
    old_specs=old_config['advanced_search']['feature_sets']
    old_candidates={c['id']:c for c in old_grid(old_config)}
    candidates={c['id']:c for c in candidate_grid(config)}
    save_json(output/'candidate_grid.json',list(candidates.values()))
    _,old_x=featurize(train.SMILES,{'feature_sets':old_specs})
    matrices,kernels=features_and_kernels(train.SMILES,config)
    y=train[TARGET].to_numpy(float)
    folds=make_folds(structures,config)
    folds['train_row_id']=train.train_row_id.to_numpy()[folds.row_position]
    keys=['scheme','seed','row_position']
    pd.testing.assert_frame_equal(folds.sort_values(keys).reset_index(drop=True),pd.read_csv(reference/'folds.csv',dtype={'train_row_id':str}).sort_values(keys).reset_index(drop=True))
    fresh_config={**config,'structure_seeds':config['fresh_seeds']['structure'],'scaffold_seeds':config['fresh_seeds']['scaffold']}
    fresh=make_folds(structures,fresh_config)
    fresh['train_row_id']=train.train_row_id.to_numpy()[fresh.row_position]
    if set(zip(folds.scheme,folds.seed)) & set(zip(fresh.scheme,fresh.seed)):
        raise ValueError('Fresh seeds must differ from previous seeds')
    folds=pd.concat([folds,fresh],ignore_index=True)
    folds.to_csv(output/'folds.csv',index=False)
    previous=pd.read_csv(reference/'oof.csv',dtype={'train_row_id':str})
    previous=previous[previous.model.isin(['advanced_search','baseline_ridge','mean'])].copy()
    previous['model']=previous.model.replace({'advanced_search':'incumbent_v4'})
    old_scores=pd.read_csv(reference/'inner_scores.csv')
    old_inner=pd.read_csv(reference/'inner_folds.csv',dtype={'train_row_id':str})
    parts,choices,rankings,blend_rows,inner_rows=[previous],[],[],[],[]
    for (scheme,seed),assignment in folds.groupby(['scheme','seed'],sort=False):
        members=assignment.sort_values('row_position').fold.to_numpy()
        for fold in np.unique(members):
            fit,held=members!=fold,members==fold
            positions=np.flatnonzero(fit)
            inner_seed=config['similarity_search']['inner_seed']+int(fold)
            ranking,predictions,membership=inner_search([x[fit] for x in matrices],
                {key:K[np.ix_(fit,fit)] for key,K in kernels.items()},y[fit],structures.loc[fit],config,inner_seed)
            rankings.append(ranking.assign(scheme=scheme,seed=int(seed),outer_fold=int(fold)))
            prior=old_scores[(old_scores.scheme==scheme)&(old_scores.seed==seed)&(old_scores.outer_fold==fold)]
            cached=not prior.empty
            if not cached:
                prior,old_members=old_search(old_x[fit],y[fit],structures.structure_id.to_numpy()[fit],structures.scaffold_id.to_numpy()[fit],old_config,inner_seed)
                for key in membership:
                    np.testing.assert_array_equal(old_members[key],membership[key])
            for inner_scheme,member in membership.items():
                if cached:
                    saved=old_inner[(old_inner.scheme==scheme)&(old_inner.seed==seed)&(old_inner.outer_fold==fold)&(old_inner.inner_scheme==inner_scheme)].sort_values('row_position')
                    np.testing.assert_array_equal(saved.row_position,positions)
                    np.testing.assert_array_equal(saved.inner_fold,member)
                inner_rows.append(pd.DataFrame({'scheme':scheme,'seed':int(seed),'outer_fold':int(fold),'inner_scheme':inner_scheme,
                    'row_position':positions,'train_row_id':train.train_row_id.to_numpy()[fit],
                    'group_id':structures[inner_scheme+'_id'].to_numpy()[fit],'inner_fold':member}))
            old_best=prior.sort_values(['objective','candidate_id']).iloc[0]
            old_candidate=old_candidates[old_best.candidate_id]
            old_pred=old_inner_predictions(old_x[fit],y[fit],old_candidate,membership,old_config)
            for inner_scheme in membership:
                np.testing.assert_allclose(np.mean((y[fit]-old_pred[inner_scheme])**2),old_best[inner_scheme+'_mse'],rtol=1e-9,atol=1e-5)
            blends=select_joint(y[fit],old_pred,predictions,config['similarity_search']['blend_weights'])
            blend_rows.append(blends.assign(scheme=scheme,seed=int(seed),outer_fold=int(fold)))
            best=blends.iloc[0]; candidate=candidates[best.candidate_id]
            old_model=fit_advanced(old_x[fit],y[fit],old_config,old_candidate)['estimator']
            before=np.maximum(old_model.predict(old_x[held]),0)
            rep,power=candidate['representation'],candidate['power']
            new=model_for(candidate)._fit(matrices[rep][fit],y[fit],kernels[(rep,power)][np.ix_(fit,fit)])
            after=new.predict_kernel(kernels[(rep,power)][np.ix_(held,fit)],matrices[rep][held])
            prediction=(1-best.new_weight)*before+best.new_weight*after
            if cached:
                prior_pred=previous[(previous.model=='incumbent_v4')&(previous.scheme==scheme)&(previous.seed==seed)&(previous.fold==fold)].sort_values('row_position')
                np.testing.assert_allclose(prior_pred.y_pred,before,rtol=1e-9,atol=1e-7)
            for strategy,pred in [('similarity_search',prediction)]+([] if cached else [('incumbent_v4',before)]):
                parts.append(pd.DataFrame({'train_row_id':train.train_row_id.to_numpy()[held],'row_position':np.flatnonzero(held),
                    'scheme':scheme,'seed':int(seed),'fold':int(fold),'model':strategy,'y_true':y[held],'y_pred':pred,'y_pred_unclipped':pred,'squared_error':(y[held]-pred)**2}))
            choices.append({'scheme':scheme,'seed':int(seed),'outer_fold':int(fold),'old_candidate':old_candidate['id'],**best.to_dict()})
            pd.concat(parts).to_csv(output/'oof.csv',index=False)
            pd.DataFrame(choices).to_csv(output/'outer_choices.csv',index=False)
            print(f'{scheme} {seed} fold {fold}: new weight={best.new_weight:g} {candidate["id"]}',flush=True)
    oof=pd.concat(parts).sort_values(['scheme','seed','model','row_position'])
    oof.to_csv(output/'oof.csv',index=False)
    pd.concat(rankings).to_csv(output/'inner_scores.csv',index=False)
    pd.concat(blend_rows).to_csv(output/'inner_blends.csv',index=False)
    pd.concat(inner_rows).to_csv(output/'inner_folds.csv',index=False)
    scores=[]
    for (scheme,seed,model),part in oof.groupby(['scheme','seed','model']):
        if len(part)!=len(train) or part.train_row_id.nunique()!=len(train):
            raise ValueError('Incomplete OOF predictions')
        scores.append({'scheme':scheme,'seed':seed,'model':model,**metrics(part.y_true,part.y_pred)})
    scores=pd.DataFrame(scores); scores.to_csv(output/'scores.csv',index=False)
    is_fresh=[int(row.seed) in config['fresh_seeds'][row.scheme] for row in scores.itertuples()]
    frozen=benchmark_table(scores.loc[np.logical_not(is_fresh)],config)
    extra=scores.loc[is_fresh].groupby(['scheme','model'],as_index=False).agg(mse=('mse','mean'),seed_sd=('mse','std'),seed_count=('seed','size')).assign(phase='fresh')
    benchmark=pd.concat([frozen,extra]); benchmark.to_csv(output/'benchmark.csv',index=False)
    promoted,checks=promotion_check(frozen,'incumbent_v4','similarity_search')
    for scheme in ('structure','scaffold'):
        sub=extra[extra.scheme==scheme].set_index('model')
        before,after=float(sub.loc['incumbent_v4','mse']),float(sub.loc['similarity_search','mse'])
        checks.append({'phase':'fresh','scheme':scheme,'baseline_mse':before,'candidate_mse':after,'improvement_percent':(1-after/before)*100,'passed':after<before})
    promoted=all(check['passed'] for check in checks)
    ranking,predictions,membership=inner_search(matrices,kernels,y,structures,config,config['similarity_search']['inner_seed'])
    ranking.to_csv(output/'final_search.csv',index=False)
    old_ranking=pd.read_csv(reference/'final_search.csv').sort_values(['objective','candidate_id'])
    old_candidate=old_candidates[old_ranking.iloc[0].candidate_id]
    old_pred=old_inner_predictions(old_x,y,old_candidate,membership,old_config)
    blends=select_joint(y,old_pred,predictions,config['similarity_search']['blend_weights'])
    blends.to_csv(output/'final_blends.csv',index=False)
    best=blends.iloc[0]; candidate=candidates[best.candidate_id]
    specs=old_specs+config['similarity_search']['feature_sets']
    bounds=np.cumsum([0]+[s['bits']+len(s.get('descriptors',[])) for s in specs])
    old_model=fit_advanced(old_x,y,old_config,old_candidate)['estimator']
    rep=candidate['representation']; idx=len(old_specs)+rep
    new_model=model_for(candidate)._fit(matrices[rep],y,kernels[(rep,candidate['power'])])
    estimator=SimilarityBlend(old_model,new_model,old_x.shape[1],int(bounds[idx]),int(bounds[idx+1]),float(best.new_weight))
    candidate_bundle={'name':f'similarity_blend:w{best.new_weight:g}:{candidate["id"]}','estimator':estimator,
        'fingerprint':{'feature_sets':specs},'log_target':False,'fallback_mean':float(y.mean()),'clip_min':0.,
        'strategy':'similarity_search','manifest':manifest,'config':config,'candidate':candidate}
    joblib.dump(candidate_bundle,output/'candidate_model.joblib',compress=3)
    _,test_x=featurize(test.SMILES,candidate_bundle['fingerprint'])
    candidate_pred,_=predict_model(candidate_bundle,test_x,test_st.valid)
    pd.DataFrame({'index':test['index'],TARGET:candidate_pred}).to_csv(output/'candidate_submission.csv',index=False)
    check_submission(output/'candidate_submission.csv',config['sample'])
    strategy='similarity_search' if promoted else 'incumbent_v4'
    if promoted:
        shutil.copy2(output/'candidate_model.joblib',output/'model.joblib')
        shutil.copy2(output/'candidate_submission.csv',output/'submission.csv')
        bundle=candidate_bundle
    else:
        shutil.copy2(reference/'model.joblib',output/'model.joblib')
        shutil.copy2(reference/'submission.csv',output/'submission.csv')
        bundle=joblib.load(reference/'model.joblib')
    checked=check_submission(output/'submission.csv',config['sample']); save_json(output/'submission_check.json',checked)
    selected_pred=pd.read_csv(output/'submission.csv')[TARGET].to_numpy()
    test_st.assign(prediction=selected_pred,prediction_unclipped=selected_pred,used_fallback=~test_st.valid).to_csv(output/'test_predictions.csv',index=False)
    selection={'strategy':strategy,'promoted':promoted,'final_model':bundle['name'],'fingerprint':bundle['fingerprint'],
        'candidate_from_full_train_cv':candidate,'candidate_new_weight':float(best.new_weight),'promotion_checks':checks,'rule':config['promotion_rule'],
        'comparison_candidate':'similarity_search','reference_strategy':'incumbent_v4','reference_label':'Выпущенная версия v4/v5',
        'model_labels':{'incumbent_v4':'Выпущенная v4/v5','similarity_search':'Сходство и ансамбль v6','baseline_ridge':'Исходный Ridge','mean':'Среднее train'},'external_data_used':False}
    save_json(output/'selection.json',selection)
    historical=frozen[frozen.phase=='historical']
    leaderboard=historical.pivot(index='model',columns='scheme',values='mse').rename(columns={'structure':'structure_mse_mean','scaffold':'scaffold_mse'}).reset_index()
    leaderboard['structure_mse_std']=leaderboard.model.map(historical[historical.scheme=='structure'].set_index('model').seed_sd)
    leaderboard.to_csv(output/'leaderboard.csv',index=False)
    oof[oof.model==strategy].to_csv(output/'oof_selected.csv',index=False)
    frozen_oof=oof[[int(r.seed) not in config['fresh_seeds'][r.scheme] for r in oof.itertuples()]]
    intervals=compare_oof(frozen_oof,structures,config,[('incumbent_v4','similarity_search')])
    intervals.to_csv(output/'reliability.csv',index=False)
    save_json(output/'reliability.json',{'rows':intervals.to_dict(orient='records'),'note':'Conditional paired group bootstrap; repeated seeds averaged per row. Not an independent test; selection uncertainty excluded.'})
    build_diagnostics(output,write_report=False)
    report=f'''# v6: similarity models and conservative blending

Promoted: **{promoted}**. Selected: **{bundle['name']}**.
Original 320 labels unchanged; SMILES only; no external training rows.
Kaggle score reported by user: **10,913,537.18910**, for v5 submission.
Rank and public/private status are unverified. No Kaggle submission made by this run.

{table_md(pd.DataFrame(checks),[('phase','Cohort'),('scheme','Grouping'),('baseline_mse','v4 MSE'),('candidate_mse','v6 MSE'),('improvement_percent','Reduction %'),('passed','Passed')])}

The candidate grid uses seven structural representations, powered Tanimoto kernels,
linear or similarity-based logistic gates and kernel regression. Count fingerprints
use a pointwise log1p transform. All labels remain on the original SI scale.
Each outer training set selects the v4 candidate using its original inner selection;
the new candidate and blend weight minimize equal-weight structure/scaffold inner MSE.
Every landmark and fitted model uses only inner training rows. Original v4 inner
scores are reused only after hash, membership and recomputed-loss checks.

The original 45 outer folds have been reused during development. The 20 fresh folds
use different seeds, but the same molecules; they measure split robustness, not
independent generalization. Preliminary full-train grouped screens (98 then 140
candidates) informed development before nested evaluation. Selection uncertainty
is not covered by the conditional bootstrap. The hidden Kaggle score may worsen.

Reproduce: `python -m vacv similarity --config configs/similarity.json --output artifacts/new-similarity-run`
`submission.csv` is the selected release; `candidate_submission.csv` is the experiment.
'''
    (output/'report.md').write_text(report,encoding='utf-8')
    save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    return {'output':str(output.resolve()),**selection,**checked}
