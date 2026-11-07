"""Post-run audit independent of the training orchestration; creates review artifacts."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vacv.data import TARGET, save_json, sha256
from vacv.pipeline import predict
from vacv.reliability import compare_oof


def verify(run_dir,output):
    run_dir,output=Path(run_dir),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    augmentation=config.get('augmentation')
    reference=Path(augmentation['reference_run'] if augmentation else config['reference_run'])
    selection=json.loads((run_dir/'selection.json').read_text(encoding='utf-8'))
    reference_key=selection['reference_strategy']
    candidate_key=selection['comparison_candidate']
    hashes=json.loads((run_dir/'artifact_hashes.json').read_text())
    for name,value in hashes.items():
        assert sha256(run_dir/name)==value,name
    for role,value in config['input_sha256'].items():
        assert sha256(config[role])==value,role
    train=pd.read_csv(run_dir/'train_for_model.csv',dtype={'train_row_id':str})
    raw=pd.read_csv(config['train'])
    np.testing.assert_array_equal(train[TARGET],raw[TARGET])
    np.testing.assert_array_equal(train.SMILES,raw.SMILES)
    assert set(train.columns)=={'train_row_id','SMILES',TARGET}
    structures=pd.read_csv(run_dir/'train_structures.csv',dtype={'train_row_id':str})
    folds=pd.read_csv(run_dir/'folds.csv',dtype={'train_row_id':str})
    old_folds=pd.read_csv(reference/'folds.csv',dtype={'train_row_id':str})
    pd.testing.assert_frame_equal(folds,old_folds)
    inner=pd.read_csv(run_dir/'inner_folds.csv',dtype={'train_row_id':str})
    count=0
    for (scheme,seed),part in folds.groupby(['scheme','seed']):
        assert len(part)==len(train)==part.train_row_id.nunique()
        linked=part.merge(structures[['train_row_id','structure_id','scaffold_id']],on='train_row_id',validate='one_to_one')
        assert linked.groupby('structure_id').fold.nunique().eq(1).all()
        assert linked.groupby(scheme+'_id').fold.nunique().eq(1).all()
        for fold in part.fold.unique():
            count+=1
            expected=set(part.loc[part.fold!=fold,'train_row_id'])
            for inner_scheme in ('structure','scaffold'):
                sub=inner[(inner.scheme==scheme)&(inner.seed==seed)&(inner.outer_fold==fold)&(inner.inner_scheme==inner_scheme)]
                assert len(sub)==len(expected)==sub.train_row_id.nunique() and set(sub.train_row_id)==expected
                assert sub.groupby('group_id').inner_fold.nunique().eq(1).all()
                np.testing.assert_array_equal(sub.group_id,structures.set_index('train_row_id').loc[sub.train_row_id,inner_scheme+'_id'])
    oof=pd.read_csv(run_dir/'oof.csv',dtype={'train_row_id':str})
    scores=pd.read_csv(run_dir/'scores.csv').set_index(['scheme','seed','model'])
    for key,part in oof.groupby(['scheme','seed','model']):
        assert len(part)==len(train)==part.train_row_id.nunique()
        np.testing.assert_array_equal(part.y_true,train.set_index('train_row_id').loc[part.train_row_id,TARGET])
        errors=part.y_true-part.y_pred
        np.testing.assert_allclose(part.squared_error,errors**2,rtol=1e-12)
        np.testing.assert_allclose(scores.loc[key,'mse'],np.mean(errors**2),rtol=1e-12)
        np.testing.assert_allclose(scores.loc[key,'mae'],np.mean(np.abs(errors)),rtol=1e-12)
    old=pd.read_csv(reference/'oof.csv',dtype={'train_row_id':str})
    keys=['scheme','seed','train_row_id']
    incumbent=oof[oof.model==reference_key].sort_values(keys)
    prior_key=json.loads((reference/'selection.json').read_text(encoding='utf-8'))['strategy']
    previous=old[old.model==prior_key].sort_values(keys)
    # CSV parse/write introduces sub-picounit roundoff; it must not change predictions materially.
    np.testing.assert_allclose(incumbent.y_pred,previous.y_pred,rtol=1e-14,atol=1e-10)
    ranking=pd.read_csv(run_dir/'inner_scores.csv')
    choices=pd.read_csv(run_dir/'outer_choices.csv')
    for choice in choices.itertuples():
        part=ranking[(ranking.scheme==choice.scheme)&(ranking.seed==choice.seed)&(ranking.outer_fold==choice.outer_fold)]
        if choice.strategy!=candidate_key:
            part=part[part.family==choice.strategy.removesuffix('_only')]
        best=part.sort_values(['objective','candidate_id']).iloc[0]
        assert choice.candidate_id==best.candidate_id
        np.testing.assert_allclose(part.objective,(part.structure_mse+part.scaffold_mse)/2,rtol=1e-12)
    comparisons=[(reference_key,candidate_key)] if augmentation else [('baseline_ridge',reference_key),(reference_key,candidate_key)]
    intervals=compare_oof(oof,structures,config,comparisons)
    pd.testing.assert_frame_equal(intervals,pd.read_csv(run_dir/'reliability.csv'),check_exact=False,rtol=1e-10)
    for check in selection['promotion_checks']:
        seed_values=config['historical_seeds'][check['scheme']]
        part=scores.reset_index()
        mask=part.seed.isin(seed_values)
        part=part[(part.scheme==check['scheme']) & (mask if check['phase']=='historical' else ~mask)]
        before=part.loc[part.model==reference_key,'mse'].mean()
        after=part.loc[part.model==candidate_key,'mse'].mean()
        np.testing.assert_allclose([before,after],[check['baseline_mse'],check['candidate_mse']],rtol=1e-12)
        assert check['passed']==(after<before)
    assert selection['promoted']==all(c['passed'] for c in selection['promotion_checks'])
    checked=predict(str(run_dir/'model.joblib'),config['test'],config['sample'],str(output/'submission.csv'))
    assert (output/'submission.csv').read_bytes()==(run_dir/'submission.csv').read_bytes()
    if not selection['promoted']:
        assert (output/'submission.csv').read_bytes()==(reference/'submission.csv').read_bytes()
    # Range comparisons average per-seed errors on paired original observations.
    paired=oof[oof.model==reference_key].merge(oof[oof.model==candidate_key],on=keys,suffixes=('_before','_after'),validate='one_to_one')
    paired['phase']=['historical' if int(r.seed) in config['historical_seeds'][r.scheme] else 'additional' for r in paired.itertuples()]
    paired['target_bin']=pd.cut(paired.y_true_before,[-1,0,10,100,1000,9999.999,10000,np.inf],labels=['0','(0,10]','(10,100]','(100,1000]','(1000,10000)','10000','>10000'])
    ranges=[]
    for (phase,scheme,target_bin),part in paired.groupby(['phase','scheme','target_bin'],observed=True):
        entry={'phase':phase,'scheme':scheme,'target_bin':str(target_bin),'n':part.train_row_id.nunique()}
        for suffix in ('before','after'):
            error=part['y_true_'+suffix]-part['y_pred_'+suffix]
            entry['mse_'+suffix]=float(np.mean(error**2))
            entry['mae_'+suffix]=float(np.mean(np.abs(error)))
        ranges.append(entry)
    pd.DataFrame(ranges).to_csv(output/'range_comparison.csv',index=False)
    medical=ROOT/'artifacts'/'medical-review-v4'
    queue=pd.read_csv(medical/'medical_review_queue.csv',dtype=str,keep_default_na=False)
    original=pd.read_csv(config['train'],dtype=str,keep_default_na=False)
    assert queue.review_status.eq('pending').all() and queue.verified_si.eq('').all()
    assert queue.train_row_id.nunique()==len(queue)
    for item in queue.itertuples():
        assert item.original_si_text==original.iloc[int(item.train_row_id)][TARGET]
        assert item.source_sha256==config['input_sha256']['train']
    result={'run':str(run_dir.resolve()),'artifacts_checked':len(hashes),'outer_folds':count,
            'identical_structures_grouped':True,'inner_folds_exclude_outer_validation':True,
            'reference_oof_unchanged_within_csv_precision':True,'labels_unchanged':True,'input_hashes_unchanged':True,
            'mse_mae_recomputed':True,'bootstrap_reproduced':True,'promotion_recomputed':True,
            'submission_byte_identical_after_reload':True,'medical_queue_rows':len(queue),'medical_conclusions_pending':True,
            'submission':checked}
    save_json(output/'verification.json',result)
    if augmentation:
        import joblib
        from vacv.augmentation import load_approved_external
        external,approval=load_approved_external(augmentation['approved_external'])
        ext_st=pd.read_csv(run_dir/'external_structures.csv',dtype={'train_row_id':str}).set_index('train_row_id')
        eligible=pd.read_csv(run_dir/'external_fold_eligibility.csv',keep_default_na=False)
        assert set(eligible.external_row_id)==set(external.train_row_id)
        assert not eligible.duplicated(['scheme','seed','outer_fold','scope','inner_scheme','inner_fold','external_row_id']).any()
        organizer=structures.set_index('train_row_id')
        expected_checks=0
        for (scheme,seed,fold),outer in eligible.groupby(['scheme','seed','outer_fold']):
            outer_ids=folds[(folds.scheme==scheme)&(folds.seed==seed)&(folds.fold==fold)].train_row_id
            outer_st=organizer.loc[outer_ids]
            assert len(outer[outer.scope=='outer'])==len(external)
            inner_part=inner[(inner.scheme==scheme)&(inner.seed==seed)&(inner.outer_fold==fold)]
            expected_checks+=len(external)*(1+sum(inner_part[inner_part.inner_scheme==s].inner_fold.nunique() for s in ('structure','scaffold')))
            for record in outer.itertuples():
                ext=ext_st.loc[record.external_row_id]
                blocked=ext.structure_id in set(outer_st.structure_id) or (scheme=='scaffold' and ext.scaffold_id in set(outer_st.scaffold_id))
                if record.scope=='inner':
                    inner_ids=inner_part[(inner_part.inner_scheme==record.inner_scheme)&(inner_part.inner_fold==int(record.inner_fold))].train_row_id
                    assert len(inner_ids)>0
                    held=organizer.loc[inner_ids]
                    blocked|=ext.structure_id in set(held.structure_id) or (record.inner_scheme=='scaffold' and ext.scaffold_id in set(held.scaffold_id))
                assert record.included==bool(not blocked)
        assert len(eligible)==expected_checks
        assert eligible.groupby(['scheme','seed','outer_fold']).ngroups==count
        candidate=joblib.load(run_dir/'candidate_model.joblib')
        ranking=pd.read_csv(run_dir/'final_search.csv').sort_values(['objective','candidate_id'])
        assert candidate['candidate']['id']==ranking.iloc[0].candidate_id
        assert candidate['manifest']['external_training_rows']==len(external)
        predict(str(run_dir/'candidate_model.joblib'),config['test'],config['sample'],str(output/'candidate_submission.csv'))
        assert (output/'candidate_submission.csv').read_bytes()==(run_dir/'candidate_submission.csv').read_bytes()
        population=pd.read_csv(run_dir/'training_population.csv',dtype={'train_row_id':str})
        expected_population=pd.concat([train,external],ignore_index=True) if selection['promoted'] else train
        pd.testing.assert_frame_equal(population,expected_population,check_dtype=False,rtol=1e-12)
        final_members=pd.read_csv(run_dir/'final_search_folds.csv',dtype={'train_row_id':str})
        for scheme,part in final_members.groupby('scheme'):
            final_groups=organizer.loc[part.train_row_id,scheme+'_id'].to_numpy()
            assert pd.DataFrame({'group':final_groups,'fold':part.fold.to_numpy()}).groupby('group').fold.nunique().eq(1).all()
        result.update(external_rows=len(external),external_eligibility_checks=len(eligible),external_group_isolation_verified=True,
                      evaluation_organizer_rows_only=True,candidate_submission_byte_identical_after_reload=True,
                      selected_training_rows=len(population),source_approval_verified=True)
        save_json(output/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='artifacts/advanced-v4')
    parser.add_argument('--output',default='artifacts/advanced-v4-review')
    args=parser.parse_args()
    print(json.dumps(verify(args.run,args.output),ensure_ascii=False,indent=2))
