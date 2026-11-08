"""Read-only verification; review outputs stay outside the frozen experiment."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from vacv.chemistry import featurize
from vacv.data import TARGET, check_submission, save_json, sha256
from vacv.models import predict_model
from vacv.robust_experiment import benchmark_table, promotion_check


def verify(run):
    run=Path(run); config=json.loads((run/'config.json').read_text())
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    candidate_key=selection['comparison_candidate']
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,expected in hashes.items():
        assert sha256(run/name)==expected,name
    for name,expected in config['input_sha256'].items():
        assert sha256(config[name])==expected,name
    reference=Path(config['reference_run'])
    assert sha256(reference/'artifact_hashes.json')==config['reference_artifact_hashes_sha256']
    for name,expected in json.loads((reference/'artifact_hashes.json').read_text()).items():
        assert sha256(reference/name)==expected,name
    if (run/'baseline_provenance.json').is_file():
        provenance=json.loads((run/'baseline_provenance.json').read_text())
        baseline=Path(provenance['run'])
        assert provenance['used_model']=='incumbent_v4' and not provenance['candidate_model_used']
        assert sha256(baseline/'artifact_hashes.json')==provenance['artifact_hashes_sha256']
        for name,expected in json.loads((baseline/'artifact_hashes.json').read_text()).items():
            assert sha256(baseline/name)==expected,name
    assert manifest['features_from']==['SMILES']
    assert not manifest['external_data_used'] and manifest['target_changes']==0 and not manifest['sample_values_used']
    train=pd.read_csv(run/'train_for_model.csv',dtype={'train_row_id':str})
    pd.testing.assert_frame_equal(train,pd.read_csv(reference/'train_for_model.csv',dtype={'train_row_id':str}))
    structures=pd.read_csv(run/'train_structures.csv',dtype={'train_row_id':str})
    folds=pd.read_csv(run/'folds.csv',dtype={'train_row_id':str})
    inner=pd.read_csv(run/'inner_folds.csv',dtype={'train_row_id':str})
    isolation_checks=0
    for (scheme,seed),part in folds.groupby(['scheme','seed']):
        assert len(part)==len(train) and part.row_position.nunique()==len(train)
        assert part.groupby('group_id').fold.nunique().eq(1).all()
        np.testing.assert_array_equal(part.sort_values('row_position').group_id,structures[scheme+'_id'])
        for fold in part.fold.unique():
            held=set(part.loc[part.fold==fold,'row_position']); fit=set(range(len(train)))-held
            for inner_scheme in ('structure','scaffold'):
                sub=inner[(inner.scheme==scheme)&(inner.seed==seed)&(inner.outer_fold==fold)&(inner.inner_scheme==inner_scheme)]
                assert len(sub)==len(fit) and set(sub.row_position)==fit
                assert sub.groupby('group_id').inner_fold.nunique().eq(1).all()
                np.testing.assert_array_equal(sub.sort_values('row_position').group_id,structures.loc[sorted(fit),inner_scheme+'_id'])
                isolation_checks+=sub.inner_fold.nunique()
    oof=pd.read_csv(run/'oof.csv',dtype={'train_row_id':str})
    scores=pd.read_csv(run/'scores.csv')
    for row in scores.itertuples():
        part=oof[(oof.scheme==row.scheme)&(oof.seed==row.seed)&(oof.model==row.model)].sort_values('row_position')
        assert len(part)==len(train) and part.train_row_id.nunique()==len(train)
        np.testing.assert_array_equal(part.y_true,train[TARGET])
        assert np.isfinite(part.y_pred).all() and (part.y_pred>=0).all()
        np.testing.assert_allclose(row.mse,np.mean((part.y_true-part.y_pred)**2),rtol=1e-12)
    chosen=pd.read_csv(run/'outer_choices.csv')
    blends=pd.read_csv(run/('inner_blends.csv' if candidate_key=='similarity_search' else 'inner_scores.csv'))
    for row in chosen.itertuples():
        sub=blends[(blends.scheme==row.scheme)&(blends.seed==row.seed)&(blends.outer_fold==row.outer_fold)].copy()
        if candidate_key=='similarity_search':
            sub['_order']=sub.objective.round(6)
            best=sub.sort_values(['_order','new_weight','candidate_id']).iloc[0]
            assert best.new_weight==row.new_weight
        else:
            best=sub.sort_values(['objective','candidate_id']).iloc[0]
        assert best.candidate_id==row.candidate_id
    is_fresh=[int(row.seed) in config['fresh_seeds'][row.scheme] for row in scores.itertuples()]
    benchmark=benchmark_table(scores.loc[np.logical_not(is_fresh)],config)
    _,checks=promotion_check(benchmark,'incumbent_v4',candidate_key)
    for scheme in ('structure','scaffold'):
        sub=scores.loc[is_fresh]; sub=sub[sub.scheme==scheme].groupby('model').mse.mean()
        checks.append({'phase':'fresh','scheme':scheme,'passed':sub[candidate_key]<sub.incumbent_v4,
                       'baseline_mse':sub.incumbent_v4,'candidate_mse':sub[candidate_key]})
    assert selection['promoted']==all(check['passed'] for check in checks)
    for actual,expected in zip(checks,selection['promotion_checks']):
        assert actual['passed']==expected['passed']
        np.testing.assert_allclose([actual['baseline_mse'],actual['candidate_mse']],[expected['baseline_mse'],expected['candidate_mse']],rtol=1e-12)
    test=pd.read_csv(config['test'],usecols=['index','SMILES'],dtype=str)
    for stem in ('model','candidate_model'):
        bundle=joblib.load(run/(stem+'.joblib'))
        st,x=featurize(test.SMILES,bundle['fingerprint'])
        pred,_=predict_model(bundle,x,st.valid)
        csv=run/('submission.csv' if stem=='model' else 'candidate_submission.csv')
        check_submission(csv,config['sample'])
        np.testing.assert_allclose(pred,pd.read_csv(csv)[TARGET],rtol=1e-12,atol=1e-9)
        regenerated=pd.DataFrame({'index':test['index'],TARGET:pred}).to_csv(index=False).encode('utf-8')
        # pandas writes platform newlines when given a filesystem path on Windows.
        assert regenerated.replace(b'\r\n',b'\n')==csv.read_bytes().replace(b'\r\n',b'\n')
    expected_source=run/'candidate_submission.csv' if selection['promoted'] else reference/'submission.csv'
    assert sha256(run/'submission.csv')==sha256(expected_source)
    return {'verified':True,'frozen_artifacts':len(hashes),'outer_folds':len(chosen),'inner_group_isolation_checks':int(isolation_checks),
            'original_labels_unchanged':True,'selected_and_candidate_reload_match':True,'promoted':selection['promoted'],
            'submission_sha256':sha256(run/'submission.csv')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=verify(args.run)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);save_json(output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
