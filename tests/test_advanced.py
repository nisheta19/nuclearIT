import copy
import json

import joblib
import numpy as np
import pandas as pd

from .test_pipeline import dataset
from vacv.advanced import TwoStageRegressor, fit_advanced, search
from vacv.advanced_experiment import advanced_run
from vacv.chemistry import featurize
from vacv.data import sha256
from vacv.models import predict_model
from vacv.pipeline import predict
from vacv.reliability import cluster_interval
from vacv.robust_experiment import robust_run


def test_two_stage_missing_class_and_probability_weighted_prediction():
    x=np.eye(4)
    for y in (np.full(4,25.),np.full(4,10000.)):
        model=TwoStageRegressor().fit(x,y)
        np.testing.assert_allclose(model.predict(x[:2]), y[:2])
    model=TwoStageRegressor().fit(x,np.array([2.,5.,10000.,12000.]))
    q=np.ones((2,4))
    p=model.classifier_.predict_proba(q)[:,1]
    expected=(1-p)*np.maximum(model.regressors_[0].predict(q),0)+p*np.maximum(model.regressors_[1].predict(q),0)
    np.testing.assert_allclose(model.predict(q),expected)


def test_cluster_bootstrap_preserves_row_weighting_and_pairing():
    frame=pd.DataFrame({'group':['a','a','b'],'before':[10.,10.,1.],'after':[8.,8.,0.]})
    result=cluster_interval(frame,'group',1000,8)
    assert result['groups']==2 and result['rows']==3
    assert np.isclose(result['improvement'],5/3)
    assert 1 <= result['ci_low'] <= result['ci_high'] <= 2
    frame['after']=frame['before']
    result=cluster_interval(frame,'group',100,8)
    assert result['ci_low']==result['ci_high']==result['improvement']==0


def test_advanced_end_to_end_and_group_isolation(dataset,tmp_path):
    data=pd.read_csv(dataset['train'])
    data.loc[2:5,'SMILES']=['c1ccoc1','c1ccsc1','C1COCC1','C1CCNC1']
    data.to_csv(dataset['train'],index=False)
    c=copy.deepcopy(dataset)
    c.update(structure_seeds=[42,19],scaffold_seeds=[-1,19],historical_seeds={'structure':[42],'scaffold':[-1]},promotion_rule='test')
    specs=[{**c['fingerprint'],'name':'r2'},{**c['fingerprint'],'radius':3,'name':'r3'}]
    c['robust_search']={'inner_folds':2,'inner_seed':17,'ridge_alphas':[10.],'kernel_alphas':[1.],'kernel_weights':[1.],'blend_linear_alpha':10.,'feature_sets':specs}
    reference=tmp_path/'reference'
    robust_run(c,reference)
    c.update(reference_run=str(reference),reference_artifact_hashes_sha256=sha256(reference/'artifact_hashes.json'),bootstrap={'replicates':100,'seed':3})
    c['advanced_search']={'inner_folds':2,'inner_seed':23,'alphas':[1.],'weights':[[.5,.5]],'stage_alphas':[1.],'classifier_C':[1.],'high_si_threshold':10000.,'feature_sets':specs}
    output=tmp_path/'advanced'
    result=advanced_run(c,output)
    inner=pd.read_csv(output/'inner_folds.csv')
    folds=pd.read_csv(output/'folds.csv')
    for (scheme,seed,fold,inner_scheme),part in inner.groupby(['scheme','seed','outer_fold','inner_scheme']):
        held=folds[(folds.scheme==scheme)&(folds.seed==seed)&(folds.fold==fold)]
        assert not set(held.train_row_id)&set(part.train_row_id)
        assert part.groupby('group_id').inner_fold.nunique().eq(1).all()
    assert len(pd.read_csv(output/'reliability.csv'))==16
    predict(str(output/'model.joblib'),c['test'],c['sample'],str(tmp_path/'predicted.csv'))
    assert (output/'submission.csv').read_bytes()==(tmp_path/'predicted.csv').read_bytes()
    if not result['promoted']:
        assert (output/'submission.csv').read_bytes()==(reference/'submission.csv').read_bytes()
    # The candidate's multi-representation serializer is checked even when it is not promoted.
    candidate=joblib.load(output/'candidate_model.joblib')
    st,x=featurize(['CCO','invalid'],candidate['fingerprint'])
    pred,_=predict_model(candidate,x,st.valid)
    assert np.isfinite(pred).all() and x.shape[1]==sum(s['bits'] for s in specs)
