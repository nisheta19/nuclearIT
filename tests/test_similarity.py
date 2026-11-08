import joblib
import numpy as np
import pandas as pd
import pytest

from vacv.advanced import TwoStageRegressor
from vacv.chemistry import featurize
from vacv.kernel import TanimotoRidge, tanimoto_kernel
from vacv.models import predict_model
from vacv.nonlinear import SimilarityRegressor
from vacv.similarity_experiment import select_joint
from vacv.similarity_search import SimilarityBlend, features_and_kernels, inner_search


@pytest.mark.parametrize('gate',['direct','ridge','logistic','linear'])
@pytest.mark.parametrize('transform',['identity','log1p'])
def test_cached_kernel_matches_fit_and_roundtrip(gate,transform,tmp_path):
    rng=np.random.default_rng(7)
    x=rng.poisson(.4,size=(18,24)).astype(float)
    y=np.array([1.,30.,10000.,100.,12000.,2.]*3)
    q=rng.poisson(.5,size=(4,24)).astype(float)
    model=SimilarityRegressor(gate=gate,transform=transform,power=2).fit(x,y)
    z=np.log1p(x) if transform=='log1p' else x
    cached=SimilarityRegressor(gate=gate,transform=transform,power=2)._fit(z,y,tanimoto_kernel(z,z)**2)
    np.testing.assert_allclose(model.predict(q),cached.predict(q),rtol=1e-12)
    assert np.isfinite(model.predict(q)).all() and (model.predict(q)>=0).all()
    joblib.dump(model,tmp_path/'model.joblib')
    np.testing.assert_array_equal(model.predict(q),joblib.load(tmp_path/'model.joblib').predict(q))


def test_original_models_are_preserved_and_single_class_supported():
    rng=np.random.default_rng(42); x=rng.integers(0,2,(20,32)).astype(float)
    y=np.array([1.,100.,10000.,36000.]*5)
    q=rng.integers(0,2,(5,32)).astype(float)
    np.testing.assert_allclose(SimilarityRegressor(gate='direct').fit(x,y).predict(q),np.maximum(TanimotoRidge().fit(x,y).predict(q),0),rtol=1e-10)
    np.testing.assert_allclose(SimilarityRegressor(gate='linear',C=.1).fit(x,y).predict(q),TwoStageRegressor(C=.1).fit(x,y).predict(q),rtol=1e-10)
    for gate in ('direct','ridge','logistic','linear'):
        for label in (0.,25.,10000.):
            np.testing.assert_allclose(SimilarityRegressor(gate=gate).fit(x,np.full(20,label)).predict(q),label)


def test_joint_selection_prefers_incumbent_on_tie():
    y=np.array([10.,20.,30.]); old={'structure':y+1,'scaffold':y-1}
    ranking=select_joint(y,old,{'same':old},[0.,.5,1.])
    assert ranking.iloc[0].new_weight==0


def test_blend_routes_raw_count_features_and_invalid_fallback():
    binary={'radius':2,'bits':64,'include_chirality':True,'kind':'binary'}
    count={**binary,'kind':'count'}
    st,x=featurize(['CCO','CCCO','c1ccccc1','CCCCCC'],{'feature_sets':[binary,count]})
    y=np.array([1.,10.,10000.,15000.])
    old=TwoStageRegressor().fit(x[:,:64],y)
    new=SimilarityRegressor(transform='log1p').fit(x[:,64:],y)
    blend=SimilarityBlend(old,new,64,64,128,.25)
    np.testing.assert_allclose(blend.predict(x),.75*old.predict(x[:,:64])+.25*new.predict(x[:,64:]))
    q=np.vstack([x[:1],np.full((1,128),np.nan)])
    p,_=predict_model({'estimator':blend,'fallback_mean':y.mean(),'log_target':False,'clip_min':0.},q,[True,False])
    assert p[1]==y.mean()


def test_inner_membership_and_landmarks_exclude_held_groups(monkeypatch):
    from vacv import similarity_search
    spec={'name':'count','radius':2,'bits':64,'kind':'count','include_chirality':True}
    smiles=['CCO','CCO','CCCO','c1ccccc1','c1ccncc1','C1CCCCC1','C1CCCC1','CCN']
    st,_=featurize(smiles,spec)
    c={'similarity_search':{'feature_sets':[spec],'inner_folds':2}}
    x,k=features_and_kernels(smiles,c); y=np.array([1.,2.,3.,10000.,10000.,4.,5.,6.])
    one={'id':'test','representation':0,'transform':'log1p','gate':'logistic','power':1,'alpha':1.,'C':1.}
    monkeypatch.setattr(similarity_search,'candidate_grid',lambda config:[one])
    ranking,pred,membership=inner_search(x,k,y,st,c,17)
    for scheme,folds in membership.items():
        assert pd.DataFrame({'group':st[scheme+'_id'],'fold':folds}).groupby('group').fold.nunique().eq(1).all()
        expected=np.empty(len(y))
        for fold in np.unique(folds):
            fit,held=folds!=fold,folds==fold
            assert not set(st.loc[fit,scheme+'_id'])&set(st.loc[held,scheme+'_id'])
            raw_train=np.expm1(x[0][fit]); raw_test=np.expm1(x[0][held])
            expected[held]=SimilarityRegressor(transform='log1p',C=1.).fit(raw_train,y[fit]).predict(raw_test)
        np.testing.assert_allclose(pred['test'][scheme],expected,rtol=1e-10)
