import joblib
import numpy as np
import pytest

from vacv.advanced import TwoStageRegressor
from vacv.chemistry import featurize
from vacv.stable import StageEnsemble,stable_search


def test_stage_ensemble_matches_components_and_survives_reload(tmp_path):
    rng=np.random.default_rng(22);x=rng.integers(0,2,(15,48)).astype(float)
    y=np.array([0.,50.,10000.]*5);q=rng.integers(0,2,(4,48)).astype(float)
    model=StageEnsemble([16,16,16],C=.1,weights=[.4,.4,.2]).fit(x,y)
    expected=sum(w*TwoStageRegressor(alpha=1.,C=.1).fit(x[:,i*16:(i+1)*16],y).predict(q[:,i*16:(i+1)*16]) for i,w in enumerate([.4,.4,.2]))
    np.testing.assert_allclose(model.predict(q),expected,rtol=1e-12)
    joblib.dump(model,tmp_path/'model.joblib');np.testing.assert_array_equal(model.predict(q),joblib.load(tmp_path/'model.joblib').predict(q))
    with pytest.raises(ValueError):StageEnsemble([16,16,16],weights=[1.,1.,1.]).fit(x,y)


def test_stable_search_preserves_duplicates_and_scaffolds():
    specs=[{'name':f'r{r}','radius':r,'bits':64,'include_chirality':True} for r in (2,3)]
    smiles=['CCO','CCO','CCCO','c1ccccc1','c1ccncc1','C1CCCCC1','C1CCCC1','CCN']
    st,x=featurize(smiles,{'feature_sets':specs});y=np.array([1.,2.,3.,10000.,10000.,4.,5.,6.])
    config={'stable_search':{'feature_sets':specs,'inner_folds':2,'classifier_C':[.1],'weights':[[.5,.5]]}}
    ranking,membership=stable_search(x,y,st,config,31)
    for scheme,folds in membership.items():
        expected=np.empty(len(y))
        for fold in np.unique(folds):
            fit,held=folds!=fold,folds==fold
            assert not set(st.loc[fit,scheme+'_id'])&set(st.loc[held,scheme+'_id'])
            expected[held]=StageEnsemble([64,64],C=.1,weights=[.5,.5]).fit(x[fit],y[fit]).predict(x[held])
        np.testing.assert_allclose(ranking.iloc[0][scheme+'_mse'],np.mean((y-expected)**2),rtol=1e-12)
