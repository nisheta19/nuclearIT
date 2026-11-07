import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .test_pipeline import dataset
from vacv.advanced import search
from vacv.augmentation import eligible_external, load_approved_external
from vacv.augmented_experiment import augmented_run
from vacv.data import TARGET, save_json, sha256
from vacv.robust_experiment import robust_run


def approved_fixture(tmp_path):
    path = tmp_path/'external.csv'
    pd.DataFrame({'record_id':['synthetic-1'],'SMILES':['N#N'],TARGET:[9.]}).to_csv(path,index=False)
    approval = {'authorization_status':'user_confirmed','authorization_date':'SYNTHETIC_TEST_ONLY',
        'training_csv_sha256':sha256(path),'excluded_bound_compounds':[],
        'records':[{'record_id':'synthetic-1','compound':'synthetic-1','SMILES':'N#N','si':9.,'doi':'synthetic',
        'approved_for_training':True,'pairing_reviewed':True,'review_status':'accepted_exact','activity_relation':'=','cc50_relation':'='}]}
    approval_path=tmp_path/'approval.json'; save_json(approval_path,approval)
    return {'train':str(path),'train_sha256':sha256(path),'approval':str(approval_path),
            'approval_sha256':sha256(approval_path),'expected_rows':1}


def test_loader_exposes_only_smiles_target_and_ids_and_refuses_changed_input(tmp_path):
    spec=approved_fixture(tmp_path)
    frame,_=load_approved_external(spec)
    assert frame.columns.tolist()==['train_row_id','SMILES',TARGET]
    assert frame.train_row_id.tolist()==['external:synthetic-1']
    Path(spec['train']).write_text('replaced',encoding='utf-8')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_approved_external(spec)


@pytest.mark.parametrize('field,value',[('cc50_relation','>'),('activity_relation','<'),('approved_for_training',False),('pairing_reviewed',False),('review_status','needs_review')])
def test_loader_rejects_bounds_and_unreviewed_rows_even_with_matching_hashes(tmp_path,field,value):
    spec=approved_fixture(tmp_path)
    approval=json.loads(Path(spec['approval']).read_text())
    approval['records'][0][field]=value
    save_json(spec['approval'],approval); spec['approval_sha256']=sha256(spec['approval'])
    with pytest.raises(ValueError,match='Bounded or unreviewed'):
        load_approved_external(spec)


def test_external_group_filter_covers_structure_and_scaffold():
    external=pd.DataFrame({'structure_id':['a','b','c'],'scaffold_id':['x','y','z']})
    held=pd.DataFrame({'structure_id':['a','d'],'scaffold_id':['w','y']})
    assert eligible_external(external,held,'structure').tolist()==[False,True,True]
    assert eligible_external(external,held,'scaffold').tolist()==[False,False,True]


def test_inner_search_never_fits_external_members_of_held_groups(monkeypatch):
    from vacv import advanced
    X=np.array([[1.,0.],[2.,0.],[3.,0.],[4.,0.]])
    structure=np.array(['a','b','c','d']); scaffold=np.array(['x','y','z','w'])
    extra={'X':np.array([[91.,0.],[92.,0.]]),'y':np.array([7.,8.]),
           'structures':pd.DataFrame({'structure_id':['a','extra'],'scaffold_id':['other','y']})}
    settings={'inner_folds':2,'inner_seed':1,'feature_sets':[{'bits':2}],
              'alphas':[1.],'weights':[[1.]],'stage_alphas':[],'classifier_C':[]}
    config={'advanced_search':settings,'clip_min':0.}
    calls=[]
    class Spy:
        def __init__(self,*args): pass
        def fit(self,x,y): calls.append(x.copy()); self.mean=float(np.mean(y)); return self
        def predict(self,x): return np.full(len(x),self.mean)
    monkeypatch.setattr(advanced,'TanimotoRidge',Spy)
    ranking,members=search(X,np.arange(4.),structure,scaffold,config,1,augmentation=extra)
    groups=pd.DataFrame({'structure_id':structure,'scaffold_id':scaffold})
    expected=[]
    for scheme,membership in members.items():
        for fold in np.unique(membership):
            held=membership==fold
            keep=eligible_external(extra['structures'],groups.loc[held],scheme)
            expected.append(np.concatenate([X[~held],extra['X'][keep]]))
    assert len(calls)==len(expected)==4
    for actual,wanted in zip(calls,expected): np.testing.assert_array_equal(actual,wanted)
    assert np.isfinite(ranking.objective).all()


def test_augmented_run_keeps_evaluation_cohort_and_saves_candidate(dataset,tmp_path):
    frame=pd.read_csv(dataset['train'])
    frame.loc[2:5,'SMILES']=['c1ccoc1','c1ccsc1','C1COCC1','C1CCNC1']
    frame.to_csv(dataset['train'],index=False)
    config=copy.deepcopy(dataset)
    config.update(structure_seeds=[42,19],scaffold_seeds=[-1,19],historical_seeds={'structure':[42],'scaffold':[-1]},bootstrap={'replicates':30,'seed':3},promotion_rule='synthetic fixture rule')
    specs=[{**config['fingerprint'],'name':'r2'}]
    config['robust_search']={'inner_folds':2,'inner_seed':17,'ridge_alphas':[10.],'kernel_alphas':[1.],'kernel_weights':[1.],'blend_linear_alpha':10.,'feature_sets':specs}
    config['advanced_search']={'inner_folds':2,'inner_seed':23,'alphas':[1.],'weights':[[1.]],'stage_alphas':[1.],'classifier_C':[1.],'high_si_threshold':10000.,'feature_sets':specs}
    reference=tmp_path/'reference'; robust_run(config,reference)
    config_path=tmp_path/'organizer.json'; save_json(config_path,config)
    settings={'organizer_config':str(config_path),'organizer_config_sha256':sha256(config_path),'reference_run':str(reference),
        'reference_artifact_hashes_sha256':sha256(reference/'artifact_hashes.json'),'training_dataset_version':'SYNTHETIC_AUGMENTED',
        'approved_external':approved_fixture(tmp_path),'promotion_rule':'all four must improve'}
    output=tmp_path/'augmented'; result=augmented_run(settings,output)
    oof=pd.read_csv(output/'oof.csv',dtype={'train_row_id':str})
    assert not oof.train_row_id.str.startswith('external:').any()
    assert oof.groupby(['model','scheme','seed']).size().eq(12).all()
    eligibility=pd.read_csv(output/'external_fold_eligibility.csv')
    assert (~eligibility.included).any()  # The synthetic external shares the ACYCLIC scaffold.
    assert (output/'candidate_model.joblib').is_file()
    population=pd.read_csv(output/'training_population.csv')
    assert len(population)==12+int(result['promoted'])
    if not result['promoted']:
        assert (output/'model.joblib').read_bytes()==(reference/'model.joblib').read_bytes()
        assert (output/'submission.csv').read_bytes()==(reference/'submission.csv').read_bytes()
