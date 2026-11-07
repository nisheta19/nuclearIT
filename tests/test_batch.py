import csv
import io

import numpy as np
import pandas as pd
import pytest

from .test_pipeline import dataset
from vacv.batch import batch_predict, parse_csv, MAX_BYTES, MAX_ROWS, SIMILARITY_SPEC
from vacv.chemistry import featurize
from vacv.data import TARGET, load_inputs
from vacv.models import fit_model, predict_model


@pytest.mark.parametrize('text', ['', 'index\n1', 'SMILES,SMILES\nCCO,CCN', 'SMILES,index\n"unterminated', 'SMILES,index\nCCO,1,extra', 'SMILES\n', 'SMILES\n'+('C'*(MAX_BYTES+1)), 'SMILES\n'+'C\n'*(MAX_ROWS+1)], ids=['empty','no-smiles','duplicate-header','quotes','extra-field','header-only','too-many-bytes','too-many-rows'])
def test_csv_rejects_ambiguous_or_oversized_inputs(text):
    with pytest.raises(ValueError):
        parse_csv(text)


@pytest.mark.parametrize('delimiter', [',',';','\t'])
def test_csv_handles_bom_delimiters_and_quoted_ids(delimiter):
    rows,ignored=parse_csv('\ufeffSMILES'+delimiter+'index\r\nCCO'+delimiter+'"a,b;c"\r\n')
    assert rows==[{'SMILES':'CCO','index':'a,b;c'}] and not ignored


def test_delimiters_inside_quoted_headers_do_not_change_csv_dialect():
    rows,ignored=parse_csv('SMILES,"notes;with;semicolons"\nCCO,"a,b"\n')
    assert rows[0]['SMILES']=='CCO' and rows[0]['notes;with;semicolons']=='a,b'
    assert ignored==['notes;with;semicolons']


def test_batch_predictions_are_paired_with_rows_ignore_metadata_and_do_not_fill_errors(dataset):
    train,_,_=load_inputs(dataset)
    structures,x=featurize(train.SMILES,dataset['fingerprint'])
    _,similarity_x=featurize(train.SMILES,SIMILARITY_SPEC)
    bundle=fit_model('ridge_si',x,train[TARGET].to_numpy(),dataset)
    bundle['config']=dataset
    text='index,SMILES,CC50,IC50\r\n001,CCO,999999,0\r\n001,invalid,0,999999\r\n"=HYPERLINK(1)",OCC,3,4\r\nempty,,3,4\r\n'
    result=batch_predict(bundle,structures,similarity_x,text,'test-run','abc')
    assert result['total']==4 and result['valid']==2 and result['invalid']==2
    assert [r['index'] for r in result['rows']]==['001','001','=HYPERLINK(1)','empty']
    assert result['rows'][1]['predicted_si'] is None and result['rows'][3]['predicted_si'] is None
    assert result['rows'][0]['predicted_si']==result['rows'][2]['predicted_si']
    st,q=featurize(['CCO'],bundle['fingerprint'])
    np.testing.assert_allclose(result['rows'][0]['predicted_si'],predict_model(bundle,q,st.valid)[0][0])
    clean=batch_predict(bundle,structures,similarity_x,'SMILES\nCCO\n', 'test-run','abc')
    assert clean['rows'][0]['predicted_si']==result['rows'][0]['predicted_si']
    exported=list(csv.DictReader(io.StringIO(result['csv'].lstrip('\ufeff'))))
    assert exported[2]['index']=="'=HYPERLINK(1)"
    assert exported[1]['predicted_si']=='' and exported[0]['index']=='001'
    assert result['ignored_columns']==['CC50','IC50']
    assert result['target_review']['status']=='not_reviewed'
    assert {r['target_review_status'] for r in exported}=={'not_reviewed'}
