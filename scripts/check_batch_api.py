"""Verify real organizer test predictions through HTTP and the advertised batch size."""
import csv
import argparse
import io
import json
import time
from pathlib import Path
from urllib.request import Request,urlopen

import numpy as np
import pandas as pd


def post(path,value):
    request=Request('http://127.0.0.1:8765'+path,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=120) as response:
        return json.load(response)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,default=Path('artifacts/advanced-v4-review/batch_api_verification.json'))
    parser.add_argument('--run',type=Path,default=Path('artifacts/augmented-v5'))
    args=parser.parse_args()
    started=time.monotonic()
    run=args.run
    original=Path('data/raw/competition/test.csv').read_text(encoding='utf-8-sig')
    result=post('/api/batch',{'csv':original})
    expected=pd.read_csv(run/'submission.csv',dtype={'index':str})
    assert result['total']==result['valid']==len(expected) and result['invalid']==0
    assert [row['index'] for row in result['rows']]==expected['index'].tolist()
    np.testing.assert_allclose([row['predicted_si'] for row in result['rows']],expected['SI VACV'],rtol=1e-12,atol=1e-9)
    exported=list(csv.DictReader(io.StringIO(result['csv'].lstrip('\ufeff'))))
    assert len(exported)==len(expected)
    assert result['target_review']['status']=='unresolved_target_semantics'
    assert {row['target_review_status'] for row in exported}=={'unresolved_target_semantics'}
    assert {row['train_sha256'] for row in exported}=={result['target_review']['train_sha256']}
    expected_external=bool(json.loads((run/'selection.json').read_text(encoding='utf-8')).get('external_data_used',False))
    assert result['target_review']['external_data_used']==expected_external
    assert {row['external_data_used'] for row in exported}=={str(expected_external)}
    np.testing.assert_allclose([float(row['predicted_si']) for row in exported],expected['SI VACV'],rtol=1e-12,atol=1e-9)
    single=post('/api/predict',{'smiles':result['rows'][0]['SMILES']})
    np.testing.assert_allclose(single['predicted_si'],result['rows'][0]['predicted_si'],rtol=1e-12,atol=1e-9)
    assert single['target_review']==result['target_review']
    maximum=post('/api/batch',{'csv':'index,SMILES\n'+'\n'.join(f'{i:04d},CCO' for i in range(2000))+'\n'})
    assert maximum['total']==maximum['valid']==2000 and maximum['invalid']==0
    np.testing.assert_allclose([row['predicted_si'] for row in maximum['rows']],maximum['rows'][0]['predicted_si'],rtol=1e-12)
    assert maximum['rows'][-1]['index']=='1999'
    outcome={'real_test_rows':len(expected),'http_batch_matches_submission':True,'csv_export_matches_submission':True,
             'single_matches_batch':True,'ignored_test_columns':result['ignored_columns'],
             'maximum_2000_rows_passed':True,'target_status_in_api_and_export':True,'elapsed_seconds':round(time.monotonic()-started,2)}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(outcome,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(outcome,ensure_ascii=False,indent=2))
