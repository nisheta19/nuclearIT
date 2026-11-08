"""Bounded CSV inference. Uploaded rows are never added to training."""
import csv
import io

import numpy as np

from .chemistry import featurize
from .diagnostics import tanimoto
from .models import predict_model
from .target_status import target_status

MAX_ROWS = 2000
MAX_BYTES = 2 * 1024 * 1024
SIMILARITY_SPEC = {'radius':2,'bits':2048,'include_chirality':True}
csv.field_size_limit(MAX_BYTES)


def parse_csv(text):
    if not isinstance(text,str) or not text.strip():
        raise ValueError('CSV пуст. Добавьте заголовок SMILES и строки структур.')
    if len(text.encode('utf-8')) > MAX_BYTES:
        raise ValueError('Размер CSV превышает 2 МБ.')
    text = text.lstrip('\ufeff')
    # Inspect parsed headers: delimiters inside a quoted header are ordinary text.
    delimiter = ','
    for candidate in (',',';','\t'):
        try:
            header = next(csv.reader(io.StringIO(text,newline=''),delimiter=candidate,strict=True))
            if 'SMILES' in [column.strip() for column in header]:
                delimiter = candidate
                break
        except (csv.Error,StopIteration):
            continue
    try:
        reader = csv.reader(io.StringIO(text,newline=''),delimiter=delimiter,strict=True)
        header = [column.strip() for column in next(reader)]
        if len(header)!=len(set(header)) or any(not column for column in header):
            raise ValueError('Названия столбцов должны быть непустыми и уникальными.')
        if 'SMILES' not in header:
            raise ValueError('В CSV нужен столбец SMILES. Дополнительно можно передать index.')
        rows=[]
        for row in reader:
            if not row:  # Empty physical lines are not CSV records.
                continue
            if len(row)!=len(header):
                raise ValueError(f'Строка CSV {reader.line_num}: число полей отличается от заголовка. Проверьте кавычки и разделитель.')
            rows.append(dict(zip(header,row)))
            if len(rows)>MAX_ROWS:
                raise ValueError('В одном файле допускается до 2000 молекул.')
        if not rows:
            raise ValueError('После заголовка нет строк с молекулами.')
    except (csv.Error,StopIteration) as error:
        raise ValueError('Не удалось прочитать CSV. Проверьте кавычки и разделители.') from error
    return rows,[column for column in header if column not in ('SMILES','index')]


def safe_csv_cell(value):
    if isinstance(value,str) and (value.lstrip().startswith(('=','+','-','@')) or value.startswith(('\t','\r','\n'))):
        return "'"+value
    return value


def result_csv(rows):
    target=io.StringIO(newline='')
    writer=csv.DictWriter(target,fieldnames=list(rows[0]),lineterminator='\r\n')
    writer.writeheader()
    writer.writerows({key:safe_csv_cell(value) for key,value in row.items()} for row in rows)
    return '\ufeff'+target.getvalue()


def batch_predict(bundle,train_st,train_x,text,run_name,model_sha256):
    review = target_status(bundle.get('manifest', {}))
    rows,ignored=parse_csv(text)
    values=[row['SMILES'].strip() for row in rows]
    oversized=[len(value)>2000 for value in values]
    safe_values=['' if too_long else value for value,too_long in zip(values,oversized)]
    st,x=featurize(safe_values,bundle['fingerprint'])
    valid=st.valid.to_numpy()
    prediction,_=predict_model(bundle,x,valid)
    _,fingerprint=featurize(safe_values,SIMILARITY_SPEC)
    similarities=np.full(len(rows),np.nan)
    similarities[valid]=tanimoto(fingerprint[valid],train_x).max(axis=1) if valid.any() else []
    counts=train_st.structure_id.value_counts()
    output=[]
    for i,(row,structure) in enumerate(zip(rows,st.itertuples())):
        ok=bool(valid[i])
        coverage='' if not ok else '<0.3' if similarities[i]<.3 else '0.3–0.5' if similarities[i]<.5 else '>=0.5'
        output.append({'row_number':i+1,'index':row.get('index',''),'SMILES':row['SMILES'],
            'canonical_smiles':structure.canonical_smiles if ok else '', 'predicted_si':float(prediction[i]) if ok else None,
            'status':'ok' if ok else 'invalid_smiles','error':'' if ok else 'SMILES длиннее 2000 символов' if oversized[i] else 'Пустой или неразбираемый SMILES',
            'max_similarity':float(similarities[i]) if ok else None,'coverage_band':coverage,
            'training_matches':int(counts.get(structure.structure_id,0)) if ok else 0,
            'model_name':bundle['name'],'run_name':run_name,'model_sha256':model_sha256,'dataset_version':bundle['config']['dataset_version'],
            'train_sha256':review['train_sha256'], 'target_review_status':review['status'], 'target_review_note':review['message'],
            'training_dataset_version':review['training_dataset_version'], 'external_data_used':review['external_data_used'],
            'external_train_sha256':review['external_train_sha256']})
    return {'rows':output,'total':len(rows),'valid':int(valid.sum()),'invalid':int((~valid).sum()),
            'below_0_3':int((similarities<.3).sum()),'below_0_5':int((similarities<.5).sum()),
            'ignored_columns':ignored,'csv':result_csv(output),'target_review':review,
            'note':review['message']+' Сходство описывает покрытие обучения, не уверенность. Для ошибочных SMILES прогноз отсутствует. Строки не добавляются в train. Потенциальные формулы в текстовых полях CSV экранированы апострофом.'}
