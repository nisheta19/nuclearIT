"""Prepare a pending review queue without changing or medically interpreting labels."""
import json
from pathlib import Path

import pandas as pd

from .data import TARGET, save_json, sha256


def build_queue(run_dir,output):
    run_dir,output=Path(run_dir),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    expected=config['input_sha256']['train']
    if sha256(config['train'])!=expected:
        raise ValueError('Original training source changed')
    header=pd.read_csv(config['train'],nrows=0).columns
    metadata=[column for column in ('Title','strain','Culture') if column in header]
    original=pd.read_csv(config['train'],usecols=['SMILES',TARGET]+metadata,dtype=str,keep_default_na=False)
    structures=pd.read_csv(run_dir/'train_structures.csv',dtype={'train_row_id':str})
    duplicate=pd.read_csv(run_dir/'duplicate_review.csv',dtype=str).set_index('structure_id')
    y=pd.to_numeric(original[TARGET])
    rows=[]
    for i,structure in enumerate(structures.itertuples()):
        flags,questions=[],[]
        if y.iloc[i]==y.max():
            flags.append('extreme_si')
            questions.append('Подтвердить точное SI и соответствие исходной записи, вируса и экспериментальных условий.')
        if y.iloc[i]==10000:
            flags.append('si_10000')
            questions.append('Что означает 10000: точное измерение, граница измерения, ограничение при подготовке или условный код? Нужна первичная запись.')
        if y.iloc[i]==0:
            flags.append('zero_si')
            questions.append('Что означает ноль: измеренное значение, пропуск, предел измерения или иной код? Нужна первичная запись.')
        related=structures[structures.structure_id==structure.structure_id].train_row_id.tolist()
        if structure.structure_id in duplicate.index and duplicate.loc[structure.structure_id,'status']=='conflicting_labels':
            flags.append('conflicting_structure')
            questions.append('Объяснить расхождение SI одинаковой структуры: различаются ли условия или есть ошибка привязки/переноса?')
        if not flags:
            continue
        priority='P1' if any(flag!='si_10000' for flag in flags) else 'P2'
        rows.append({'train_row_id':str(structure.train_row_id),'source_record':i+1,'source_sha256':expected,
                     'priority':priority,'reasons':'; '.join(flags),'original_si_text':original.iloc[i][TARGET],
                     'SMILES':original.iloc[i].SMILES,'structure_id':structure.structure_id,
                     'related_row_ids':'; '.join(related),**{name:original.iloc[i][name] for name in metadata},
                     'questions':' '.join(questions),'review_status':'pending','primary_reference':'',
                     'reviewer':'','reviewed_at':'','verified_si':'','evidence':'','decision':''})
    frame=pd.DataFrame(rows).sort_values(['priority','source_record'])
    frame.to_csv(output/'medical_review_queue.csv',index=False,encoding='utf-8-sig')
    save_json(output/'medical_review_queue.json',frame.to_dict(orient='records'))
    summary={'rows':len(frame),'priority_counts':frame.priority.value_counts().to_dict(),
             'reason_counts':{flag:int(frame.reasons.str.contains(flag,regex=False).sum()) for flag in ('extreme_si','si_10000','zero_si','conflicting_structure')},
             'train_sha256':expected,'source_run':str(run_dir),'status':'pending','label_changes':0,
             'note':'Очередь вопросов, не медицинское заключение. Поля решения пусты. Исправления не применяются автоматически; внешние данные не добавлены.'}
    save_json(output/'summary.json',summary)
    return summary
