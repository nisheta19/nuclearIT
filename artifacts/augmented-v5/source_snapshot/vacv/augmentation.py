"""Explicitly approved external targets; concentrations remain in provenance only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET, headers, sha256


def load_approved_external(spec):
    for name in ('train', 'approval'):
        if sha256(spec[name]) != spec[name + '_sha256']:
            raise ValueError(f'Approved external {name} hash mismatch')
    if headers(spec['train']) != ['record_id', 'SMILES', TARGET]:
        raise ValueError('External model table must contain only record_id, SMILES and SI VACV')
    approval = json.loads(Path(spec['approval']).read_text(encoding='utf-8'))
    if approval.get('training_csv_sha256') != spec['train_sha256']:
        raise ValueError('Approval does not match the external training table')
    if approval.get('authorization_status') != 'user_confirmed':
        raise ValueError('External data require explicit user confirmation')
    frame = pd.read_csv(spec['train'], dtype=str, keep_default_na=False)
    if len(frame) != spec['expected_rows'] or frame.record_id.eq('').any() or frame.record_id.duplicated().any():
        raise ValueError('Unexpected external row count or invalid record IDs')
    records = approval['records']
    if len(records) != len(frame) or set(frame.record_id) != {r['record_id'] for r in records}:
        raise ValueError('Approved record set differs from external training rows')
    lookup = {r['record_id']: r for r in records}
    y = pd.to_numeric(frame[TARGET], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid external targets')
    for row, target in zip(frame.itertuples(index=False, name=None), y):
        record_id, smiles, _ = row
        record = lookup[record_id]
        if (record.get('approved_for_training') is not True or record.get('pairing_reviewed') is not True
                or record.get('review_status') != 'accepted_exact'
                or record.get('activity_relation') != '=' or record.get('cc50_relation') != '='):
            raise ValueError('Bounded or unreviewed records are not exact training targets')
        if smiles != record['SMILES'] or not np.isclose(target, record['si'], rtol=1e-12, atol=0):
            raise ValueError('External structure or target differs from approval')
    result = frame.rename(columns={'record_id': 'train_row_id'})
    result['train_row_id'] = 'external:' + result.train_row_id
    result[TARGET] = y
    return result[['train_row_id', 'SMILES', TARGET]], approval


def eligible_external(external_structures, held_structures, scheme):
    """Exclude held-out structures and, in scaffold validation, their whole scaffold."""
    if scheme not in ('structure', 'scaffold'):
        raise ValueError('Unknown grouping scheme')
    keep = ~external_structures.structure_id.isin(held_structures.structure_id).to_numpy()
    if scheme == 'scaffold':
        keep &= ~external_structures.scaffold_id.isin(held_structures.scaffold_id).to_numpy()
    return keep
