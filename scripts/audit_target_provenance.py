"""Read-only source/target investigation. Concentrations are audit fields, never features."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zipfile import ZipFile

TARGET = 'SI VACV'
IC50 = 'IC50_uM_VACV'
MISSING = {'', 'nan', 'na', 'n/a', 'null', 'none'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_table(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    columns = reader.fieldnames or []
    if len(set(columns)) != len(columns) or not {TARGET, 'SMILES'}.issubset(columns):
        raise ValueError('Train needs unique headers, SMILES and SI VACV')
    rows = list(reader)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed CSV row')
        try:
            y = Decimal(row[TARGET])
        except InvalidOperation as error:
            raise ValueError('Invalid target') from error
        if not y.is_finite() or y < 0:
            raise ValueError('Invalid target')
    return columns, rows


def compare_versions(local_raw, other_raw):
    columns, local = read_table(local_raw)
    other_columns, other = read_table(other_raw)
    if set(columns) != set(other_columns):
        raise ValueError('Versions have different columns')
    fields = sorted(set(columns) - {TARGET})
    key = lambda row: tuple(row[field] for field in fields)
    local_keys, other_keys = Counter(map(key, local)), Counter(map(key, other))
    if local_keys != other_keys:
        raise ValueError('Non-target fields differ; not a target-only version change')
    if any(n > 1 for n in local_keys.values()):
        raise ValueError('Ambiguous duplicate records: require an independent stable row identifier')
    lookup = {key(row): (i, row) for i, row in enumerate(other)}
    changed = []
    for i, row in enumerate(local):
        j, alternate = lookup[key(row)]
        if Decimal(row[TARGET]) != Decimal(alternate[TARGET]):
            changed.append({'local_row_position': i, 'other_row_position': j,
                            'local_si_text': row[TARGET], 'other_si_text': alternate[TARGET]})
    return {'status': 'verified_against_provided_file', 'other_train_sha256': digest(other_raw),
            'non_target_fields_equal': True, 'changed_rows': len(changed), 'changes': changed}


def investigate(train_path, archive_path=None, other_train=None):
    raw = Path(train_path).read_bytes()
    columns, rows = read_table(raw)
    if IC50 not in columns:
        raise ValueError('IC50 audit column is absent; missingness cannot be checked')
    missing = {i for i, row in enumerate(rows) if row[IC50].strip().lower() in MISSING}
    ten = {i for i, row in enumerate(rows) if Decimal(row[TARGET]) == 10000}
    zeros = {i for i, row in enumerate(rows) if Decimal(row[TARGET]) == 0}
    archive = {'status': 'not_supplied', 'pre_archive_provenance': 'unknown'}
    if archive_path:
        path = Path(archive_path)
        with ZipFile(path) as source:
            names = [name for name in source.namelist() if Path(name).name == 'train.csv']
            if len(names) != 1:
                raise ValueError('Archive must contain exactly one train.csv')
            member = source.read(names[0])
        archive.update(status='verified', archive_name=path.name, archive_sha256=digest(path.read_bytes()),
                       member=names[0], member_sha256=digest(member), byte_equal_to_local=member == raw)
        if member != raw:
            raise ValueError('Archive train differs from local train; no provenance confirmation')
    comparison = (compare_versions(raw, Path(other_train).read_bytes()) if other_train else
                  {'status': 'not_supplied', 'changed_rows': None})
    report = {'schema_version': 1, 'train_sha256': digest(raw), 'train_rows': len(rows),
              'zero_targets': len(zeros), 'targets_10000': len(ten), 'missing_ic50': len(missing),
              'si_10000_equals_missing_ic50_rows': ten == missing,
              'si_10000_row_positions': sorted(ten), 'zero_row_positions': sorted(zeros),
              'archive': archive, 'alternate_version': comparison,
              'label_semantics': 'unresolved', 'labels_changed': 0, 'features_from': ['SMILES'],
              'note': 'Missing IC50 is inspected only for provenance, never used as a feature or a rule to repair labels.'}
    records = [{'train_sha256': digest(raw), 'row_position': i, 'csv_record_number': i + 2,
                'Title': rows[i].get('Title', ''), 'SMILES': rows[i]['SMILES'],
                'si_text': rows[i][TARGET], 'ic50_missing_audit_only': i in missing,
                'review_status': 'pending_organizer_clarification'} for i in sorted(ten)]
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/raw/competition/train.csv'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--other-train', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New directory; source files are never modified')
    args = parser.parse_args()
    report, records = investigate(args.train, args.archive, args.other_train)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'labels_for_clarification.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['train_sha256', 'row_position', 'csv_record_number', 'Title', 'SMILES', 'si_text', 'ic50_missing_audit_only', 'review_status']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('row_positions')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
