import copy
import csv
import io
from zipfile import ZipFile

import pytest

from scripts.audit_target_provenance import investigate, compare_versions, digest
from vacv.target_status import target_status, require_same_dataset, DISPUTED_TRAIN_SHA256


def table(rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['SMILES', 'SI VACV', 'IC50_uM_VACV', 'Title'])
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def test_archive_provenance_is_read_only_and_missingness_is_audit_only(tmp_path):
    raw = table([['CCO', '10000', '', 'a'], ['CCN', '0', '1', 'b'], ['CCC', '12.50', '2', 'c']])
    train, archive = tmp_path/'train.csv', tmp_path/'source.zip'
    train.write_bytes(raw)
    with ZipFile(archive, 'w') as output:
        output.writestr('train.csv', raw)
    source_bytes = archive.read_bytes()
    report, rows = investigate(train, archive)
    assert report['archive']['byte_equal_to_local']
    assert report['si_10000_equals_missing_ic50_rows']
    assert report['alternate_version']['status'] == 'not_supplied'
    assert report['labels_changed'] == 0 and report['features_from'] == ['SMILES']
    assert rows[0]['row_position'] == 0 and rows[0]['si_text'] == '10000'
    assert train.read_bytes() == raw and archive.read_bytes() == source_bytes


def test_archive_mismatch_fails_without_changing_train(tmp_path):
    original = table([['CCO', '10000', '', 'a']])
    train, archive = tmp_path/'train.csv', tmp_path/'source.zip'
    train.write_bytes(original)
    with ZipFile(archive, 'w') as output:
        output.writestr('train.csv', table([['CCO', '0', '', 'a']]))
    with pytest.raises(ValueError, match='Archive train differs'):
        investigate(train, archive)
    assert train.read_bytes() == original


def test_version_comparison_matches_records_even_after_reordering():
    original = table([['CCO', '10000', '', 'a'], ['CCN', '12.50', '2', 'b']])
    other = table([['CCN', '12.5', '2', 'b'], ['CCO', '0', '', 'a']])
    result = compare_versions(original, other)
    assert result['changed_rows'] == 1
    assert result['other_train_sha256'] == digest(other)
    assert result['changes'] == [{'local_row_position': 0, 'other_row_position': 1, 'local_si_text': '10000', 'other_si_text': '0'}]


def test_version_comparison_rejects_metadata_changes_and_ambiguous_matches():
    raw = table([['CCO', '10000', '', 'a']])
    with pytest.raises(ValueError, match='Non-target fields differ'):
        compare_versions(raw, table([['CCO', '0', '1', 'a']]))
    duplicates = table([['CCO', '10000', '', 'a'], ['CCO', '0', '', 'a']])
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        compare_versions(duplicates, duplicates)


def test_review_is_bound_to_train_hash_not_just_version_name():
    manifest = {'dataset_version': 'same_name', 'inputs': {'train': {'sha256': DISPUTED_TRAIN_SHA256}}}
    assert target_status(manifest)['disputed_rows'] == 81
    assert target_status(manifest)['status'] == 'unresolved_target_semantics'
    manifest['inputs']['train']['sha256'] = 'new_data'
    assert target_status(manifest)['status'] == 'not_reviewed'
    assert target_status(manifest)['disputed_rows'] is None


@pytest.mark.parametrize('field', ['train', 'test', 'sample', 'dataset_version', 'missing_hash'])
def test_cross_run_comparison_refuses_different_or_missing_dataset_identity(field):
    original = {'dataset_version': 'v1', 'inputs': {name: {'sha256': name+'hash'} for name in ('train', 'test', 'sample')}}
    require_same_dataset(original, copy.deepcopy(original))
    changed = copy.deepcopy(original)
    if field == 'dataset_version':
        changed[field] = 'v2'
    elif field == 'missing_hash':
        del changed['inputs']['train']['sha256']
    else:
        changed['inputs'][field]['sha256'] = 'different'
    with pytest.raises(ValueError, match='cross-version MSE'):
        require_same_dataset(original, changed)
