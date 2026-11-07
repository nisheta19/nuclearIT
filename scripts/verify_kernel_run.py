"""Independent checks of saved splits, original labels, scores and model reload."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vacv.data import TARGET, save_json, sha256
from vacv.pipeline import predict


def verify(run_dir, baseline_dir, output):
    run_dir, baseline_dir, output = map(Path, (run_dir, baseline_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    hashes = json.loads((run_dir / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for relative, expected in hashes.items():
        assert sha256(run_dir / relative) == expected, relative
    for role, expected in config['input_sha256'].items():
        assert sha256(config[role]) == expected, role
    train = pd.read_csv(run_dir / 'train_for_model.csv', dtype={'train_row_id': str})
    source = pd.read_csv(config['train'])
    np.testing.assert_array_equal(train[TARGET], source[TARGET])
    np.testing.assert_array_equal(train.SMILES, source.SMILES)
    assert set(train.columns) == {'train_row_id', 'SMILES', TARGET}
    structures = pd.read_csv(run_dir / 'train_structures.csv', dtype={'train_row_id': str}).set_index('train_row_id')
    folds = pd.read_csv(run_dir / 'folds.csv', dtype={'train_row_id': str})
    inner = pd.read_csv(run_dir / 'inner_folds.csv', dtype={'train_row_id': str})
    assert set(inner.inner_scheme) == {'structure', 'scaffold'}
    count_outer = 0
    for (scheme, seed), assignment in folds.groupby(['scheme', 'seed']):
        assert len(assignment) == len(train) == assignment.train_row_id.nunique()
        assert assignment.groupby('group_id').fold.nunique().eq(1).all()
        # Identical structures stay together under either outer scheme.
        linked = assignment.merge(structures[['structure_id']], on='train_row_id')
        assert linked.groupby('structure_id').fold.nunique().eq(1).all()
        for fold in assignment.fold.unique():
            count_outer += 1
            fit_ids = set(assignment.loc[assignment.fold != fold, 'train_row_id'])
            for inner_scheme in ('structure', 'scaffold'):
                part = inner[(inner.scheme == scheme) & (inner.seed == seed) & (inner.outer_fold == fold) & (inner.inner_scheme == inner_scheme)]
                assert len(part) == len(fit_ids) and set(part.train_row_id) == fit_ids
                assert part.groupby('group_id').inner_fold.nunique().eq(1).all()
                groups = structures.loc[part.train_row_id, inner_scheme + '_id'].to_numpy()
                np.testing.assert_array_equal(part.group_id, groups)
    old_folds = pd.read_csv(baseline_dir / 'folds.csv', dtype={'train_row_id': str})
    common = folds[[int(row.seed) in config['historical_seeds'][row.scheme] for row in folds.itertuples()]]
    keys = ['scheme', 'seed', 'row_position']
    pd.testing.assert_frame_equal(common.sort_values(keys).reset_index(drop=True), old_folds.sort_values(keys).reset_index(drop=True))
    oof = pd.read_csv(run_dir / 'oof.csv', dtype={'train_row_id': str})
    scores = pd.read_csv(run_dir / 'scores.csv').set_index(['scheme', 'seed', 'model'])
    for key, part in oof.groupby(['scheme', 'seed', 'model']):
        assert len(part) == len(train) == part.train_row_id.nunique()
        np.testing.assert_array_equal(part.y_true, train.set_index('train_row_id').loc[part.train_row_id, TARGET])
        assert np.isfinite(part.y_pred).all()
        np.testing.assert_allclose(part.squared_error, (part.y_true - part.y_pred) ** 2, rtol=1e-12)
        np.testing.assert_allclose(scores.loc[key, 'mse'], np.mean((part.y_true - part.y_pred) ** 2), rtol=1e-12)
    ranking = pd.read_csv(run_dir / 'inner_scores.csv')
    choices = pd.read_csv(run_dir / 'outer_choices.csv')
    for choice in choices.itertuples():
        part = ranking[(ranking.scheme == choice.scheme) & (ranking.seed == choice.seed) & (ranking.outer_fold == choice.outer_fold)]
        best = part.sort_values(['objective', 'candidate_id']).iloc[0]
        assert choice.candidate_id == best.candidate_id
        np.testing.assert_allclose(part.objective, (part.structure_mse + part.scaffold_mse) / 2, rtol=1e-12)
    loaded = predict(str(run_dir / 'model.joblib'), config['test'], config['sample'], str(output / 'submission.csv'))
    assert (output / 'submission.csv').read_bytes() == (run_dir / 'submission.csv').read_bytes()
    pairs = scores.reset_index().pivot(index=['scheme', 'seed'], columns='model', values='mse')
    pairs['improvement_percent'] = (1 - pairs.robust_search / pairs.baseline_ridge) * 100
    pairs.to_csv(output / 'per_seed_comparison.csv')
    result = {'run': str(run_dir.resolve()), 'artifacts_checked': len(hashes), 'input_hashes_unchanged': True,
              'all_labels_and_smiles_unchanged': True, 'outer_folds_checked': count_outer,
              'inner_schemes_per_outer_fold': 2, 'held_out_rows_excluded_from_inner_search': True,
              'historical_folds_identical': True, 'mse_recomputed_from_oof': True,
              'saved_submission_byte_identical_after_reload': True,
              'seed_comparisons_improved': int((pairs.improvement_percent > 0).sum()),
              'seed_comparisons_total': len(pairs), 'submission': loaded}
    save_json(output / 'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='artifacts/kernel-v3')
    parser.add_argument('--baseline', default='artifacts/baseline-original')
    parser.add_argument('--output', default='artifacts/kernel-v3-review')
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.baseline, args.output), ensure_ascii=False, indent=2))
