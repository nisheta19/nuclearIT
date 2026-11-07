"""Paired cluster bootstrap of saved OOF errors, conditional on fitted models."""
import numpy as np
import pandas as pd


def cluster_interval(frame, group_column, replicates=5000, seed=9143):
    """Rows already average repeated-seed errors; keep each group's rows together."""
    group = frame.groupby(group_column).agg(before=('before','sum'), after=('after','sum'), n=('before','size'))
    if len(group) < 2:
        raise ValueError('At least two groups are required')
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(group), size=(replicates,len(group)))
    denominator = group.n.to_numpy()[drawn].sum(axis=1)
    before = group.before.to_numpy()[drawn].sum(axis=1)/denominator
    after = group.after.to_numpy()[drawn].sum(axis=1)/denominator
    delta = before-after
    low, high = np.quantile(delta, [.025,.975])
    return {'groups':len(group), 'rows':len(frame), 'before':float(frame.before.mean()), 'after':float(frame.after.mean()),
            'improvement':float((frame.before-frame.after).mean()), 'ci_low':float(low), 'ci_high':float(high),
            'bootstrap_fraction_positive':float((delta>0).mean()), 'replicates':replicates}


def compare_oof(oof, structures, config, comparisons):
    rows = []
    for before_name, after_name in comparisons:
        before = oof[oof.model == before_name]
        after = oof[oof.model == after_name]
        keys = ['scheme','seed','train_row_id']
        merged = before.merge(after, on=keys, suffixes=('_before','_after'), validate='one_to_one')
        if len(merged) != len(before) or len(merged) != len(after):
            raise ValueError('Paired comparison requires identical OOF rows')
        np.testing.assert_array_equal(merged.y_true_before, merged.y_true_after)
        merged['phase'] = ['historical' if int(r.seed) in config['historical_seeds'][r.scheme] else 'additional' for r in merged.itertuples()]
        for (phase, scheme), part in merged.groupby(['phase','scheme']):
            group_column = scheme+'_id'
            for metric in ('mse','mae'):
                power = 2 if metric == 'mse' else 1
                errors = part[['train_row_id']].assign(before=np.abs(part.y_true_before-part.y_pred_before)**power,
                                                      after=np.abs(part.y_true_after-part.y_pred_after)**power)
                # A molecule repeated across seeds is one statistical observation, not N independent ones.
                errors = errors.groupby('train_row_id', as_index=False)[['before','after']].mean()
                errors = errors.merge(structures[['train_row_id',group_column]], on='train_row_id', validate='one_to_one')
                rows.append({'reference':before_name,'candidate':after_name,'phase':phase,'scheme':scheme,'metric':metric,
                             'seed_count':part.seed.nunique(), **cluster_interval(errors, group_column, config['bootstrap']['replicates'],config['bootstrap']['seed'])})
    return pd.DataFrame(rows)
