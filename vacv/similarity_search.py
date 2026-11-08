"""Grouped inner selection of similarity models and a conservative v4 blend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced import fit_advanced
from .chemistry import featurize
from .kernel import tanimoto_kernel
from .nonlinear import SimilarityRegressor
from .search import inner_assignments


def candidate_grid(config):
    rows = []
    for i, spec in enumerate(config['similarity_search']['feature_sets']):
        transform = 'log1p' if spec.get('kind') == 'count' else 'identity'
        for power in (1, 2):
            for alpha in (.1, 1., 10.):
                rows.append(dict(representation=i, transform=transform, gate='direct', power=power, alpha=alpha, C=1.))
            for C in (1., 10., 100.):
                rows.append(dict(representation=i, transform=transform, gate='logistic', power=power, alpha=1., C=C))
        for alpha in (.1, 1.):
            rows.append(dict(representation=i, transform=transform, gate='ridge', power=1, alpha=alpha, C=1.))
            for C in (.01, .1, 1.):
                rows.append(dict(representation=i, transform=transform, gate='linear', power=1, alpha=alpha, C=C))
    for row in rows:
        row['id'] = f"{config['similarity_search']['feature_sets'][row['representation']]['name']}:{row['gate']}:p{row['power']}:a{row['alpha']:g}:C{row['C']:g}"
    return rows


def model_for(candidate):
    return SimilarityRegressor(**{k:v for k,v in candidate.items() if k not in ('id', 'representation')})


def features_and_kernels(smiles, config):
    matrices, kernels = [], {}
    for i, spec in enumerate(config['similarity_search']['feature_sets']):
        _, x = featurize(smiles, spec)
        if spec.get('kind') == 'count':
            x = np.log1p(x)
        matrices.append(x)
        k = tanimoto_kernel(x,x)
        for power in (1,2):
            kernels[(i,power)] = k ** power
    return matrices, kernels


def inner_search(matrices, kernels, y, structures, config, seed):
    membership = {scheme:inner_assignments(structures[scheme+'_id'],config['similarity_search']['inner_folds'],seed)
                  for scheme in ('structure','scaffold')}
    predictions, rows = {}, []
    for candidate in candidate_grid(config):
        rep, power = candidate['representation'], candidate['power']
        x, K = matrices[rep], kernels[(rep,power)]
        by_scheme, row = {}, {'candidate_id':candidate['id']}
        for scheme, folds in membership.items():
            pred = np.empty(len(y))
            for fold in np.unique(folds):
                fit, held = folds != fold, folds == fold
                model = model_for(candidate)._fit(x[fit], y[fit], K[np.ix_(fit,fit)])
                pred[held] = model.predict_kernel(K[np.ix_(held,fit)], x[held])
            by_scheme[scheme] = pred
            row[scheme+'_mse'] = float(np.mean((pred-y)**2))
        row['objective'] = (row['structure_mse']+row['scaffold_mse'])/2
        rows.append(row)
        predictions[candidate['id']] = by_scheme
    return pd.DataFrame(rows).sort_values(['objective','candidate_id']).reset_index(drop=True), predictions, membership


def old_inner_predictions(x, y, candidate, membership, old_config):
    predictions = {}
    for scheme, folds in membership.items():
        pred = np.empty(len(y))
        for fold in np.unique(folds):
            fit, held = folds != fold, folds == fold
            model = fit_advanced(x[fit], y[fit], old_config, candidate)['estimator']
            pred[held] = np.maximum(model.predict(x[held]),0)
        predictions[scheme] = pred
    return predictions


def choose_blend(y, old_predictions, new_predictions, weights):
    rows = []
    for weight in weights:
        row = {'new_weight':weight}
        for scheme in ('structure','scaffold'):
            pred = (1-weight)*old_predictions[scheme] + weight*new_predictions[scheme]
            row[scheme+'_mse'] = float(np.mean((y-pred)**2))
        row['objective'] = (row['structure_mse']+row['scaffold_mse'])/2
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['objective','new_weight']).reset_index(drop=True)


class SimilarityBlend:
    """Compatible with the existing SMILES-only saved model/prediction API."""
    def __init__(self, old_model, new_model, old_width, new_start, new_stop, new_weight):
        self.old_model, self.new_model = old_model, new_model
        self.old_width, self.new_start, self.new_stop, self.new_weight = old_width, new_start, new_stop, new_weight

    def predict(self, X):
        old = np.maximum(self.old_model.predict(X[:,:self.old_width]),0)
        if self.new_weight == 0:
            return old
        new = self.new_model.predict(X[:,self.new_start:self.new_stop])
        return (1-self.new_weight)*old + self.new_weight*new
