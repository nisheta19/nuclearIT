"""Predeclared cross-representation ensembles and conditional-mean regression."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .kernel import TanimotoRidge
from .search import inner_assignments


class TwoStageRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1., C=1., threshold=10000.):
        self.alpha, self.C, self.threshold = alpha, C, threshold

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        high = y >= self.threshold
        self.probability_ = float(high.mean())
        self.classifier_ = None
        if high.any() and not high.all():
            with threadpool_limits(limits=1):
                self.classifier_ = LogisticRegression(C=self.C, max_iter=2000, tol=1e-7).fit(X, high)
            if self.classifier_.n_iter_.max() >= 2000:
                raise RuntimeError("Two-stage classifier did not converge")
        self.regressors_, self.means_ = [], []
        for mask in (~high, high):
            # Missing classes use a harmless mean; their mixture weight is exactly zero.
            self.means_.append(float(y[mask].mean()) if mask.any() else float(y.mean()))
            self.regressors_.append(TanimotoRidge(self.alpha).fit(X[mask], y[mask]) if mask.sum() >= 2 else None)
        return self

    def predict(self, X):
        with threadpool_limits(limits=1):
            p = self.classifier_.predict_proba(X)[:, 1] if self.classifier_ is not None else np.full(len(X), self.probability_)
            low, high = [np.maximum(model.predict(X), 0) if model is not None else np.full(len(X), mean)
                         for model, mean in zip(self.regressors_, self.means_)]
        return (1-p)*low + p*high


class RepresentationModel(RegressorMixin, BaseEstimator):
    def __init__(self, widths, candidate):
        self.widths, self.candidate = widths, candidate

    def fit(self, X, y):
        bounds = np.cumsum([0] + list(self.widths))
        self.models_ = []
        candidate = self.candidate
        weights = candidate['weights'] if candidate['family'] == 'ensemble' else [int(i == candidate['representation']) for i in range(len(self.widths))]
        for i, weight in enumerate(weights):
            if weight:
                model = TanimotoRidge(candidate['alpha']) if candidate['family'] == 'ensemble' else TwoStageRegressor(candidate['alpha'], candidate['C'], candidate['threshold'])
                model.fit(X[:, bounds[i]:bounds[i+1]], y)
                self.models_.append((int(bounds[i]), int(bounds[i+1]), weight, model))
        return self

    def predict(self, X):
        return sum(weight * model.predict(X[:, start:stop]) for start, stop, weight, model in self.models_)


def grid(config):
    settings = config['advanced_search']
    candidates = []
    for alpha in settings['alphas']:
        for weights in settings['weights']:
            if len(weights) != len(settings['feature_sets']) or not np.isclose(sum(weights), 1) or min(weights) < 0:
                raise ValueError('Ensemble weights must be nonnegative and sum to one')
            candidates.append({'id': f"ensemble:a{alpha:g}:w" + '-'.join(f'{w:.4g}' for w in weights), 'family':'ensemble', 'alpha':alpha, 'weights':weights})
    for i, spec in enumerate(settings['feature_sets']):
        for alpha in settings['stage_alphas']:
            for C in settings['classifier_C']:
                candidates.append({'id':f"two_stage:{spec['name']}:a{alpha:g}:C{C:g}", 'family':'two_stage', 'representation':i, 'alpha':alpha, 'C':C, 'threshold':settings['high_si_threshold']})
    return candidates


def fit_advanced(X, y, config, candidate):
    specs = config['advanced_search']['feature_sets']
    widths = [spec['bits'] + len(spec.get('descriptors', [])) for spec in specs]
    return {'name':candidate['id'], 'estimator':RepresentationModel(widths, candidate).fit(X,y),
            'log_target':False, 'fallback_mean':float(np.mean(y)), 'clip_min':config['clip_min'],
            'fingerprint':{'feature_sets':specs}, 'candidate':candidate}


def search(X, y, structure_groups, scaffold_groups, config, seed, augmentation=None):
    settings = config['advanced_search']
    candidates = grid(config)
    memberships = {name:inner_assignments(groups, settings['inner_folds'], seed)
                   for name, groups in [('structure', structure_groups), ('scaffold', scaffold_groups)]}
    bounds = np.cumsum([0] + [s['bits'] + len(s.get('descriptors', [])) for s in settings['feature_sets']])
    losses = {c['id']:{} for c in candidates}
    # External observations train the model only; selection loss remains on organizer rows.
    augmented_fit = {}
    if augmentation is not None:
        from .augmentation import eligible_external
        organizer_groups = pd.DataFrame({'structure_id':structure_groups, 'scaffold_id':scaffold_groups})
        for scheme, membership in memberships.items():
            for fold in np.unique(membership):
                keep = eligible_external(augmentation['structures'], organizer_groups.loc[membership == fold], scheme)
                augmented_fit[(scheme, int(fold))] = (augmentation['X'][keep], augmentation['y'][keep])
    for scheme, membership in memberships.items():
        # Cache component OOF predictions. Every estimator fits only its inner training rows.
        cache = {}
        for i in range(len(bounds)-1):
            x = X[:, bounds[i]:bounds[i+1]]
            components = [('kernel', alpha, None) for alpha in settings['alphas']]
            components += [('two_stage', alpha, C) for alpha in settings['stage_alphas'] for C in settings['classifier_C']]
            for kind, alpha, C in components:
                prediction = np.empty(len(y))
                for fold in np.unique(membership):
                    held = membership == fold
                    model = TanimotoRidge(alpha) if kind == 'kernel' else TwoStageRegressor(alpha, C, settings['high_si_threshold'])
                    fit_x, fit_y = x[~held], y[~held]
                    if augmentation is not None:
                        extra_x, extra_y = augmented_fit[(scheme, int(fold))]
                        fit_x = np.concatenate([fit_x, extra_x[:, bounds[i]:bounds[i+1]]])
                        fit_y = np.concatenate([fit_y, extra_y])
                    model.fit(fit_x, fit_y)
                    prediction[held] = model.predict(x[held])
                cache[(i,kind,alpha,C)] = prediction
        for candidate in candidates:
            if candidate['family'] == 'ensemble':
                pred = sum(w*cache[(i,'kernel',candidate['alpha'],None)] for i,w in enumerate(candidate['weights']) if w)
            else:
                pred = cache[(candidate['representation'],'two_stage',candidate['alpha'],candidate['C'])]
            if config['clip_min'] is not None:
                pred = np.maximum(pred, config['clip_min'])
            if not np.isfinite(pred).all():
                raise ValueError('Non-finite advanced OOF prediction')
            losses[candidate['id']][scheme+'_mse'] = float(np.mean((y-pred)**2))
    records = [{'candidate_id':c['id'], 'family':c['family'], **losses[c['id']],
                'objective':sum(losses[c['id']].values())/2} for c in candidates]
    return pd.DataFrame(records).sort_values(['objective','candidate_id']).reset_index(drop=True), memberships
