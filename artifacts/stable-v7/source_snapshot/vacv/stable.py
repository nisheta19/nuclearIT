"""Small-grid ensembles of conditional-mean regressors on Morgan fingerprints."""
import numpy as np
import pandas as pd

from .advanced import TwoStageRegressor
from .search import inner_assignments


class StageEnsemble:
    def __init__(self,widths,C=.1,weights=(.5,.5,0.),alpha=1.):
        self.widths,self.C,self.weights,self.alpha=widths,C,weights,alpha

    def fit(self,X,y):
        if len(self.widths)!=len(self.weights) or min(self.weights)<0 or not np.isclose(sum(self.weights),1):
            raise ValueError('Invalid stage ensemble weights')
        bounds=np.cumsum([0]+list(self.widths));self.models_=[]
        for i,w in enumerate(self.weights):
            if w:
                start,stop=int(bounds[i]),int(bounds[i+1])
                self.models_.append((start,stop,w,TwoStageRegressor(alpha=self.alpha,C=self.C).fit(X[:,start:stop],y)))
        return self

    def predict(self,X):
        return sum(w*model.predict(X[:,start:stop]) for start,stop,w,model in self.models_)


def stable_grid(config):
    settings=config['stable_search']
    return [dict(id=f'stage_ensemble:C{C:g}:w'+ '-'.join(f'{w:g}' for w in weights),C=C,weights=weights,alpha=1.)
            for C in settings['classifier_C'] for weights in settings['weights']]


def stable_search(X,y,structures,config,seed):
    settings=config['stable_search'];widths=[s['bits'] for s in settings['feature_sets']]
    bounds=np.cumsum([0]+widths)
    membership={scheme:inner_assignments(structures[scheme+'_id'],settings['inner_folds'],seed) for scheme in ('structure','scaffold')}
    cache={}
    for scheme,folds in membership.items():
        for i in range(len(widths)):
            x=X[:,bounds[i]:bounds[i+1]]
            for C in settings['classifier_C']:
                pred=np.empty(len(y))
                for fold in np.unique(folds):
                    fit,held=folds!=fold,folds==fold
                    pred[held]=TwoStageRegressor(alpha=1.,C=C).fit(x[fit],y[fit]).predict(x[held])
                cache[(scheme,i,C)]=pred
    rows=[]
    for candidate in stable_grid(config):
        row={'candidate_id':candidate['id']}
        for scheme in membership:
            pred=sum(w*cache[(scheme,i,candidate['C'])] for i,w in enumerate(candidate['weights']) if w)
            row[scheme+'_mse']=float(np.mean((y-pred)**2))
        row['objective']=(row['structure_mse']+row['scaffold_mse'])/2
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['objective','candidate_id']).reset_index(drop=True),membership
