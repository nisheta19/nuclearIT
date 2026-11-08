import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from vacv.advanced import TwoStageRegressor
from vacv.chemistry import featurize
from vacv.data import save_json

ROOT=Path('.artifact-work/representation-v8')
t=pd.read_csv('artifacts/advanced-v4/train_for_model.csv');y=t['SI VACV'].to_numpy(float)
folds=pd.read_csv('artifacts/advanced-v4/folds.csv');before=pd.read_csv('artifacts/advanced-v4/oof_selected.csv')
specs={}
for bits in [2048,8192]:
 for radius in [2,3]:
  spec={'radius':radius,'bits':bits,'include_chirality':True}
  specs[(bits,radius)]=featurize(t.SMILES,spec)[1]
save_json(ROOT/'plan.json',{'fixed_C':.1,'alpha':1,'radii':[2,3],'bits':[2048,8192],'classifier_min_counts':[1,3,5],
 'primary':'Increase fingerprint width to 8192 and average r2/r3, fixed parameters from the released v4 model. Filtered 2048 and fixed unfiltered average are diagnostic alternatives. All 45 folds are development data; no independent test claim.'})
parts=[]
for (scheme,seed),part in folds.groupby(['scheme','seed'],sort=False):
 ids=part.sort_values('row_position').fold.to_numpy()
 for fold in np.unique(ids):
  fit,held=ids!=fold,ids==fold
  preds={}
  for bits in [2048,8192]:
   for radius in [2,3]:
    x=specs[(bits,radius)]
    model=TwoStageRegressor(alpha=1,C=.1).fit(x[fit],y[fit])
    preds[(bits,radius,1)]=model.predict(x[held])
    if bits==2048:
     for min_count in [3,5]:
      count=(x[fit]>0).sum(axis=0);keep=(count>=min_count)&(count<=fit.sum()-min_count)
      with threadpool_limits(limits=1):
       gate=LogisticRegression(C=.1,max_iter=2000,tol=1e-7).fit(x[fit][:,keep],y[fit]>=10000)
       p=gate.predict_proba(x[held][:,keep])[:,1]
      low,high=[np.maximum(r.predict(x[held]),0) if r is not None else np.full(held.sum(),mean) for r,mean in zip(model.regressors_,model.means_)]
      preds[(bits,radius,min_count)]=(1-p)*low+p*high
  for name,p in [('wide_8192',(preds[(8192,2,1)]+preds[(8192,3,1)])/2),('fixed_2048',(preds[(2048,2,1)]+preds[(2048,3,1)])/2),
                 ('filter3_2048',(preds[(2048,2,3)]+preds[(2048,3,3)])/2),('filter5_2048',(preds[(2048,2,5)]+preds[(2048,3,5)])/2)]:
   parts.append(pd.DataFrame({'scheme':scheme,'seed':seed,'fold':int(fold),'row_position':np.flatnonzero(held),'model':name,'y_true':y[held],'y_pred':p,'squared_error':(y[held]-p)**2}))
 pd.concat(parts).to_csv(ROOT/'oof.csv',index=False)
 print(scheme,seed,'complete',flush=True)
oof=pd.concat(parts)
scores=oof.groupby(['scheme','seed','model']).squared_error.mean().rename('mse').reset_index()
base=before.groupby(['scheme','seed']).squared_error.mean().rename('v4').reset_index();scores=scores.merge(base,on=['scheme','seed']);scores['reduction_pct']=(1-scores.mse/scores.v4)*100
scores.to_csv(ROOT/'scores.csv',index=False)
historical={'structure':[42,137,2026],'scaffold':[-1]};scores['phase']=['historical' if int(r.seed) in historical[r.scheme] else 'additional' for r in scores.itertuples()]
bench=scores.groupby(['phase','scheme','model']).agg(mse=('mse','mean'),v4=('v4','mean'));bench['reduction_pct']=(1-bench.mse/bench.v4)*100
bench.to_csv(ROOT/'benchmark.csv');print(bench.round(2).to_string(),flush=True)
