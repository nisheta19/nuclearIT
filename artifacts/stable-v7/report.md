# v7: small-grid two-stage ensembles

Promoted: **False**. Selected model: **two_stage:r3_binary:a1:C0.1**.

| Cohort | Grouping | v4 MSE | v7 MSE | Reduction % | Passed |
|---|---|---|---|---|---|
| historical | structure | 14,816,985.027 | 14,425,228.541 | 2.644 | True |
| historical | scaffold | 18,713,792.961 | 18,630,460.486 | 0.445 | True |
| additional | structure | 14,421,766.978 | 14,193,528.825 | 1.583 | True |
| additional | scaffold | 17,010,830.457 | 17,152,852.202 | -0.835 | False |
| fresh | structure | 13,646,313.014 | 13,749,434.833 | -0.756 | False |
| fresh | scaffold | 17,113,240.404 | 17,256,388.639 | -0.836 | False |

Six candidates: C = 0.03, 0.1, 0.3; fixed regression alpha = 1;
Morgan r2/r3 weights 0.5/0.5 or r2/r3/feature-r2 weights 0.4/0.4/0.2.
All hyperparameter selection uses grouped inner validation in each outer train.
The final model uses the same selection procedure. No Kaggle score is used as a
selection loss. The six-candidate grid was fixed before observing any v7 OOF
results or the 20 fresh-fold results, after v6 worsened additional structure MSE.
The 45 earlier folds have been reused during development; fresh seeds use the
same molecules, so neither cohort is an independent test.

All 320 original labels unchanged; only SMILES as input; no external rows.
Kaggle MSE 10,913,537.18910 is user-reported for v5; its rank and public/private
status are unknown. A lower local MSE does not guarantee a lower Kaggle MSE.

Reproduce: `python scripts/run_stable_ensemble.py --config configs/stable.json --output artifacts/new-stable-run`
The v4 fresh baseline is taken from the verified similarity-v6 run; its candidate
is never used. `submission.csv` is the selected model, `candidate_submission.csv`
is experimental if promotion failed.
