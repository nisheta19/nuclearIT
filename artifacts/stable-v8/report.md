# v8: two-stage ensembles

Promoted: **True**. Selected model: **stage_ensemble:C0.1:w0.5-0.5**.

| Cohort | Grouping | v4 MSE | v8 MSE | Reduction % | Passed |
|---|---|---|---|---|---|
| historical | structure | 14,816,985.027 | 14,262,906.092 | 3.739 | True |
| historical | scaffold | 18,713,792.961 | 18,284,534.908 | 2.294 | True |
| additional | structure | 14,421,766.978 | 14,086,798.335 | 2.323 | True |
| additional | scaffold | 17,010,830.457 | 16,963,591.735 | 0.278 | True |
| fresh | structure | 13,646,313.014 | 13,622,860.063 | 0.172 | True |
| fresh | scaffold | 17,113,240.404 | 16,904,444.287 | 1.220 | True |

1 candidate(s): C = [0.1]; fixed regression alpha = 1;
weights = [[0.5, 0.5]]. Fingerprints: ['r2_binary', 'r3_binary'].
One fixed candidate: equal r2/r3 two-stage ensemble, C=0.1 and alpha=1 inherited from the released v4 full-train choice. Selected after a four-alternative exploratory screen on the 45 reused folds; their scores are therefore development estimates, not unbiased nested-selection estimates. No adaptive inner selection occurs for this fixed model; stored inner scores are diagnostics. The fresh cohort is a predeclared split-robustness guard on the same molecules, not an independent dataset. All six promotion checks must pass.
The final model uses the same procedure. No Kaggle score is used as a selection loss.
The 45 earlier folds have been reused during development; fresh seeds use the
same molecules, so neither cohort is an independent test.

All 320 original labels unchanged; only SMILES as input; no external rows.
Kaggle MSE 10,913,537.18910 is user-reported for v5; its rank and public/private
status are unknown. A lower local MSE does not guarantee a lower Kaggle MSE.

Reproduce: `python scripts/run_stable_ensemble.py --config configs/stable_fixed.json --output artifacts/new-stable-run`
The v4 fresh baseline is taken from the verified similarity-v6 run; its candidate
is never used. `submission.csv` is the selected model, `candidate_submission.csv`
is experimental if promotion failed.
