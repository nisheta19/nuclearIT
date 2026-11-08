# v6: similarity models and conservative blending

Promoted: **False**. Selected: **two_stage:r3_binary:a1:C0.1**.
Original 320 labels unchanged; SMILES only; no external training rows.
Kaggle score reported by user: **10,913,537.18910**, for v5 submission.
Rank and public/private status are unverified. No Kaggle submission made by this run.

| Cohort | Grouping | v4 MSE | v6 MSE | Reduction % | Passed |
|---|---|---|---|---|---|
| historical | structure | 14,816,985.027 | 14,490,712.093 | 2.202 | True |
| historical | scaffold | 18,713,792.961 | 19,657,515.748 | -5.043 | False |
| additional | structure | 14,421,766.978 | 14,471,419.609 | -0.344 | False |
| additional | scaffold | 17,010,830.457 | 17,154,064.520 | -0.842 | False |
| fresh | structure | 13,646,313.014 | 13,345,864.247 | 2.202 | True |
| fresh | scaffold | 17,113,240.404 | 17,478,154.631 | -2.132 | False |

The candidate grid uses seven structural representations, powered Tanimoto kernels,
linear or similarity-based logistic gates and kernel regression. Count fingerprints
use a pointwise log1p transform. All labels remain on the original SI scale.
Each outer training set selects the v4 candidate using its original inner selection;
the new candidate and blend weight minimize equal-weight structure/scaffold inner MSE.
Every landmark and fitted model uses only inner training rows. Original v4 inner
scores are reused only after hash, membership and recomputed-loss checks.

The original 45 outer folds have been reused during development. The 20 fresh folds
use different seeds, but the same molecules; they measure split robustness, not
independent generalization. Preliminary full-train grouped screens (98 then 140
candidates) informed development before nested evaluation. Selection uncertainty
is not covered by the conditional bootstrap. The hidden Kaggle score may worsen.

Reproduce: `python -m vacv similarity --config configs/similarity.json --output artifacts/new-similarity-run`
`submission.csv` is the selected release; `candidate_submission.csv` is the experiment.
