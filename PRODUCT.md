# VACV SI Workbench

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Existing Python 3.12 / RDKit / scikit-learn project. A local browser screen calls the same saved Python model; buildless HTML/CSS/JS is an implementation choice for this small surface. No hosted deployment.

## Users

The hackathon team examining a SMILES-to-SI regression experiment and demonstrating its output.

## Product Purpose

Predict SI from one molecule or a CSV list, inspect structural coverage, and understand validation results and known failures. Export row-level results with model identity, and prepare questions for medical reviewers without changing labels. The user explicitly accepted these additions on 7 November 2026 by asking to implement all proposed improvements and recheck the result.

## Operating Context

Local Windows workspace, current organizer dataset hakathonMiphi, 320 train rows and 597 test rows. Predictions and reports are artifacts of the selected run.

## Capabilities and Constraints

Only SMILES-derived features. No concentration-derived features, no external data, no label repair. MSE in original SI scale. Prediction is an experimental model output, not measured activity. Exact structures are grouped for validation. Dataset identity is pinned by SHA-256. Project stored in the private GitHub repository nisheta19/nuclearIT; no submission to Kaggle.

## Evidence on Hand

Real baseline OOF, model, audits and reports in artifacts/baseline-original. The completed, promoted run artifacts/advanced-v4 supplies the current model, nested validation and reports, with artifacts/kernel-v3 retained as its comparator. Conditional paired cluster-bootstrap intervals do not cover all training/tuning uncertainty; one scaffold interval includes zero. OOF evaluates the selection procedure, not one fixed final configuration. The pending medical queue has 89 records; it contains questions, not medical conclusions. No invented metrics or scientific confidence claims.

## Product Principles

As of 8 November 2026, all 81 SI=10000 records coincide with missing IC50 in the
byte-verified source archive. Their semantics remain unresolved. A colleague reports
a zero-label alternative; that file has not been received here. Historical v4 results
remain reproducible but are conditional on the disputed labels. Show the hash-bound
review status in every prediction workflow and export. Do not replace labels or
present cross-version MSE as model improvement. The original archive and reproducible
investigation are included in the project; organizer clarification is pending.

- Show the structure next to the prediction.
- State which experiment produced the number.
- Similarity describes training coverage; it is not confidence probability.
- Keep audit evidence accessible without turning it into an input feature.
