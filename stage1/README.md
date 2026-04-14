# Stage 1 - Probability of Default (PD)

This folder contains the stable Stage 1 risk model workstream.

## Purpose

Stage 1 predicts applicant default probability:

$$
\hat{p}(x) = P(\text{Default} \mid x)
$$

This PD is used as a risk haircut in Stage 2.

## Key Files

- `test-stage-1.ipynb`: stage-local copy of the Stage 1 notebook
- `artifacts/pd_model_metrics.csv`: Stage 1 metrics snapshot
- `artifacts/preprocessing_audit.csv`: preprocessing lineage and data quality audit
- `artifacts/stability_results.csv`: multi-seed stability run output

## Path Conventions

- Data path: `../data/accepted_loans.csv`
- Plot path: `../shared_assets/plots`
- Stage 1 artifact path: `./artifacts`

## Typical Outputs

- ROC-AUC and calibration diagnostics
- SHAP explainability plots (written to `../shared_assets/plots`)
- saved model + preprocessing bundle under shared models

## Notes

- Stage 1 is treated as the reference risk component while Stage 2 is reworked.
- Keep Stage 1 behavior stable unless explicitly improving risk labeling or leakage controls.

