# CLARE: Credit Limit Allocation with Risk Estimation

## Executive Summary

CLARE is a two-stage machine learning pipeline built on Lending Club-style accepted loan data to produce risk-adjusted lending limits.

1. Stage 1 predicts Probability of Default (PD) per applicant.
2. Stage 2 predicts Raw Capacity (the unconstrained affordable loan amount).
3. Final lending limit is risk-adjusted using:

$$
Final\_Limit = Raw\_Capacity \times (1 - PD)
$$

What this means in business terms:

- Higher-risk applicants get a larger haircut.
- Lower-risk applicants retain more of their raw capacity.
- The system creates a transparent balance between growth and credit risk control.

Latest run highlights from saved artifacts in this repo:

- Stage 1 PD model test ROC-AUC: 0.7672
- Stage 1 CV ROC-AUC mean: 0.7599 (std: 0.0021)
- Stage 2 allocator test MAE: 46.62
- Stage 2 allocator test RMSE: 235.37
- Stage 2 allocator test R2: 0.9993
- Average portfolio risk haircut: 45.29%
- Dataset rows used in Stage 1 modeling: 2,245,134

## Project At A Glance

This repository contains a production-style credit allocation workflow:

- Notebook 1: Stage 1 PD modeling, validation, explainability, and stability testing
- Notebook 2: Stage 2 capacity modeling and integration with Stage 1 PD
- Saved model artifacts for inference-only scoring (no retraining required)
- Audit and metrics files for reproducibility

Core files:

- accepted_loans.csv: primary dataset used by both stages
- test-stage-1.ipynb: PD model development and evaluation
- test-stage-2.ipynb: capacity model, risk-adjusted allocation, and Stage 2 SHAP explainability
- artifacts/pd_model_metrics.csv: Stage 1 headline metrics
- artifacts/stage2_allocator_metrics.csv: Stage 2 headline metrics
- artifacts/stage2_deployment_log.json: deployment-style Stage 2 log
- artifacts/stage2_shap_feature_importance.csv: Stage 2 global feature impact ranking
- artifacts/models/: saved preprocessors and trained models
- plots/: generated diagnostics and explainability charts

## How The System Works

### Stage 1: PD Estimator

Goal: learn $P(Default\mid x)$ from accepted loan records.

High-level flow:

1. Load accepted loan data and optimize memory footprint.
2. Create target using status mapping:
	- Bad: Charged Off, Default, Late (31-120 days)
	- Good: Fully Paid, Current
3. Drop leakage and administrative columns.
4. Apply missingness policy:
	- Drop columns with more than 90% missing
	- Add missingness flags for 50-90% missing columns
5. Impute:
	- Numeric: median
	- Categorical: mode
6. Outlier handling on annual income and DTI:
	- Clip at 1st and 99th percentiles
	- Robust scaling
7. Label-encode categorical features.
8. Train XGBoost classifier on GPU with class-imbalance weighting:

$$
scale\_pos\_weight = \frac{N_{good}}{N_{bad}}
$$

9. Optimize with Optuna (AUC objective).
10. Evaluate with confusion matrix, ROC-AUC, and calibration curve.
11. Generate SHAP global and local explanations.
12. Run persona-based stability experiments across multiple random seeds.

Outputs:

- stage1_classifier.json
- artifacts/pd_model_metrics.csv
- artifacts/preprocessing_audit.csv
- stability_results.csv
- explainability and validation plots in plots/

### Stage 2: Capital Allocator

Goal: predict raw affordable loan capacity and convert it into final risk-adjusted limit.

High-level flow:

1. Load same accepted loan dataset.
2. Define Stage 2 training population as Fully Paid loans.
3. Target variable: loan_amnt.
4. Reuse robust preprocessing approach (missingness policy, imputation, encoding, scaling).
5. Train XGBoost regressor on GPU.
6. Tune with Optuna (RMSE objective).
7. Evaluate with MAE, RMSE, R2, and cross-validated error.
8. Score test applicants with Stage 1 PD model.
9. Apply haircut logic:

$$
Final\_Limit = \max(0, Raw\_Capacity) \times (1 - PD)
$$

10. Generate Stage 2 SHAP global explainability (top factors affecting Raw Capacity).
11. Save simulation outputs, deployment metrics, and SHAP artifacts.

Outputs:

- stage2_regressor.json
- artifacts/stage2_allocator_metrics.csv
- artifacts/stage2_deployment_log.json
- artifacts/stage2_final_limit_simulation.csv
- artifacts/stage2_shap_feature_importance.csv
- stage2 parity/residual/distribution/SHAP plots in plots/

## ML Workflow Used For This Dataset

This is the exact workflow pattern used on accepted_loans.csv in this repo.

### A. Data and Target Design

1. Input source: accepted_loans.csv
2. Stage 1 target: default_flag from loan_status mapping
3. Stage 2 target: loan_amnt on Fully Paid subset

### B. Leakage and Governance

1. Remove post-origination payment and recovery fields.
2. Remove administrative identifiers and free-text metadata.
3. Persist preprocessing audit for traceability.

### C. Feature Engineering and Preprocessing

1. Parse and normalize rates/dates (int_rate, issue_year).
2. Missingness strategy:
	- Drop >90% missing columns
	- Missingness indicators for 50-90% missing columns
3. Numeric imputation (median) and categorical imputation (mode).
4. Outlier clipping and robust scaling for heavy-tailed financial variables.
5. Label encoding for categorical predictors.

### D. Modeling Strategy

1. Stage 1 model: XGBClassifier
	- Objective: binary:logistic
	- Metric: AUC
	- Class imbalance correction via scale_pos_weight
2. Stage 2 model: XGBRegressor
	- Objective: reg:squarederror
	- Metric: RMSE/MAE/R2
3. Hyperparameter search: Optuna TPE sampler
4. Validation:
	- Stage 1: holdout + 5-fold stratified CV AUC
	- Stage 2: holdout + 5-fold CV MAE/RMSE

### E. Explainability and Stability

1. SHAP summary and force plots for Stage 1.
2. SHAP global feature ranking and summary plot for Stage 2 capacity model.
3. Multi-seed persona stress test for Stage 1 robustness.
4. Saved stability logs and Stage 2 SHAP artifacts for reproducibility.

### F. Inference and Deployment Pattern

1. Load saved Stage 1 and Stage 2 models plus preprocessing bundles.
2. Transform new applicants with the saved bundles (no retraining).
3. Compute PD and raw capacity.
4. Return risk-adjusted final limit and haircut percentage.

## Inference Contract

Given applicant profile(s), the pipeline returns:

- PD: probability of default
- Raw_Capacity: unconstrained predicted limit
- Final_Limit: risk-adjusted limit after haircut
- Risk_Haircut_Pct: $PD \times 100$

This contract is demonstrated in the inference-only section of Stage 2 notebook using mock applicant profiles and saved artifacts.

## Reproducing The Pipeline

1. Run Stage 1 notebook to train and validate PD model artifacts.
2. Run Stage 2 notebook to train allocator and produce final limit simulation.
3. Use saved models in artifacts/models/ for inference-only scoring in notebook or service layer.

## Generated Artifacts Overview

- artifacts/pd_model_metrics.csv: Stage 1 AUC and run metadata
- artifacts/preprocessing_audit.csv: dropped/flagged columns and encoding audit
- stability_results.csv: multi-seed Stage 1 robustness results
- artifacts/stage2_allocator_metrics.csv: Stage 2 regression metrics and haircut summary
- artifacts/stage2_deployment_log.json: deployment-style metrics snapshot
- artifacts/stage2_final_limit_simulation.csv: sampled final limit distribution
- artifacts/stage2_shap_feature_importance.csv: Stage 2 global SHAP feature importance ranking
- artifacts/models/stage1_preprocessor_bundle.joblib: Stage 1 transform bundle
- artifacts/models/stage2_preprocessor_bundle.joblib: Stage 2 transform bundle
- stage1_classifier.json and stage2_regressor.json: trained model binaries

## Notes

- The current implementation is built on accepted loan data and is suitable for offline modeling and policy simulation.
- For production use, add strict schema validation, drift monitoring, periodic recalibration, and policy constraints (min/max limits, affordability caps, regulatory checks).
