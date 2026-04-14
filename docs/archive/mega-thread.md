# CLARE Mega Thread (Interview Master Document)

Generated: 2026-04-12

This document is the single-source, interview-ready consolidation of:
- README.md
- ML_workflow.md
- stage-1.md
- stage-1-inferences.md
- stage-2.md
- stage-2-inferences.md

And a repository skim to fill gaps using:
- Notebook source code in test-stage-1.ipynb and test-stage-2.ipynb
- Saved artifacts in artifacts/
- Saved model binaries and preprocessing bundles
- Dataset metadata and distribution checks
- Plot inventory

Everything below is organized to help you answer both business and technical interview questions.

---

## 1) What CLARE Is

CLARE = Credit Limit Allocation with Risk Estimation.

It is a two-stage ML pipeline built on Lending Club-style accepted-loan data:

1. Stage 1 predicts default probability (PD).
2. Stage 2 predicts raw unconstrained capacity (loan amount).
3. Final limit applies a risk haircut:

$$
Final\_Limit = \max(0, Raw\_Capacity) \times (1 - PD)
$$

Business meaning:
- High PD -> larger haircut -> lower final line.
- Low PD -> smaller haircut -> final line closer to raw capacity.
- It provides a continuous, risk-sensitive allocation policy instead of only a binary approve/reject.

---

## 2) Repository Scope and Purpose

Top-level assets and their role:

- accepted_loans.csv: core dataset used by both stages.
- test-stage-1.ipynb: Stage 1 PD model development, evaluation, SHAP, stability testing.
- test-stage-2.ipynb: Stage 2 allocator, Stage 1 integration, deployment-style outputs, SHAP.
- stage1_classifier.json: Stage 1 model binary at root.
- stage2_regressor.json: Stage 2 model binary at root.
- artifacts/: generated metrics, logs, simulation output, SHAP ranking, model bundles.
- artifacts/models/: deployment artifacts (model + preprocessor bundles).
- plots/: generated diagnostics and explainability images.
- data/extra_datasets/: additional CSV sources not used in primary two-stage training flow.

---

## 3) Data Inventory and Labeling Logic

### 3.1 Core dataset

accepted_loans.csv (verified):
- Rows: 2,260,701
- Columns: 151

First 25 columns:
- id
- member_id
- loan_amnt
- funded_amnt
- funded_amnt_inv
- term
- int_rate
- installment
- grade
- sub_grade
- emp_title
- emp_length
- home_ownership
- annual_inc
- verification_status
- issue_d
- loan_status
- pymnt_plan
- url
- desc
- purpose
- title
- zip_code
- addr_state
- dti

### 3.2 loan_status distribution in accepted_loans.csv (verified)

- Fully Paid: 1,076,751
- Current: 878,317
- Charged Off: 268,559
- Late (31-120 days): 21,467
- In Grace Period: 8,436
- Late (16-30 days): 4,349
- Does not meet the credit policy. Status:Fully Paid: 1,988
- Does not meet the credit policy. Status:Charged Off: 761
- Default: 40
- NA: 33

Unique statuses present: 10

### 3.3 Stage target definitions

Stage 1 target (default_flag):
- Bad = Charged Off, Default, Late (31-120 days)
- Good = Fully Paid, Current

Only these statuses are retained for Stage 1.

Stage 2 target:
- Train on Fully Paid only
- Target = loan_amnt

### 3.4 Extra datasets (repository skim)

- data/extra_datasets/loan.csv: 2,260,668 rows, 145 columns
- data/extra_datasets/loans_full_schema.csv: 10,000 rows, 55 columns
- data/extra_datasets/rejected_loans.csv: 27,648,741 rows, 9 columns

Note: Primary pipeline described in notebooks uses accepted_loans.csv.

---

## 4) End-to-End Pipeline Design

Architecture:

$$
x \rightarrow \widehat{PD}(x)\ \text{and}\ \widehat{RawCapacity}(x) \rightarrow \widehat{FinalLimit}(x)
$$

Final decision outputs:
- PD
- Raw_Capacity
- Final_Limit
- Risk_Haircut_Pct

Inference contract:

$$
Risk\_Haircut\_Pct = PD \times 100
$$

---

## 5) Stage 1 Deep Technical Walkthrough (PD Model)

### 5.1 Objective

Learn:

$$
\hat{p}(x) = P(Default \mid x)
$$

### 5.2 Imports and stack

Used in notebook:
- numpy, pandas
- matplotlib, seaborn
- optuna
- shap
- sklearn (imputation, metrics, CV/splits, preprocessing)
- xgboost.XGBClassifier

Stability cell additionally checks:
- cupy (optional)
- torch (optional)

### 5.3 Key constants (from notebook source)

- RANDOM_STATE = 42
- N_TRIALS = 25
- MAX_ROWS_FOR_TUNING = 300,000
- GPU config: tree_method="hist", device="cuda:0", max_bin=256

Status sets:
- BAD_STATUSES = {Charged Off, Default, Late (31-120 days)}
- GOOD_STATUSES = {Fully Paid, Current}

Leakage/admin dropped fields include:
- out_prncp, out_prncp_inv
- total_pymnt, total_pymnt_inv
- total_rec_prncp, total_rec_int, total_rec_late_fee
- recoveries, collection_recovery_fee
- last_pymnt_d, last_pymnt_amnt, next_pymnt_d
- last_credit_pull_d
- last_fico_range_high, last_fico_range_low
- hardship_flag, debt_settlement_flag
- id, member_id, url, desc, title, zip_code

### 5.4 Data prep and feature engineering

1. Load CSV and optimize memory:
   - integer downcast
   - float downcast
   - object -> category when unique ratio < 0.2

2. Parse/clean fields:
   - loan_status stripped string
   - issue_year from issue_d (%b-%Y)
   - int_rate parsed from percent strings
   - dti/annual_inc to numeric

3. Apply target filter and create default_flag.

4. Missingness policy:
   - Drop columns with missing ratio > 90%
   - Add flag columns for missing ratio in [50%, 90%]

5. Imputation:
   - Numeric median
   - Categorical mode

6. Outlier policy:
   - annual_inc and dti clipped to [1st, 99th] percentile
   - RobustScaler applied

7. Categorical encoding:
   - LabelEncoder
   - Priority encode order: grade, sub_grade, home_ownership, then remaining categoricals

### 5.5 Model training and tuning

Train/test:
- 80/20 split
- stratified by default_flag

Class imbalance correction:

$$
scale\_pos\_weight = \frac{N_{good}}{N_{bad}}
$$

Optuna objective:
- maximize validation ROC-AUC
- TPE sampler with seed
- tune on capped subset if needed

Stage 1 search space:
- n_estimators: 200 to 900
- max_depth: 3 to 10
- learning_rate: 0.01 to 0.30 (log)
- gamma: 0.0 to 10.0

Fixed params in tuning/final model:
- subsample = 0.85
- colsample_bytree = 0.85
- objective = binary:logistic
- eval_metric = auc
- random_state = 42
- n_jobs = -1
- tree_method = hist
- device = cuda:0
- max_bin = 256
- scale_pos_weight = computed ratio

### 5.6 Evaluation outputs

Generated in notebook:
- Confusion matrix plot
- ROC curve plot
- Calibration curve plot
- SHAP summary plot (global)
- SHAP force plot for highest-risk test case (local)

SHAP details:
- sample size = min(test_rows, 10,000)
- high risk test index = argmax(predicted PD)

### 5.7 Stage 1 saved artifacts

- stage1_classifier.json
- artifacts/pd_model_metrics.csv
- artifacts/preprocessing_audit.csv
- stability_results.csv
- plots in plots/

### 5.8 Verified Stage 1 metrics (artifacts/pd_model_metrics.csv)

- test_auc: 0.7671752223
- cv_auc_mean: 0.7599295882
- cv_auc_std: 0.0020624902
- scale_pos_weight: 6.7400723111
- rows_used: 2,245,134
- features_used: 97
- high_risk_test_index: 60,253
- optuna_best_auc: 0.7605154985

Interpretation:
- Good ranking performance (~0.76 to 0.77 AUC).
- Very low CV std -> stable performance across folds.

---

## 6) Stage 1 Stability and Persona Testing

From stability experiment cell in test-stage-1.ipynb:

Seeds:
- split seeds: [42, 123, 2024, 7, 88]
- model seeds: [42, 123, 2024, 7, 88]

Personas:

Specialist:
- learning_rate 0.01
- n_estimators 1000
- max_depth 5
- gamma 0.0
- subsample 1.0

Generalist:
- learning_rate 0.05
- n_estimators 500
- max_depth 4
- gamma 0.0
- subsample 1.0

Conservative:
- learning_rate 0.05
- n_estimators 500
- max_depth 3
- gamma 5.0
- subsample 0.8

Additional model settings:
- objective binary:logistic
- eval_metric auc
- early_stopping_rounds = 50
- tree_method hist
- device cuda:0
- max_bin 256
- colsample_bytree 0.85
- n_jobs 1

Stability score formula:

$$
stability\_score = mean\_auc - 2 \cdot std\_auc
$$

### 6.1 Full run table (stability_results.csv)

- 42/42 Specialist: train 0.763356, test 0.761590, gap 0.001766, 48.80s, best_iter 999
- 42/42 Generalist: train 0.766729, test 0.764565, gap 0.002164, 30.67s, best_iter 499
- 42/42 Conservative: train 0.761851, test 0.761713, gap 0.000138, 30.52s, best_iter 498
- 123/123 Specialist: train 0.763597, test 0.761016, gap 0.002582, 49.50s, best_iter 999
- 123/123 Generalist: train 0.766892, test 0.763674, gap 0.003219, 32.07s, best_iter 499
- 123/123 Conservative: train 0.762085, test 0.760993, gap 0.001092, 32.05s, best_iter 499
- 2024/2024 Specialist: train 0.763456, test 0.761373, gap 0.002083, 50.11s, best_iter 999
- 2024/2024 Generalist: train 0.766860, test 0.764244, gap 0.002616, 32.73s, best_iter 499
- 2024/2024 Conservative: train 0.761968, test 0.761391, gap 0.000577, 31.66s, best_iter 499
- 7/7 Specialist: train 0.763598, test 0.760565, gap 0.003032, 49.28s, best_iter 999
- 7/7 Generalist: train 0.767047, test 0.763379, gap 0.003667, 32.54s, best_iter 499
- 7/7 Conservative: train 0.762140, test 0.760406, gap 0.001734, 30.82s, best_iter 499
- 88/88 Specialist: train 0.763764, test 0.760328, gap 0.003435, 49.40s, best_iter 999
- 88/88 Generalist: train 0.767198, test 0.763411, gap 0.003787, 32.71s, best_iter 499
- 88/88 Conservative: train 0.762224, test 0.760187, gap 0.002037, 31.86s, best_iter 499

### 6.2 Aggregated summary (computed from CSV)

By stability score:
1. Generalist: mean_auc 0.763855, std 0.000527, stability_score 0.762801
2. Specialist: mean_auc 0.760974, std 0.000530, stability_score 0.759914
3. Conservative: mean_auc 0.760938, std 0.000644, stability_score 0.759651

Training speed summary:
- Specialist mean ~49.42s
- Generalist mean ~32.14s
- Conservative mean ~31.38s

Overall test AUC range across all runs:
- min 0.7601865466
- max 0.7645648079

---

## 7) Stage 1 Business Inference Narrative (from stage-1-inferences.md)

Documented interpretation includes:

- Class imbalance handled with scale_pos_weight ~6.74.
- Reported confusion-matrix framing:
  - False Positives: 79,219
  - False Negatives: 18,348
  - True Positives: 39,520
- Emphasis on recall (~68.3%) and conservative risk capture.
- Precision discussed as ~33.3% and treated as acceptable in lender-risk context.

Lender dilemma framing:
- Missing defaults (FN) is treated as costlier than rejecting good borrowers (FP).
- Financial scenario in doc (illustrative assumptions):
  - +$2,000 per good loan
  - -$8,000 per charged-off loan
- Claimed net effect in this framing:
  - ~+$316M from caught defaults
  - ~-$158M from missed good loans
  - Net positive ~+$158M

Interview interpretation:
- Stage 1 is optimized for risk containment, not generic classification accuracy.

---

## 8) Stage 2 Deep Technical Walkthrough (Capital Allocator)

### 8.1 Objective

Predict unconstrained capacity first, then haircut with PD.

$$
Final\_Limit = \max(0, \widehat{RawCapacity}) \times (1 - \widehat{PD})
$$

### 8.2 Key constants (from notebook source)

- RANDOM_STATE = 42
- N_TRIALS_STAGE2 = 20
- MAX_ROWS_FOR_TUNING = 250,000
- MAX_ROWS_STAGE1_TRAIN = 600,000

Priority features:
- annual_inc
- dti
- emp_length
- total_rev_hi_lim

Model/bundle artifact paths:
- stage1_classifier.json
- stage2_regressor.json
- artifacts/models/stage1_classifier.json
- artifacts/models/stage2_regressor.json
- artifacts/models/stage1_preprocessor_bundle.joblib
- artifacts/models/stage2_preprocessor_bundle.joblib

### 8.3 Preprocessor bundle design

Functions:
- build_preprocessor_bundle(...)
- transform_with_bundle(...)
- transform_for_inference(...)

Bundle stores:
- drop_cols
- flag_cols
- numeric_cols
- categorical_cols
- num_imputer
- encoders (with fill_value + mapping)
- scaler
- feature_cols (ordered with priority features first)
- target_col

Transform behavior:
- Drops training-defined drop_cols
- Adds missingness flags
- Adds absent numeric/categorical columns at inference
- Numeric imputation + scaling from saved objects
- Categorical mapping with unknown fallback to -1
- Adds any missing trained features as 0
- Returns exact training feature order

### 8.4 Stage 2 training flow

1. Build stage2_df = Fully Paid only, drop NA loan_amnt.
2. Drop leakage/admin columns.
3. Split stage2_model_df into train/test (80/20, random_state=42).
4. Build Stage 2 bundle on train split.
5. Transform test split with same bundle.
6. Fit Optuna-tuned XGBRegressor.

Stage 2 search space:
- n_estimators: 300 to 1000
- max_depth: 3 to 10
- learning_rate: 0.01 to 0.20 (log)
- gamma: 0.0 to 8.0

Fixed params:
- subsample = 0.85
- colsample_bytree = 0.85
- objective = reg:squarederror
- tree_method = hist
- device = cuda:0
- max_bin = 256
- n_jobs = -1

Validation:
- Holdout MAE/RMSE/R2
- 5-fold KFold CV MAE/RMSE (n_jobs=1)

### 8.5 Stage 1 integration in Stage 2 notebook

- Transform stage2_test_df using Stage 1 bundle.
- Predict PD with Stage 1 model.
- Clip Stage 2 predictions at zero.
- Compute final limits with (1 - PD).
- Compute average portfolio haircut = mean(PD) * 100.
- Save 1,000-row simulation sample.

### 8.6 Stage 2 explainability

- SHAP TreeExplainer on Stage 2 model
- Sample size = min(5000, len(test_matrix))
- Compute mean absolute SHAP per feature
- Save:
  - plots/stage2_shap_global_importance_top15.png
  - plots/stage2_shap_summary_plot.png
  - artifacts/stage2_shap_feature_importance.csv

### 8.7 Stage 2 saved outputs

- stage2_regressor.json
- artifacts/stage2_allocator_metrics.csv
- artifacts/stage2_deployment_log.json
- artifacts/stage2_final_limit_simulation.csv
- artifacts/stage2_shap_feature_importance.csv

---

## 9) Verified Stage 2 Metrics and Simulation

### 9.1 Current metric snapshot (artifact files)

From artifacts/stage2_allocator_metrics.csv and stage2_deployment_log.json:

- stage2_test_mae: 47.7696654082
- stage2_test_rmse: 237.0047894958
- stage2_test_r2: 0.9992570400
- stage2_cv_mae_mean: 48.2501311445
- stage2_cv_rmse_mean: 231.3673429785
- average_risk_haircut_pct: 45.2854232788

### 9.2 Simulation summary (artifacts/stage2_final_limit_simulation.csv)

Rows: 1,000

PD:
- mean 0.454679
- min 0.017714
- max 0.856694

Raw_Capacity:
- mean 14,222.39
- min 996.48
- max 40,069.27

Final_Limit:
- mean 7,771.14
- min 468.37
- max 35,680.06

Final_Limit quantiles:
- p10 2,122.05
- p25 3,411.20
- p50 6,187.68
- p75 10,297.87
- p90 16,433.91

Interpretation:
- Median final line is ~6.2k in saved simulation sample.
- Wide spread supports continuous allocation (not fixed tiers).

---

## 10) Stage 2 Business Inference Narrative (from stage-2-inferences.md)

Document-level points:
- Very high fit quality ($R^2 \approx 0.9993$) interpreted as near-deterministic rule capture.
- Haircut method reduces exposure continuously rather than binary decisioning.
- Average haircut ~45.29% is materially conservative.
- Test/CV closeness framed as deployment-stable.

Document also flags high $R^2$ as potentially suspicious in generic ML but plausible if policy boundaries are deterministic.

---

## 11) Preprocessing Audit Details (artifacts/preprocessing_audit.csv)

### 11.1 Dropped columns (>90% missing)

- annual_inc_joint
- dti_joint
- verification_status_joint
- revol_bal_joint
- sec_app_fico_range_low
- sec_app_fico_range_high
- sec_app_earliest_cr_line
- sec_app_inq_last_6mths
- sec_app_mort_acc
- sec_app_open_acc
- sec_app_revol_util
- sec_app_open_act_il
- sec_app_num_rev_accts
- sec_app_chargeoff_within_12_mths
- sec_app_collections_12_mths_ex_med
- sec_app_mths_since_last_major_derog
- hardship_type
- hardship_reason
- hardship_status
- deferral_term
- hardship_amount
- hardship_start_date
- hardship_end_date
- payment_plan_start_date
- hardship_length
- hardship_dpd
- hardship_loan_status
- orig_projected_additional_accrued_interest
- hardship_payoff_balance_amount
- hardship_last_payment_amount
- debt_settlement_flag_date
- settlement_status
- settlement_date
- settlement_amount
- settlement_percentage
- settlement_term

### 11.2 Flagged columns (50% to 90% missing)

- mths_since_last_delinq
- mths_since_last_record
- mths_since_last_major_derog
- mths_since_recent_bc_dlq
- mths_since_recent_revol_delinq

### 11.3 Outlier columns processed

- annual_inc
- dti

### 11.4 Label-encoded columns

- addr_state
- application_type
- disbursement_method
- earliest_cr_line
- emp_length
- emp_title
- grade
- home_ownership
- initial_list_status
- issue_d
- purpose
- pymnt_plan
- sub_grade
- term
- verification_status

---

## 12) Saved Bundle Internals (joblib artifacts)

Stage 1 bundle (artifacts/models/stage1_preprocessor_bundle.joblib):
- drop_cols_count: 36
- flag_cols_count: 5
- numeric_cols_count: 82
- categorical_cols_count: 15
- feature_cols_count: 97
- target_col: default_flag

Stage 2 bundle (artifacts/models/stage2_preprocessor_bundle.joblib):
- drop_cols_count: 36
- flag_cols_count: 19
- numeric_cols_count: 95
- categorical_cols_count: 15
- feature_cols_count: 110
- target_col: loan_amnt

Common encoded categorical features (15):
- term, grade, sub_grade, emp_title, emp_length, home_ownership,
  verification_status, issue_d, pymnt_plan, purpose, addr_state,
  earliest_cr_line, initial_list_status, application_type, disbursement_method

---

## 13) Model Artifact Integrity and Metadata

Root and artifacts/models copies are byte-identical (verified by SHA-256):

Stage 1 model:
- root path size: 3,004,245 bytes
- artifacts/models size: 3,004,245 bytes
- sha256: ab86eee7c68ab5221941cf4552d6d7303650b62e76fed695d078dd4dcde948bc
- identical: true

Stage 2 model:
- root path size: 6,156,907 bytes
- artifacts/models size: 6,156,907 bytes
- sha256: ec1e00555734e7d45c23a974963c5d5de8d8981fa87244da045bc37028c90d4f
- identical: true

XGBoost JSON model metadata:
- Stage 1 objective: binary:logistic, num_feature: 97, trees: 450
- Stage 2 objective: reg:squarederror, num_feature: 110, trees: 987

---

## 14) SHAP Feature Impact (Stage 2)

Top 15 global factors by mean absolute SHAP (artifacts/stage2_shap_feature_importance.csv):

1. funded_amnt: 6302.7495
2. funded_amnt_inv: 489.73026
3. installment: 226.17477
4. term: 77.94812
5. int_rate: 34.81199
6. issue_year: 19.823635
7. annual_inc: 13.450969
8. sub_grade: 8.359825
9. emp_title: 8.3574295
10. fico_range_low: 7.1147475
11. issue_d: 6.5963945
12. verification_status: 4.916666
13. grade: 3.2968206
14. revol_bal: 3.283826
15. earliest_cr_line: 2.5048048

Important interview note:
- funded_amnt and funded_amnt_inv dominate Stage 2 signal.
- This is powerful predictive signal but also a governance discussion point (possible policy echo/proxy effects).

---

## 15) Plot Inventory (verified)

16 plots currently in plots/:
- calibration_curve.png
- confusion_matrix.png
- default_rate_per_year.png
- dti_distribution_by_status.png
- final_limit_distribution_simulation.png
- int_rate_distribution_by_status.png
- roc_auc_curve.png
- shap_force_plot_high_risk.png
- shap_summary_plot.png
- stage1_persona_test_auc_boxplot.png
- stage1_training_time_vs_auc_scatter.png
- stage2_parity_plot.png
- stage2_residual_plot.png
- stage2_shap_global_importance_top15.png
- stage2_shap_summary_plot.png
- top_numeric_correlation_heatmap.png

---

## 16) Inference-Only Behavior and Mock Results

Stage 2 notebook defines get_risk_adjusted_limit(applicant_data) that accepts:
- dict
- pandas Series
- pandas DataFrame

It always returns:
- PD
- Raw_Capacity
- Final_Limit
- Risk_Haircut_Pct

Verified mock run (6 profiles, no retraining) produced:

- profile 1: PD 0.4413, raw 12,215.32, final 6,825.19, haircut 44.13%
- profile 2: PD 0.6851, raw 11,956.61, final 3,765.27, haircut 68.51%
- profile 3: PD 0.2958, raw 12,048.33, final 8,484.67, haircut 29.58%
- profile 4: PD 0.4653, raw 12,281.23, final 6,566.81, haircut 46.53%
- profile 5: PD 0.7296, raw 11,954.18, final 3,232.80, haircut 72.96%
- profile 6: PD 0.3240, raw 11,785.26, final 7,967.00, haircut 32.40%

---

## 17) ML Workflow Philosophy (from ML_workflow.md)

The workflow notes in ML_workflow.md emphasize:

1. Collect data (features + labels).
2. Prep:
   - Imputation (.fillna mean / 0.0)
   - Outlier identification and manual review
   - One-hot encoding for text features
   - Drop useless columns
   - Normalize large-range features
   - Optional unsupervised pass for pseudo-labels/new features
3. Split data.
4. Train models (XGB / SVM / Logistic Regression / Random Forest / Neural Networks).
5. Ensemble options (bagging, boosting, voting).
6. Hyperparameter tuning (GridSearchCV noted in this file).
7. Inference.
8. Explainability via SHAP.

Also explicitly references cupy and sklearn.

Note:
- This file is a general methodology sketch and not a strict verbatim run log.

---

## 18) Metric Drift and Versioning Notes (Important for Interviews)

There is a metrics mismatch across docs vs current artifact snapshot:

README.md and stage-2.md mention:
- MAE 46.62
- RMSE 235.37
- R2 0.9993
- CV MAE 47.69
- CV RMSE 230.49

Current artifact files report:
- MAE 47.77
- RMSE 237.00
- R2 0.999257
- CV MAE 48.25
- CV RMSE 231.37

How to explain this in interview:
- The pipeline has been run multiple times/versions.
- Narrative docs contain an earlier metric snapshot.
- artifacts/stage2_allocator_metrics.csv and artifacts/stage2_deployment_log.json are the authoritative latest saved run in this repo state.

---

## 19) Risks, Assumptions, and Governance Talking Points

From docs + repository skim, strongest discussion points are:

1. Stage 2 selection bias:
   - Trained on Fully Paid only.
   - Good for policy-style capacity signal but may underrepresent riskier repayment dynamics.

2. Very high Stage 2 fit:
   - R2 near 1 can mean deterministic policy reconstruction.
   - Also warrants leakage/proxy checks.

3. Feature governance:
   - funded_amnt and funded_amnt_inv dominate Stage 2 SHAP.
   - Interviewers may ask whether these features encode historical policy decisions.

4. Accepted-only dataset:
   - Rejected applicants are not part of core training set.
   - Limits causal claims about full applicant population.

5. Unknown category handling:
   - Unseen categorical values mapped to -1.
   - Practical for deployment, but should be monitored for drift.

6. Production controls listed in README Notes:
   - schema validation
   - drift monitoring
   - periodic recalibration
   - policy constraints (min/max limits, affordability, regulatory checks)

---

## 20) Reproduction Runbook

As documented and validated by notebook flow:

1. Run test-stage-1.ipynb to train/evaluate PD and generate Stage 1 artifacts.
2. Run test-stage-2.ipynb to train allocator, integrate Stage 1 risk, and write Stage 2 artifacts.
3. Use saved model + bundle artifacts for inference-only scoring without retraining.

Primary dependencies implied by notebook imports:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- optuna
- shap
- joblib
- cupy (optional in stability memory utilities)
- torch (optional in stability memory utilities)

---

## 21) Interview Q and A Bank

1) What business problem does CLARE solve?
- It allocates credit limits using both capacity and risk, reducing expected losses via PD haircuts.

2) Why two stages instead of one?
- Stage 1 captures default risk, Stage 2 captures affordability/capacity; combining them separates risk from capacity and improves policy transparency.

3) What is the core formula?
- Final_Limit = max(0, Raw_Capacity) * (1 - PD).

4) How is default labeled?
- Bad: Charged Off, Default, Late (31-120 days). Good: Fully Paid, Current.

5) How is class imbalance handled in Stage 1?
- scale_pos_weight = N_good / N_bad in XGBoost.

6) Why include calibration curve in Stage 1?
- To verify predicted PD values align with observed default frequencies, not only ranking quality.

7) What is Stage 1 performance?
- Test AUC 0.7672; CV mean 0.7599; CV std 0.0021.

8) What is Stage 2 performance?
- Test MAE 47.77; RMSE 237.00; R2 0.999257; CV MAE 48.25; CV RMSE 231.37.

9) Why is Stage 2 R2 so high?
- Likely strong deterministic policy signal in historical approved amounts; still requires leakage/proxy governance checks.

10) What is average portfolio haircut?
- 45.29%.

11) Which features drive Stage 2 most?
- funded_amnt, funded_amnt_inv, installment, term, int_rate, then issue_year and annual_inc.

12) What anti-leakage steps are used?
- Drops post-origination payment/recovery/fico-pull features and admin IDs/text metadata.

13) What missing-data policy is used?
- Drop >90% missing columns, add missingness flags for 50-90% missing columns, impute numeric median and categorical mode.

14) How are outliers treated?
- annual_inc and dti clipped at 1st/99th percentiles, then robust scaled.

15) How are categoricals handled?
- Label encoding with saved mapping dictionaries in preprocessor bundles.

16) How is inference reproducible?
- Saved bundle enforces schema, imputation, scaling, encoding, ordering, and unknown-category fallback.

17) What happens on unseen categories?
- Encoded as -1.

18) What data scale is this trained on?
- Stage 1 modeling rows: 2,245,134 with 97 features.

19) What is Stage 2 feature dimensionality?
- 110 features after Stage 2 bundle transforms.

20) How is Stage 1 robustness tested?
- Multi-seed persona experiments with stability score = mean_auc - 2*std_auc.

21) Which Stage 1 persona won by stability score?
- Generalist.

22) Why might Conservative still be attractive?
- Lowest overfit tendency (smaller train-test AUC gap), good for risk-averse deployment posture.

23) What is the deployment contract output?
- PD, Raw_Capacity, Final_Limit, Risk_Haircut_Pct.

24) Can CLARE run inference without retraining?
- Yes, using saved model JSON files and joblib preprocessor bundles.

25) What governance improvements are recommended by docs?
- Schema validation, drift monitoring, recalibration, min/max policy constraints, affordability and regulatory checks.

26) Why train Stage 2 only on Fully Paid?
- To model capacity from successful repayment behavior; tradeoff is potential selection bias.

27) What is the practical effect of high Stage 1 recall?
- More default capture at the expense of rejecting some good borrowers; this can be economically rational if default loss is much larger than missed interest.

28) What are the two strongest explainability artifacts?
- Stage 1 SHAP summary + force plot, Stage 2 SHAP global ranking + summary plot.

29) Where are the deployment metrics stored?
- artifacts/stage2_deployment_log.json and artifacts/stage2_allocator_metrics.csv.

30) Are root model files and artifacts/models files consistent?
- Yes, verified identical by file size and SHA-256.

---

## 22) 60-Second Interview Pitch

CLARE is a production-style, two-stage credit limit system on accepted loan data. Stage 1 predicts PD with strong and stable AUC (~0.76 to 0.77), and Stage 2 predicts raw loan capacity with high fit, then applies a continuous PD haircut to produce final limits. The pipeline is artifact-driven, reproducible, and explainable via SHAP in both stages. It includes leakage controls, missingness governance, robust preprocessing bundles for inference, persona stability testing, and deployment-style outputs. Current saved results show an average portfolio haircut of ~45.29%, with robust metric consistency and full artifact traceability.

---

## 23) Last-Minute Recall Checklist

- Formula: Final = max(0, Raw) * (1 - PD)
- Stage 1 AUC: 0.7672 test, 0.7599 CV mean
- Stage 2: MAE 47.77, RMSE 237.00, R2 0.999257
- Avg haircut: 45.29%
- Stage 1 rows/features: 2,245,134 / 97
- Stage 2 features: 110
- Stability winner: Generalist
- Key Stage 2 SHAP #1 feature: funded_amnt
- Main governance caveat: very high Stage 2 fit + policy signal risk
- Inference outputs: PD, Raw_Capacity, Final_Limit, Risk_Haircut_Pct
