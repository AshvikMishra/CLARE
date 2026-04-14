# Stage 2 Deep Dive

Stage 2 in this project is a capital allocator, not a risk classifier.  
It estimates a borrower’s unconstrained affordable amount first, then applies Stage 1 risk as a haircut.

Core logic from test-stage-2.ipynb:

$$
Final\_Limit = Raw\_Capacity \times (1 - PD)
$$

In implementation, Raw Capacity is clipped to be non-negative before haircut:

$$
Final\_Limit = \max(0, \widehat{RawCapacity}) \times (1 - \widehat{PD})
$$

---

### 1. What Stage 2 actually does

1. It learns a regression model for loan capacity (target = loan_amnt) on Fully Paid loans only, defined in test-stage-2.ipynb.
2. It trains or loads a Stage 1 PD model to score the same applicants with default risk in test-stage-2.ipynb.
3. It combines both outputs into a final risk-adjusted limit in test-stage-2.ipynb.
4. It adds SHAP-based global explainability to show which features drive Raw Capacity predictions the most.

So the architecture is:

$$
x \rightarrow \widehat{PD}(x),\ \widehat{RawCapacity}(x)\ \rightarrow\ \widehat{FinalLimit}(x)
$$

---

### 2. Data definition and cohort strategy

From test-stage-2.ipynb:

1. Full dataset loaded from accepted_loans.csv.
2. Stage 1 cohort: statuses in bad + good sets to build default_flag.
3. Stage 2 cohort: loan_status == Fully Paid only.
4. Stage 2 target: loan_amnt.

Why this matters: Stage 2 is trained on “successful repayment population,” which gives a policy-like capacity signal, but it can introduce selection bias because defaults are excluded from the Stage 2 target-building cohort.

---

### 3. Preprocessing system (most important implementation detail)

All preprocessing primitives are in test-stage-2.ipynb.  
This notebook uses a reusable preprocessor bundle pattern.

Bundle creation (build_preprocessor_bundle):
1. Drops columns with >90% missing.
2. Adds missingness flags for 50% to 90% missing columns.
3. Median-imputes numeric features.
4. Mode-imputes categorical features.
5. Label-encodes categoricals and stores mapping dictionaries.
6. Robust-scales numeric columns.
7. Reorders features with priority fields first (annual_inc, dti, emp_length, total_rev_hi_lim), if present.

Bundle application (transform_with_bundle / transform_for_inference):
1. Enforces the same feature schema used at training.
2. Adds absent numeric/categorical columns when scoring.
3. Encodes unseen categories as -1 fallback.
4. Applies stored imputer + scaler + encoder mappings.
5. Returns columns in exact trained order.

This is what makes inference reproducible and compatible with training artifacts.

---

### 4. Model training and tuning path

Training logic is in test-stage-2.ipynb.

1. Stage 2 train/test split is random 80/20 (regression split, no stratification).
2. Optuna tuning uses a tuning subset capped at 250,000 rows.
3. Objective minimizes validation RMSE.
4. Search space includes n_estimators, max_depth, learning_rate, gamma.
5. Final XGBoost regressor is retrained on full Stage 2 train matrix with best params.
6. 5-fold CV is run for MAE and RMSE on tuning matrix.

Config and constants are set in test-stage-2.ipynb.

---

### 5. Integration with Stage 1 PD and portfolio simulation

Combination logic is in test-stage-2.ipynb.

1. Stage 1 bundle transforms Stage 2 test applicants for PD scoring.
2. Stage 2 model predicts Raw Capacity.
3. Raw Capacity is clipped at zero.
4. Final Limit applies $(1 - PD)$ haircut.
5. A 1,000-row simulation sample is saved for distribution analysis.

Outputs written:
1. Metrics file: stage2_allocator_metrics.csv
2. Deployment log: stage2_deployment_log.json
3. Final limit simulation: stage2_final_limit_simulation.csv

---

### 6. Explainability (SHAP) for Stage 2

Stage 2 now includes a final SHAP explainability block in test-stage-2.ipynb.

What it does:
1. Builds a SHAP explainer on the trained Stage 2 regressor.
2. Samples up to 5,000 rows from the Stage 2 test feature matrix.
3. Computes SHAP values for Raw Capacity predictions.
4. Ranks features by mean absolute SHAP value (global impact).
5. Prints the top factors affecting Raw Capacity.
6. Saves two plots:
	- stage2_shap_global_importance_top15.png
	- stage2_shap_summary_plot.png
7. Saves full ranked importance table:
	- stage2_shap_feature_importance.csv

Why this matters: This closes the interpretability gap for Stage 2 by showing exactly which variables are driving the allocator outputs.

---

### 7. Inference contract (deployment-style usage)

The callable scoring path is in test-stage-2.ipynb, with an inference-only mock demo in test-stage-2.ipynb.

Input accepted:
1. dict
2. Series
3. DataFrame

Returned fields:
1. PD
2. Raw_Capacity
3. Final_Limit
4. Risk_Haircut_Pct

This is effectively your service contract.

---

### 8. How to interpret current Stage 2 results

From stage2_allocator_metrics.csv and stage2_deployment_log.json:

1. Test MAE: 46.62
2. Test RMSE: 235.37
3. Test $R^2$: 0.9993
4. CV MAE mean: 47.69
5. CV RMSE mean: 230.49
6. Average risk haircut: 45.29%

Practical read:
1. Capacity predictions are very tight on this dataset.
2. The portfolio-level PD haircut is materially large, so Stage 1 meaningfully compresses exposure.
3. SHAP rankings identify which factors most influence Raw Capacity, improving transparency for policy review.
4. Very high $R^2$ is strong, but still worth governance checks for hidden proxy/leakage features in future iterations.