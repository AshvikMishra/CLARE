# CLARE

Credit Limit Allocation and Risk Engine

## 1. Project Overview

This project is evolving from binary loan approval into a continuous capital allocation engine.

Primary outcome:
- Predict a safe loan allocation amount per borrower (regression), not just approve/reject.

Why this matters:
- A continuous allocation target is closer to real-world lending operations.
- It supports better risk pricing, portfolio construction, and capital efficiency.
- It creates more business value than a pure binary decision model.

## 2. Core Objective

Build a leakage-safe model that estimates how much capital can be allocated to each applicant while controlling downside risk.

Formally:
- Input: applicant-level, decision-time features.
- Output: safe allocation amount (USD).
- Model family: tree boosting first (XGBoost), with explainability via SHAP.

## 3. Data Reality Check (Critical)

### 3.1 Data Leakage Policy

Any post-origination or post-performance columns must be excluded from training.

Drop these columns before modeling:
- `loan_status`
- `balance`
- `paid_total`
- `paid_principal`
- `paid_interest`
- `paid_late_fees`
- `issue_month`
- `out_prncp` (if present)

Reason:
- These variables are generated after underwriting/servicing events.
- Including them would produce unrealistic offline performance and fail in production.

### 3.2 Features to Prioritize

Capacity to pay:
- `annual_income`
- `debt_to_income`
- `total_credit_limit`

Credit history and behavior:
- `delinq_2y`
- `earliest_credit_line`
- `num_historical_failed_to_pay`
- `inquiries_last_12m`

Current burden / utilization:
- `total_credit_utilized`
- `open_credit_lines`

Contextual categoricals:
- `homeownership`
- `loan_purpose`
- `emp_length`

Optional decision-time signals:
- `grade`, `sub_grade` (only if truly available at decision time in your target deployment workflow).

## 4. Defining the Regression Target

Do not use raw `loan_amount` as the target by itself.

If you do, the model mainly learns historical officer behavior instead of true risk-adjusted affordability.

### Approach A: Risk-Adjusted Target Engineering (Recommended Start)

Construct a synthetic safe allocation target from affordability.

Example framework:

1. Estimate safe monthly payment capacity:

$$
M_{safe} = \max\left(0, \alpha \cdot \frac{annual\_income \cdot (1 - dti)}{12}\right)
$$

Where:
- $dti$ is debt-to-income expressed as a fraction.
- $\alpha$ is a policy coefficient (for example 0.2 to 0.35, tuned with risk team input).

2. Convert payment capacity into maximum principal using an annuity relationship:

$$
L_{safe} = M_{safe} \cdot \frac{1 - (1+r)^{-T}}{r}
$$

Where:
- $r$ is monthly base interest rate.
- $T$ is term in months (for example, 36).

This yields a continuous target aligned with affordability and repayment capacity.

### Approach B: Two-Stage Risk + Allocation

1. Stage 1: predict default probability (classification).
2. Stage 2: predict allocation amount (regression), conditioned on risk threshold.

Use this when policy requires explicit probability-of-default gating before amount optimization.

## 5. End-to-End Workflow

This workflow consolidates and upgrades `ML_workflow.md`.

1. Define target
- Build `safe_loan_amount` (Approach A) or choose two-stage setup (Approach B).

2. EDA
- Distribution checks, missingness audit, outlier review, pairwise correlation, and leakage scan.

3. Data cleaning and leakage prevention
- Remove post-origination fields and ID-like non-generalizable artifacts.

4. Preprocessing
- Imputation:
    - Numeric: median/mean.
    - Categorical: "Unknown" placeholder where needed.
- Encoding:
    - One-hot for nominal categories (`homeownership`, `loan_purpose`, etc.).
    - Ordinal encoding for naturally ordered grades (`grade`, `sub_grade`).
- Scaling:
    - Not required for XGBoost.
    - Required for models like SVM, logistic regression, neural networks.

5. Train/test split
- Typical split: 80/20.
- Perform split before any resampling to avoid bleed.

6. Baseline model
- Train `DummyRegressor` (mean predictor).
- Every advanced model must beat this baseline.

7. Model training
- Main model: `XGBRegressor`.
- Benchmark models: linear regression, random forest, optionally CatBoost/LightGBM.

8. Hyperparameter tuning
- `RandomizedSearchCV` or `GridSearchCV`.
- Priority parameters:
    - `n_estimators`
    - `max_depth`
    - `learning_rate`
    - `min_child_weight`
    - `subsample`
    - `colsample_bytree`
    - `reg_lambda`, `reg_alpha`

9. Evaluation
- Primary metrics:
    - RMSE
    - MAE
- Secondary metrics:
    - $R^2$
    - Error by borrower segments (fairness and stability check).

10. Explainability (xAI)
- Use SHAP global summary + local force/waterfall explanations.
- Verify learned behavior aligns with credit policy intuition.

## 6. Modeling Math Foundation

### 6.1 XGBoost Objective

At boosting step $t$, optimize:

$$
Obj^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)
$$

Interpretation:
- $l(\cdot)$: prediction loss (for regression, typically squared error).
- $f_t$: the new tree that corrects prior residual errors.
- $\Omega(f_t)$: regularization penalty on tree complexity to reduce overfitting.

XGBoost uses first and second derivatives (gradient + hessian) to choose splits efficiently.

### 6.2 SHAP Values

SHAP value for feature $i$:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}\left[v(S \cup \{i\}) - v(S)\right]
$$

Interpretation:
- Measures each feature's marginal contribution across all feature coalitions.
- Produces additive local explanations: baseline prediction + feature contributions = final prediction.

## 7. Credit Risk Context (Portfolio Layer)

The allocation model can later be connected to classic risk equations:

Expected Loss:

$$
EL = PD \cdot LGD \cdot EAD
$$

Portfolio loss approximation:

$$
PL = \sum_i PD_i \cdot LGD_i \cdot EAD_i
$$

Portfolio expected return (lending lens):

$$
E[R] = \sum \text{Interest Income} - EL
$$

Future tail-risk extension:
- VaR and Expected Shortfall (CVaR) can be layered at portfolio optimization stage.

## 8. Implementation Plan and Milestones

Phase 1: Data audit and target design
- Finalize leakage-safe feature list.
- Implement `safe_loan_amount` target generation and document assumptions.

Phase 2: Baselines and first XGBoost
- Build baseline regressors.
- Train first leakage-safe `XGBRegressor`.

Phase 3: Tuning and validation
- Run hyperparameter search.
- Validate performance across risk segments and loan purposes.

Phase 4: Explainability and policy review
- Generate SHAP global and local reports.
- Confirm model behavior with business/risk logic.

Phase 5: Portfolio integration (future)
- Connect per-loan predictions to portfolio constraints.
- Add regime-aware stress testing and tail-risk controls.

## 9. Definition of Done

Minimum acceptable project state:
- Leakage columns removed and documented.
- Target formulation documented with assumptions.
- Baseline vs tuned XGBoost comparison available.
- MAE/RMSE reported on held-out test set.
- SHAP explanation artifacts generated.
- Reproducible training pipeline with fixed random seeds.

## 10. Immediate Next Build Tasks

1. Create a data dictionary marking each feature as decision-time vs post-origination.
2. Implement and validate `safe_loan_amount` target engineering script.
3. Build a single sklearn pipeline for imputation, encoding, and model training.
4. Train baseline + XGBoost and record RMSE/MAE.
5. Generate SHAP summary and top-20 feature impact report.