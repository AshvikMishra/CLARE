# Stage 1 Deep Dive

Stage 1 in this project is the risk engine.  
It estimates Probability of Default (PD) for each applicant, and that PD later becomes the risk haircut used in Stage 2 final-limit allocation.

Core logic from test-stage-1.ipynb:

$$
\hat{p}(x) = P(\text{Default} \mid x)
$$

Class-imbalance correction used during model training:

$$
scale\_pos\_weight = \frac{N_{good}}{N_{bad}}
$$

Calibration objective monitored in evaluation:

$$
\hat{p}(x) \approx \Pr(\text{Default}=1 \mid \hat{p}(x)=p)
$$

---

### 1. What Stage 1 actually does

1. It builds a binary default target from accepted-loan status labels.
2. It trains an XGBoost classifier on GPU to estimate PD.
3. It validates discrimination and calibration, then produces explainability outputs.

So the Stage 1 architecture is:

$$
x \rightarrow \widehat{PD}(x)
$$

---

### 2. Data definition and target strategy

From test-stage-1.ipynb:

1. Input source is accepted_loans.csv.
2. Keep only statuses in a controlled good/bad set.
3. Build default_flag as:
   - Bad: Charged Off, Default, Late (31-120 days)
   - Good: Fully Paid, Current
4. Parse and normalize key fields (int_rate, issue_year, dti, annual_inc).

Why this matters: The target is directly tied to observed repayment outcomes, so Stage 1 learns rank-order credit risk from real portfolio behavior.

---

### 3. Leakage control and preprocessing system

All preprocessing primitives are in test-stage-1.ipynb.

Leakage and governance controls:
1. Drop post-origination payment/recovery fields.
2. Drop administrative/non-causal metadata fields.
3. Remove loan_status before training.

Missingness and feature preparation:
1. Drop columns with >90% missingness.
2. Add missingness flags for 50% to 90% missing columns.
3. Median-impute numeric columns.
4. Mode-impute categorical columns.
5. Clip annual_inc and dti at 1st and 99th percentiles.
6. Robust-scale processed outlier-sensitive columns.
7. Label-encode categorical features (with ordered priority for grade, sub_grade, home_ownership when present).

This creates a leakage-aware, model-ready matrix and preserves a preprocessing audit trail.

---

### 4. Model training and tuning path

Training logic is in test-stage-1.ipynb.

1. Split data 80/20 with stratification on default_flag.
2. Compute class weight using scale_pos_weight.
3. Run Optuna TPE tuning (25 trials) with AUC objective.
4. Use a capped tuning subset (300,000 rows max) for tractable optimization.
5. Train final XGBClassifier with best hyperparameters on training data.
6. Run 5-fold stratified CV AUC for robustness.

Config constants (random seed, trial count, tuning cap, GPU settings) are set in the notebook header.

---

### 5. Evaluation, calibration, and explainability

Stage 1 evaluation outputs include:

1. Confusion matrix (thresholded behavior at 0.5).
2. ROC curve and test ROC-AUC (ranking power).
3. Calibration curve (probability reliability).
4. SHAP summary plot (global drivers).
5. SHAP force plot on highest-risk test case (local driver decomposition).

These outputs make the model usable both for decisioning and model-risk review.

---

### 6. Stability experiment (seed and persona stress test)

A dedicated Stage 1 stress test reruns training across multiple split seeds and model seeds, with three persona configurations:

1. Specialist
2. Generalist
3. Conservative

For each run, the notebook logs train/test AUC, AUC gap, training time, and best iteration. It then computes:

$$
stability\_score = mean\_auc - 2 \cdot std\_auc
$$

This favors candidates that are both strong and stable, not just high on a single split.

---

### 7. Artifacts and outputs written

1. Trained PD model: stage1_classifier.json
2. Metrics snapshot: artifacts/pd_model_metrics.csv
3. Preprocessing audit: artifacts/preprocessing_audit.csv
4. Stability log: stability_results.csv
5. Diagnostic and explainability plots: plots/

---

### 8. How to interpret current Stage 1 results

From artifacts/pd_model_metrics.csv:

1. Test ROC-AUC: 0.7672
2. CV ROC-AUC mean: 0.7599
3. CV ROC-AUC std: 0.0021
4. Rows used: 2,245,134
5. Features used: 97

Practical read:
1. AUC around 0.76 to 0.77 indicates meaningful risk ranking power.
2. Low CV standard deviation indicates consistent out-of-sample behavior.
3. At this dataset scale, Stage 1 is robust enough to serve as the risk haircut backbone for Stage 2 allocation.