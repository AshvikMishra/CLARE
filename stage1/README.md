# Stage 1 - Probability of Default (PD)

This folder contains the Stage 1 credit risk model that estimates the probability a borrower defaults.

Core objective:

$$
\hat{p}(x) = P(\text{Default}=1 \mid x,\ t=0)
$$

The model is designed for origination-time decisioning and produces calibrated probabilities that can be passed into Stage 2.

## What Stage 1 Produces

- A leakage-safe binary PD model.
- Out-of-time validation metrics.
- Calibrated probabilities (sigmoid and isotonic).
- A feature availability audit for governance.
- Mock applicant scoring examples for business interpretation.

## Main Notebook

- `test-stage-1.ipynb`: end-to-end implementation of Stage 1.

## Step-by-Step Pipeline

1. **Load Raw Loan Data**
	- Read `../data/accepted_loans.csv`.
	- Standardize target source column (`loan_status`) for clean filtering.

2. **Define Strict Target Labels**
	- Keep only:
	  - `Fully Paid` -> 0
	  - `Charged Off` -> 1
	- Drop all ambiguous statuses to avoid label leakage.

3. **Apply Maturity Window**
	- Parse issue date to create `issue_date_parsed`.
	- Remove recent loans that have not had enough time to realize default outcomes.
	- This reduces right-censoring bias.

4. **Create Origination-Time Date Features**
	- Derive `issue_year` and `issue_month`.
	- These are allowed as origination-time predictors.

5. **Run Feature Availability Audit**
	- Build a governance table with:
	  - `Feature`
	  - `Available at Origination?`
	  - `Kept?`
	  - `Reason`
	- Explicitly drop post-origination and leakage-prone fields.

6. **Enforce Leakage Removal Rules**
	- Drop payment and performance outcomes (for example `total_pymnt`, `recoveries`).
	- Drop collection/recovery/settlement/hardship patterns.
	- Drop non-origination decision-derived fields (`funded_amnt`, `funded_amnt_inv`, `installment`).
	- Drop raw issue date fields from modeling inputs after split.

7. **Time-Aware Split (Out-of-Time)**
	- Sort by `issue_date_parsed`.
	- Use first 80% as train, last 20% as test.
	- Confirm both classes exist in both partitions.
	- No random split when valid time is available.

8. **Training-Only Sampling Cap**
	- If training rows exceed threshold, sample only the training set.
	- Use stratified sampling to preserve class balance.
	- Test set remains untouched.

9. **Build Model-Specific Preprocessing**
	- Logistic Regression pipeline:
	  - Numeric: median imputation + standard scaling
	  - Categorical: most-frequent imputation + one-hot encoding
	- XGBoost pipeline:
	  - Numeric: median imputation only (no scaling)
	  - Categorical: most-frequent imputation + one-hot encoding

10. **Handle Class Imbalance for XGBoost**
	 - Compute:
		$$
			ext{scale\_pos\_weight} = \frac{\#\text{negatives}}{\#\text{positives}}
		$$
	 - Pass this value into `XGBClassifier`.

11. **Train and Cross-Validate**
	 - Fit baseline Logistic Regression and raw XGBoost on train only.
	 - Run 5-fold `StratifiedKFold` CV on train data for robust AUC stability check.
	 - Report CV AUC mean and standard deviation.

12. **Calibrate Probabilities**
	 - Fit two calibrators on training data only:
		- Sigmoid
		- Isotonic
	 - Use `CalibratedClassifierCV(cv=5, ensemble=False)`.

13. **Evaluate on Out-of-Time Test Set**
	 - Report:
		- ROC AUC
		- Brier score
		- ECE (Expected Calibration Error, custom implementation)
	 - Compare Raw vs Sigmoid vs Isotonic.

14. **Visual Diagnostics**
	 - Calibration curve with low bin count (finance-friendly stability).
	 - Probability distribution histogram.
	 - ROC comparison chart for discrimination sanity check.

15. **Interpretation and Model Selection**
	 - Check overconfidence/underconfidence behavior.
	 - Check Brier and ECE improvements after calibration.
	 - Flag potential isotonic overfit behavior.
	 - Select best calibration method using ECE/Brier/AUC tie-break logic.

16. **Mock Candidate Scoring**
	 - Score five synthetic borrower profiles.
	 - Output raw PD, calibrated PD, and risk band.
	 - Demonstrates how model outputs map into real underwriting narratives.

## Key Outputs

- Model quality summary in notebook output (AUC, Brier, ECE).
- Feature availability audit table in notebook output.
- Calibration and distribution plots inline.
- Mock-candidate prediction table inline.

## Paths Used

- Data: `../data/accepted_loans.csv`
- Notebook: `./test-stage-1.ipynb`

## Operating Notes

- Stage 1 is the reference PD component for downstream allocation logic.
- Any future changes should preserve origination-time feature integrity and out-of-time validation discipline.

