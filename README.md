# CLARE: Credit Decision System Roadmap

## Status (At a Glance)

- Stage 1 complete: leakage-safe, calibrated Probability of Default (PD) model.
- Current notebook: `stage1/test-stage-1.ipynb`.
- Current data: `data/accepted_loans.csv`.

## Stage 1 Deliverable

- Clean PD model with out-of-time validation.
- Core metrics: AUC, Brier, calibration quality.
- Output used as risk input for downstream decisioning.

## What Comes Next (Concise Plan)

1. Reject inference
- Debias PD by incorporating rejected applicants.
- Compare baseline vs corrected model on AUC, calibration, and PD shift.

2. Approval policy from PD
- Convert PD into approve/reject decisions using expected loss:
	$$EL = PD \times LGD \times EAD$$
- Approve when risk is below policy threshold.

3. Profit-risk optimization
- Evaluate PD cutoffs by approval rate, default rate, and expected profit:
	$$Profit = Interest - (PD \times LGD \times EAD)$$

4. Loan allocation engine
- Replace "predict loan amount" with policy optimization.
- For each applicant, choose the highest loan size satisfying risk constraints.

5. Unified decision engine
- Pipeline: Applicant features -> PD -> Risk -> Approve/Reject -> Loan amount.

6. Portfolio evaluation
- Report portfolio-level default rate, approval rate, and expected profit.

## Planned Notebook Flow

1. Notebook 1: Clean PD model (done).
2. Notebook 2: Reject inference.
3. Notebook 3: Decision policy and threshold tuning.
4. Notebook 4: Allocation engine and portfolio simulation.

