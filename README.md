# CLARE: Credit Limit Allocation with Risk Estimation

This repository is now organized into separate Stage 1 and Stage 2 workstreams.

## Project Structure

- `stage1/`: Probability of Default (PD) model workflow (stable baseline)
- `stage2/`: Capacity allocator workflow (isolated for rebuild)
- `shared_assets/models/`: shared model and preprocessing bundles
- `shared_assets/plots/`: shared diagnostic and explainability plots
- `docs/`: migration notes, critique notes, and archived legacy docs
- `data/`: additional datasets
- `data/accepted_loans.csv`: primary dataset currently used by both stages

## Stage Separation

- Stage 1 remains the reference risk engine.
- Stage 2 is separated so it can be rebuilt from scratch with minimal coupling.
- Legacy markdown content was archived to reduce root-level documentation clutter.

## Quick Start

1. Open `stage1/test-stage-1.ipynb` for PD modeling work.
2. Open `stage2/test-stage-2.ipynb` for allocator transition and redesign.
3. Review `docs/MIGRATION_NOTES.md` before deleting or deduplicating old files.

## Safety

- Reorganization was done non-destructively first (copy before cleanup).
- Root-level original models and artifacts are still present until explicit cleanup approval.

