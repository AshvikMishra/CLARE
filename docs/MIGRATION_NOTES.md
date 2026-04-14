# Migration Notes

## Goal

Reorganize the repository into clear Stage 1 and Stage 2 workstreams without data loss.

## What Was Done (Non-Destructive)

1. Created new directories: `stage1/`, `stage2/`, `shared_assets/models/`, `shared_assets/plots/`, `docs/archive/`.
2. Copied notebooks into stage folders.
3. Copied models into `shared_assets/models/` and plots into `shared_assets/plots/`.
4. Copied stage-specific metrics/log outputs into `stage1/artifacts/` and `stage2/artifacts/`.
5. Updated copied notebook path constants to support stage-local execution.

## Path Mapping

### Stage 1 Notebook

- `Path("accepted_loans.csv")` -> `Path("../data/accepted_loans.csv")`
- `Path("plots")` -> `Path("../shared_assets/plots")`
- `Path("artifacts")` -> `Path("./artifacts")`

### Stage 2 Notebook

- `Path("accepted_loans.csv")` -> `Path("../data/accepted_loans.csv")`
- `Path("plots")` -> `Path("../shared_assets/plots")`
- `Path("artifacts")` -> `Path("./artifacts")`
- `STAGE1_MODEL_PATH` -> `../shared_assets/models/stage1_classifier.json`
- `STAGE2_MODEL_PATH` -> `../shared_assets/models/stage2_regressor.json`
- `STAGE1_BUNDLE_PATH` -> `../shared_assets/models/stage1_preprocessor_bundle.joblib`
- `STAGE2_BUNDLE_PATH` -> `../shared_assets/models/stage2_preprocessor_bundle.joblib`

## Safety Policy

- Originals remain in place until explicit cleanup approval.
- Archive legacy docs rather than deleting content.
- Validate notebook paths before deleting duplicates.

