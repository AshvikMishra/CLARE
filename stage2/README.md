# Stage 2 - Capacity Allocator (Planned Rebuild)

This folder is the isolated Stage 2 workspace for redesigning allocation logic safely.

## Current Intent

- Keep historical Stage 2 artifacts for traceability.
- Rebuild Stage 2 from scratch without disrupting Stage 1.

## Current Formula Contract

$$
Final\_Limit = \max(0, \widehat{RawCapacity}) \times (1 - \widehat{PD})
$$

## Key Files

- `test-stage-2.ipynb`: stage-local notebook copy for transition and rebuild
- `artifacts/stage2_allocator_metrics.csv`: allocator metrics snapshot
- `artifacts/stage2_deployment_log.json`: deployment-style log output
- `artifacts/stage2_final_limit_simulation.csv`: simulation output
- `artifacts/stage2_shap_feature_importance.csv`: global feature-importance export

## Path Conventions

- Data path: `../data/accepted_loans.csv`
- Plot path: `../shared_assets/plots`
- Stage 2 artifact path: `./artifacts`
- Shared model bundle path: `../shared_assets/models`

## Rebuild Guardrails

- Avoid post-decision or target-proxy leakage fields.
- Preserve old artifacts during redesign for comparability.
- Validate with time-aware splits before claiming performance.

