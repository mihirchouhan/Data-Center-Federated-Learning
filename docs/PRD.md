# Research requirements and scope

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Problem, motivation and stakeholders

Node power forecasting may support later operational planning. Missing telemetry, heterogeneous nodes and temporal dependence require controlled evaluation, rather than an assumed FL advantage. Forecast accuracy alone is not a control-policy outcome. Stakeholders are the researcher, supervisor/examiners and college-server administrator; a future operator deployment needs a separate specification.

## Objectives and research questions

- [IMPLEMENTED] Compare GRU FedAvg and matched centralized SGD with persistence on chronological power-forecasting windows.
- [IMPLEMENTED] Study enlarged forward-fill training pools under fixed processed counts and common observed holdout windows.
- [PLANNED] Determine whether 1/6/24-hour history changes errors after coverage, normalization and budgets are controlled.
- [PARTIALLY IMPLEMENTED] Define power/temperature forecasting with observed joint targets and output-balanced validation; real execution [BLOCKED].
- [PLANNED] Isolate environmental information on identical windows and evaluate a distinct forecast-informed decision policy.

Completed v5 covers known compute nodes from one M100 facility, January–February 2022; 36 ten-minute inputs predict six future power-bin means. Phase 2 retains original train/validation cuts and reserves March as a later retrospective test period. Its actual coverage and eligible clients are [UNKNOWN].

## Requirements and acceptance

| ID | Requirement | Status / acceptance evidence |
| --- | --- | --- |
| R1 | Preserve node mapping, UTC identities, targets and chronology | [IMPLEMENTED] v5 contract, `SplitData.batch` |
| R2 | No target imputation, future-input leakage or gap bridging | [IMPLEMENTED] observed-power checks, causal ages and window construction |
| R3 | Training-only client capabilities; coordinator validation | [IMPLEMENTED] `TrainingRegistry.local` / `LocalData`; no OS isolation |
| R4 | Paired schedules, initialization, processed budgets and optimizer settings | [IMPLEMENTED] sampling/row receipts; trajectories still differ |
| R5 | Validate update counts/names/shapes/dtypes/finiteness and fixed buffers | [IMPLEMENTED] `GlobalModelState` / `aggregate_fedavg` |
| R6 | Validation selection then immutable all-seed test lock | [IMPLEMENTED] v5 receipts and lock v2 |
| R7 | Physical-unit pooled/per-node/macro MAE/RMSE and persistence | [IMPLEMENTED] saved outputs; all seeds and sample SD |
| R8 | No-clobber outputs, checkpoint provenance, bounded data access | [IMPLEMENTED] stage-specific safeguards; not a guarantee against privileged tampering |
| R9 | Real two-output data/scaler/model/test adapters | [PARTIALLY IMPLEMENTED] pure contracts only; execution [BLOCKED] |
| R10 | Preserve failed checks and historical artifacts | [IMPLEMENTED] investigation records; old preflight stays FAILED |

Non-functional requirements: explicit units, finite arrays, immutable evidence, reproducible seeded plans, recorded software/device/resource scope, failure preservation, focused tests and no silent CPU fallback for locked v5. Exact cross-hardware bit identity is not promised; local hashes are not independent signatures.

## Non-goals and constraints

No completed energy controller, what-if simulator, carbon-emission predictor, measured savings, PUE/SLA optimization or fully privacy-preserving deployment is demonstrated. No physical multi-data-center federation, REST application or account system is implemented. Do not infer a general ranking of learning paradigms from this benchmark, or assume 24-hour history improves accuracy.

Accept new experiments only after verified inputs/units, approved eligibility/common cohorts, observed targets, explicit missing-data/scaling/split rules, synthetic tests, new output paths, approved validation-only pilot and cost review. Training and test inference need explicit authorization; session approvals persist for the same scope. Tests never grant authorization.

Retain every declared seed and adverse result. Means average scores, not predictions. Seeds share one dataset; their SD is not an independent-dataset confidence interval. January–February test data was already observed; March requires an exposure audit before stronger claims.

## Risks and open questions

Historical global inventory failure remains unresolved; semantic reproduction is [UNKNOWN]. Temperature calibration/placement, March coverage, fitted multi-output scalers and runtime remain unverified. Original cohort admission used holdout availability. Histories share observations and node/facility conditions. Centralized SGD is a schedule-replay control, not every centralized method. See [Plan](Plan.md) for blockers and [Testing](Testing.md) for gates.
