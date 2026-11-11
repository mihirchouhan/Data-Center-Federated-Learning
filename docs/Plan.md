# Research roadmap and completion gates

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

Checked items describe existing evidence, not fresh execution here. Numbering never authorizes an automatic next stage.

## Completed milestones

- [x] [IMPLEMENTED] Original causal preprocessing and one-hour frozen benchmark.
- [x] [IMPLEMENTED] v1 LSTM/GRU/XGBoost × weather × seeds comparison.
- [x] [IMPLEMENTED] v2 FedAvg / v3 matched centralized-SGD loops and final saved-prediction/node reporting.
- [x] [IMPLEMENTED] History/fill audits; v5 common cohorts and reference-size sampling budgets.
- [x] [IMPLEMENTED] v5 24 fits, evaluator repair, immutable lock v2, completed test and thesis/report packages.
- [x] [IMPLEMENTED] Read-only integrity, target and temperature semantics investigations; unknowns preserved.
- [x] [IMPLEMENTED] Isolated acquisition helpers/background workflow and synthetic evidence. Current external March success [UNKNOWN] here.
- [x] [PARTIALLY IMPLEMENTED] Phase 1.3 synthetic two-output prototype and Phase 2 frozen design/NumPy contracts.
- [x] [IMPLEMENTED] This documentation navigation system; experiment code/artifacts unchanged.

Earlier guides retain their original then-future wording; later completed artifacts establish subsequent status. Do not edit history to conceal chronology.

## Blockers and definitions of done

| Gate | State | Completion criterion |
| --- | --- | --- |
| Historical reproduction | [BLOCKED] old global inventory; numerical replay [UNKNOWN] | Approved provenance resolution/new-scope acceptance, no forged metadata |
| March acquisition | [UNKNOWN] current external success/checksums | Authorized immutable exact-size/MD5/SHA256 receipt; don't inspect a growing payload |
| March schema/coverage | [BLOCKED] not established | Approved read-only units/joint targets/full-input histories/eligibility audit |
| Real two-output bridge | [PLANNED] absent | Tested UTC/grid/raw contributor/full-history/target containment and common IDs |
| Actual cohort/scalers | [PLANNED] not frozen | Training-only eligible population and scaler provenance; no holdout-based admission |
| Train/test adapters | [PLANNED] pure definitions only | Tested initialization, optimizer, aggregation, selection, resume and actual checkpoint-byte locking |
| Pilot/fits/test | [BLOCKED] prerequisites/approval unmet | Separate explicit approvals following correctness and measured cost review |

## Additive work in dependency order

1. Verify acquisition identity and approved scope. Read-only coverage is a separate authorized operation. Phase 2 readiness has no extraction command.
2. Implement the real bridge in new approved paths, reusing verified preparation only where semantics match. Prove hourly origins, observed joint targets and full-history/raw-contributor split containment. Stop if declared cohorts or >=98 training clients cannot be supported.
3. Freeze accepted suite IDs/mapping/scalers after review. Full fourteen slots; weather enabled/masked on identical windows. Original train/validation cuts; old February test excluded; March retrospective test subject to coverage/exposure review.
4. Implement trainable two-output adapters against [Phase 2](experiments/phase2_multioutput/v1/PROTOCOL.md), with no silent power-only residual reuse. Reuse generic state validation only after proving compatibility.
5. Run approved synthetic checks and a separately authorized validation-only pilot. Measure runtime/memory, finite outputs, export/reload and interruption/accounting. Pilot success means engineering correctness, not improved model quality.
6. After cost review, authorize the matrix. Design declares 36 fit identities and four seedless persistence references, not completed results. Primary and history suites may use different cohorts; don't collapse them.
7. Lock all declared validation selections, then separately authorize test evaluation into unused output. Audit prior March exposure; retrospective data is not automatically untouched.
8. Publish saved-artifact analysis and thesis with per-seed/node errors, adverse results, macro/pooled definitions and limitations. Never choose a test-winning seed or policy retrospectively.

## Later work

[PLANNED] Matched 1/6/24-hour histories with per-split intersections and native-coverage sensitivities. [PLANNED] Environmental ablation separating indoor 6 from outdoor 7/8. [PLANNED] Independent energy-management policy/what-if study with actions, workload constraints, uncertainty, energy measurement boundaries and controlled outcomes. Real privacy/security deployment needs a separate threat model and tested mechanisms.

Every stage ends with scoped tests, actual evidence, preservation, known limitations and review before the next gated action. Software completion never implies accuracy gains or energy savings.
