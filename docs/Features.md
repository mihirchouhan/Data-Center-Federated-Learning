# Feature catalog and verification boundaries

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Existing capabilities

Each row identifies purpose, source/interface, inputs/outputs and verification limits. Abbreviated original modules are under src; v5 modules under src/v5_missing_data.

| Capability / status | Files and interface | Inputs → outputs; verification and limitations |
| --- | --- | --- |
| Acquisition [IMPLEMENTED] | src/download.py; isolated helper v2/wrapper v3 | Official metadata/owned session → partial, checkpoint, verified receipt. Synthetic safety evidence is not present March success |
| Causal preprocessing [IMPLEMENTED] | preprocess._range_mask/_weather_asof; multinode.weighted_bins | Raw records → coverage-qualified bins. Duplicate/range/age rules; no guarantee of continuous observation |
| Features/windows [IMPLEMENTED] | multinode.node_windows/prepare_multinode | Node bins → chronological windows/scalers/audit. No gap/node joining; original cohort admission used later coverage |
| Frozen one-hour data [IMPLEMENTED] | experiment_data.FrozenData/weather_view | Prepared source → hashed arrays/copied mask. Explicit six-step assumptions |
| GRU/LSTM/XGBoost [IMPLEMENTED] | model.SequenceRegressor; experiment_models.py | Frozen batches/lags → validation-selected models. Completed v1 comparison, not March results |
| Client data [IMPLEMENTED] | federated_data.py; v5 TrainingRegistry.local | Training partition/planned rows → copied local tensors. API separation is not OS isolation |
| FedAvg state [IMPLEMENTED] | federated_train.GlobalModelState/aggregate_fedavg | Snapshot, counts, checked updates → new CPU state. Finite/name/shape/dtype/fixed-buffer checks; full embedding table |
| Round loop [IMPLEMENTED] | federated_rounds.py; v5 run_one | Config/local plans → atomic ledgers/selected exports. No silent replacement; cross-hardware numerical identity unproven |
| Central control [IMPLEMENTED] | centralized_sgd.py; v5 central branch | Paired plans → continuing model. Matched settings/exposures, not same optimization |
| History audit [IMPLEMENTED] | history_data_v4.py/history_coverage_v4.py | Validity/reference → coverage. Current historical inventory may block a rerun |
| Missing-data cohorts [IMPLEMENTED] | v5 forward_fill_audit.py/build_training_cohorts.py | Original bins/ages → native pools/common IDs. Never fill labels; causal per-split reset |
| Fixed sampling [IMPLEMENTED] | sampling.round_plan/verify_budget_totals | Seed, native pool, m_k → row plan/accounting. Pools intentionally differ |
| Checkpoints/resume [IMPLEMENTED] | training_io.commit/load_commits/export | State/rows/previous receipt → numbered commits. Preserve pending evidence; exact bindings required |
| Locked evaluation [IMPLEMENTED] | evaluate_policy_models_v2 + original | 24 selected models/lock → common test panels. Lock verification differs from output verification |
| Metrics [IMPLEMENTED] | experiment_eval.metric_pair; predict_and_score | Aligned W predictions/targets/persistence → pooled/per-node/macro errors. Not energy outcomes |
| Reporting [IMPLEMENTED] | results_analysis/scripts; thesis packages | Saved artifacts → tables/charts/docs/provenance. No inference needed; sealed packages cannot accept arbitrary additions |

Workflow/dependencies are in [Architecture](Architecture.md), artifacts in [Database](Database.md), exact CLI stages in [API](API.md). Verify changes with the focused test groups in [Testing](Testing.md), actual bound inputs and no-clobber checks; never infer successful real execution from implemented interfaces alone.

## Partial and future capabilities

| Capability | Status / completion gate |
| --- | --- |
| Synthetic two-output GRU | [PARTIALLY IMPLEMENTED] phase1_3.model.MultiOutputGRU shape demo; no real trained model |
| March two-output protocol | [PARTIALLY IMPLEMENTED] phase2_multioutput contracts/design; actual cohort/scalers/train/test adapters [BLOCKED] |
| Controlled 1/6/24-hour comparison | [PLANNED] per-split intersections, invariant targets/budgets; current historical full-input 24h test coverage inadequate |
| Phase 2 weather comparison | [PLANNED] enabled/masked on identical full-input windows/scalers |
| Energy-aware decisions / what-if | [PLANNED] executable actions, constraints and measured policy comparison required |
| Networked/private deployment | [PLANNED] threat model, authentication/transport/isolation and tested privacy mechanisms |
| Web/auth/database/MCP layer | [NOT APPLICABLE] absent from current research CLI |

A past synthetic pass does not demonstrate numerical GPU resume, sensor calibration, future coverage or historical reproduction. Protocol changes and incomplete functionality must remain explicit, versioned and separately approved.
