# Documentation evidence and verification record

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Scope and evidence

Documentation-only review dated 11 November 2026. The local Vibe_Coding_15_Prompts.pdf supplied a documentation outline, not authority to build generic web/database/auth services. Actual user instructions and inspected research evidence govern this release.

All 22 requested files plus this verification record were unused before publication. Root README, old thesis, sealed reports and existing code/configuration remain unchanged. No earlier root/ancestor/nested-documentation AGENTS.md was found. A structure/interface audit read 138 root-guide/source/config/test/protocol files, supplemented by direct reads of relevant implementations, manifests and saved results. This is not a claim to inspect every duplicated source snapshot as new code.

| Subject | Exact evidence entry points |
| --- | --- |
| Setup/history | [README](../README.md), [EXPERIMENTS](../EXPERIMENTS.md), [launcher](../README_RUN_REMAINING_EXPERIMENTS.md) |
| Dependencies | [requirements](../requirements.txt), [optional requirements](../requirements-experiments.txt), .venv/pyvenv.cfg and installed distribution metadata |
| Model/preparation/metrics | [model](../src/model.py), [multinode](../src/multinode.py), [preprocess](../src/preprocess.py), [frozen data](../src/experiment_data.py), [metrics](../src/experiment_eval.py) |
| Original frozen evidence | [contract](../output/experiments/v1/frozen/contract.json), [preprocessing](../output/experiments/v1/frozen/preprocessing.json) |
| v1 comparison | [comparison.json](../output/experiments/v1/comparison/comparison.json) |
| v2/v3 | [final analysis](../output/experiments/v3_centralized_sgd/reports/final_test_comparison_v1/analysis.json), [historical thesis](thesis/v1/THESIS_METHODS_RESULTS.md) |
| v5 design | [protocol](experiments/v5_missing_data/NO_FILL_FORWARD_FILL_PROTOCOL_v1.md), [implementation](experiments/v5_missing_data/TRAINING_IMPLEMENTATION_v1.md), [plan](../configs/experiments/v5_missing_data/gru_gap_policy_v1.json) |
| v5 cohorts | [REPORT](../output/experiments/v5_missing_data/model_comparison/v1/cohorts/REPORT.md), [contract](../output/experiments/v5_missing_data/model_comparison/v1/cohorts/contract.json) |
| v5 execution | [runner](../src/v5_missing_data/run_policy_experiments.py), [data](../src/v5_missing_data/training_data.py), [sampling](../src/v5_missing_data/sampling.py), [I/O](../src/v5_missing_data/training_io.py), [aggregation](../src/federated_train.py) |
| v5 evaluation | [v2](../src/v5_missing_data/evaluate_policy_models_v2.py), [original](../src/v5_missing_data/evaluate_policy_models.py), [lock](../output/experiments/v5_missing_data/model_comparison/v1/selection_locks/v2/selection_lock.json), [results](../output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2/results.json) |
| v5 analysis | [report](../output/experiments/v5_missing_data/model_comparison/v1/results_analysis/COMPLETE_V5_RESULTS_REPORT.md), [verification](../output/experiments/v5_missing_data/model_comparison/v1/results_analysis/verification.json) |
| History caution | [comparison](../output/experiments/v5_missing_data/model_comparison/v1/history_comparison_v1/comparison_report.md) |
| Integrity | [investigation](../output/phase1_integrity_review/integrity_review.md), [summary](../output/phase1_integrity_review/integrity_summary.json), [reconciliation](../output/phase1_provenance_reconciliation/reconciliation_gates.md) |
| Temperature | [semantics](../output/phase1_4/temperature_semantics.md) |
| Future design | [protocol](experiments/phase2_multioutput/v1/PROTOCOL.md), [evaluation](experiments/phase2_multioutput/v1/EVALUATION_SPEC.md), [FL](experiments/phase2_multioutput/v1/FEDERATED_PROTOCOL.md), [completion](../output/phase2_0/completion.json) |
| Acquisition | [helper v2](../tools/acquire_march_2022_v2.py), [background v3](../tools/march_background_v3.py), [isolated v3](../tools/march_isolated_v3.py) |

## Saved numerical evidence, not new scoring

This table copies the eight groups in saved v5 test results.json. Means/sample SD were reconciled in memory against the 24 saved per-seed metric records, not rescored predictions. Every saved record has 6,590 windows/570 nodes, ordered identity SHA256 `7d44fa703b877579791162594922088bea42674c215de8503e65768ca8a2648a`, matching targets/persistence hashes. Per-node/macro fields exist. Persistence MAE 27.147075 W / RMSE 58.269808 W belongs to this cohort, not the older 75,566-window test.

| Method | Policy | Test RMSE W, mean ± sample SD | Test MAE W, mean ± sample SD |
| --- | --- | --- | --- |
| fedavg | no_fill | 56.851084 ± 0.111688 | 29.916955 ± 0.119912 |
| fedavg | ff10 | 56.834380 ± 0.120754 | 29.825478 ± 0.139497 |
| fedavg | ff20 | 56.830222 ± 0.114482 | 29.855553 ± 0.117770 |
| fedavg | ff30 | 56.831318 ± 0.115181 | 29.841830 ± 0.123997 |
| centralized_sgd | no_fill | 51.505102 ± 0.711372 | 28.158903 ± 0.641842 |
| centralized_sgd | ff10 | 50.976723 ± 0.284497 | 27.346591 ± 0.608253 |
| centralized_sgd | ff20 | 51.068241 ± 0.394274 | 27.563574 ± 0.620546 |
| centralized_sgd | ff30 | 51.193037 ± 0.714977 | 27.554376 ± 1.274690 |

Centralized SGD has lower measured mean v5 pooled errors than FedAvg for each policy. FF20 is predeclared primary; a descriptive policy ranking depends on method/metric and is not test-based retuning. Seeds share one observed development/test dataset. This is not a March/two-output outcome; use the existing report for per-seed/node detail rather than mixing studies.

## Documentation validation

- All 22 requested destinations plus this new audit record exist in staging; 18 Markdown files and five standalone Mermaid files. No existing files require replacement.
- UTF-8/newline/fence checks and **163 local documentation links** passed.
- **48 source interface/CLI-flag assertions** passed through static inspection; experiment CLI commands were not executed.
- **Nine Mermaid blocks** (five standalone plus four embedded) passed actual Mermaid parsing and SVG rendering using the installed VS Code Mermaid bundle and offline headless Chrome with GPU disabled. No dependency was installed. The first sandboxed renderer launch failed on a restricted socket operation; the permitted offline run passed. Rendering used local DevTools loopback only, not an external service.
- Eight group summaries were reconciled with all 24 saved per-seed metric records. The 75 evaluation artifacts, 40 plan-bound source/evidence files, 24 model exports, 24 configuration files, 14 original frozen arrays and prior Phase 2 completion binding passed **178 actual-byte hash checks**. Checkpoints/prediction vectors were hashed as bytes, never deserialized or scored.
- The previous Phase 2 baseline matched 20,089 older files and 25 bound additions. Fresh pre-change preservation covers **20,115 files / 36,118,143,599 bytes**, with zero mismatches. The previous completion itself also matches its separately recorded SHA-256.
- Cross-document review separates historical one-hour, current six-hour scalar-power and future two-output protocols; covers exact features, split/scaler differences, simulated clients and observed test exposure. Only installed/source-defined commands are listed. No database/auth/REST/MCP implementation is invented.
- After publishing the 22 requested documents, a full actual-byte/metadata comparison passed for **all 20,115 pre-existing files / 36,118,143,599 bytes**, zero mismatches (41.34 seconds). The only additions were the 22 requested documentation files. This audit record is then created exclusively, with another existing-file metadata check. Historical strict preflight remains FAILED; no semantic replay occurred.
- Installed documents exactly match staged bytes. The nine rendered Mermaid definitions match the installed definitions by identity. The root AGENTS.md is under 100 lines. Explicit checks confirm separate historical/prior-split versus future/whole-history rules and blocked multi-output status.


## Preservation scope

Before writing, compare the previous Phase 2 inventory and bound additions against actual bytes/metadata, then capture a fresh current inventory. After publication, compare every pre-existing regular file for SHA256, bytes, mtime_ns, mode, inode and device. New root/docs entries necessarily change parent directory mtimes; that is not a change to an existing file. Access times are excluded.

Exclude .git, .venv, __pycache__ and externally managed acquisition trees data/march_2022_manual, data/march_2022_acquisition/v3, output/march_2022_acquisition/v3, output/march_2022_background/v3. Do not read/hash a potentially growing March payload. Old V1/V2 acquisition evidence remains included; the current environment was inspected via metadata only.

No existing code/config/data/checkpoint/prediction/result/hash/receipt/baseline was edited. No model initialization, training, inference, scoring, preparation/extraction, archive transfer, external network request, GPU query or dependency install occurred. Offline rendering used local DevTools loopback only. No historical experiment tests or official semantic reproduction ran. Diagram rendering is a documentation operation, not a model evaluation.

[UNKNOWN] Current March archive success/coverage, precise sensor calibration, cross-hardware bit identity and end-to-end historical reproduction. Privacy deployment and energy decisions are not implemented. New docs cannot repair FAILED historical preflight. Externally excluded active-transfer files are outside the preservation claim.

## Exact content anchors

These are observed SHA-256 values for citation, not updated expected hashes or a new experiment baseline.

| Source | SHA-256 |
| --- | --- |
| `output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2/results.json` | `ea369a30c4f32efe040f13b0f36244b0014b2a34bd218528fc484036250b1fe7` |
| `output/experiments/v5_missing_data/model_comparison/v1/selection_locks/v2/selection_lock.json` | `e79d43cb47bf95c805474894ad21680f91d66b57e812e80f2a17bcbfa32c74ef` |
| `configs/experiments/v5_missing_data/gru_gap_policy_v1.json` | `ff63776db8b4be6ace0dab76572f61d426c32f3635fbb25a2280910b56120301` |
| `output/experiments/v1/frozen/contract.json` | `24b2224ea9a7e1f255faf6357ffe7a3536fcbcfd90ab8b73179798349736b722` |
| `output/phase2_0/completion.json` | `c2ef98ab4104bee67e5a9beefbea4288d952f0f200e8d4f6359aa7493779f30f` |

## Files created

No pre-existing file was updated. The additions are:

- `AGENTS.md`
- `docs/API.md`
- `docs/Agents.md`
- `docs/Architecture.md`
- `docs/Authentication.md`
- `docs/DOCUMENTATION_AUDIT.md`
- `docs/Database.md`
- `docs/Debugging.md`
- `docs/Design.md`
- `docs/Features.md`
- `docs/INDEX.md`
- `docs/Integration.md`
- `docs/MCP.md`
- `docs/PRD.md`
- `docs/Plan.md`
- `docs/README.md`
- `docs/Setup.md`
- `docs/Testing.md`
- `docs/diagrams/data_pipeline.mmd`
- `docs/diagrams/experiment_workflow.mmd`
- `docs/diagrams/federated_learning.mmd`
- `docs/diagrams/research_workflow.mmd`
- `docs/diagrams/system_architecture.mmd`
