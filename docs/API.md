# CLI and Python interface reference

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

[NOT APPLICABLE] No REST endpoints, application HTTP server or frontend interface was found. Execution is through explicit CLIs and Python functions. Commands below are source-inspected interfaces, not permission to fit, regenerate, evaluate or download. No experiment CLI was executed for this documentation review.

## Read-only orientation

```bash
cd /media/csedept/cse2014/Project/m100_training_template
.venv/bin/python -B --version
.venv/bin/python -B -m json.tool configs/experiments/phase2_multioutput/v1/protocol.json
.venv/bin/python -B -m src.phase2_multioutput.readiness --stage protocol
```

The last command returns `protocol_verified_data_not_frozen`, not March readiness. `--stage data` deliberately exits 2 with `blocked_before_real_data_stage`. Optional `--config-dir` selects the design bundle; `--evidence` accepts bounded metadata JSON. No real-data trainer is hidden behind this gate.

## Entrypoints and side effects

| Entrypoint | Source-defined interface | Side effects / expected artifacts |
| --- | --- | --- |
| `main.py` | `--config`, `--data-path`, `--output-dir`, `--model lstm|gru`, `--device auto|cpu|cuda`, `--fresh`, `--stop-after-step`, hyperparameters | Original synthetic/prepared trainer; checkpoint/log writes, not current v5 |
| `pipeline.py` | `--config`, `--data-dir`, `--output-dir`, `--stage all|download|prepare|train`, `--ca-bundle`, `--epochs`, `--device` | Default all can download/extract/prepare/train and evaluate; not inspection |
| `experiments.py` | Subcommands freeze, fit, report, matrix; matrix `--stage fit|report` | v1. Report executes test prediction/evaluation; not just document rendering |
| `federated.py` | `--config`, `--data-dir`, `--output-dir`, `--resume`, `--stop-after-round` | Original one-hour FedAvg, not v5 |
| `python -m src.centralized_sgd` | Config/data/output, `--resume` or `--verify-only`, `--stop-after-block` | Historical one-hour control; verify reloads CPU tensors without inference |
| `python -m src.v5_missing_data.run_policy_experiments` | Plan/data/output; `--stage plan|smoke|fit|verify`; method/policy/seed or `--matrix primary|sensitivity|all`; resume/stop-after-round | Plan is read-only; smoke cap 3, main cap 200. Verify reloads/initializes CPU state but never scores |
| `python -m src.v5_missing_data.evaluate_policy_models_v2` | `--stage preflight|lock|verify|evaluate`; plan/data; run roots or lock-dir; output-dir only for evaluate | Preflight discovers 24 identities; lock writes metadata; verify checks lock; evaluate runs test inference |
| `python -m src.v5_missing_data.evaluate_policy_models` | `--stage lock|evaluate|verify`; required output-dir, optional lock/run roots | Restored original; completed-output verify checks exact output inventory/content |
| `bash run_remaining_experiments.sh` | `--dry-run`, `--resume` | Sequential fits/verification and UUID visibility mapping, no automatic test |
| `python -m src.phase2_multioutput.readiness` | `--stage protocol|data`, config-dir/evidence | Design/metadata only; no model/data adapter |
| `tools/march_isolated_v3.py` | Delegated preflight/status/start/resume/stop/verify and authorization/quota options | Isolated transfer lifecycle; actual transfer requires separate explicit authorization |

## Current v5 contract

Plan: [gru_gap_policy_v1.json](../configs/experiments/v5_missing_data/gru_gap_policy_v1.json). Data: [cohorts](../output/experiments/v5_missing_data/model_comparison/v1/cohorts/REPORT.md). `load_plan` enforces settings, logical `cuda:0`, algorithms fedavg/centralized_sgd, policies no_fill/ff10/ff20/ff30, seeds 42/43/44, 34 source and six evidence hashes. No `--device cuda:1` runner override exists. CUDA_VISIBLE_DEVICES can remap a physical GPU to logical 0; review provenance and numerical continuity before resume rather than editing the plan.

v2 uses manifest identity, not directory spelling. Flat `centralized_` prefixes and `none` folder aliases can represent canonical centralized_sgd/no_fill. Exactly 24 unique expected complete main identities are required; missing/duplicate/invalid/symlinked runs stop discovery. Existing full_runs is completed, not a pending matrix authorization.

## Existing-artifact verification examples

These are manual non-training/non-inference commands, not executed here. They may fail on source/software/provenance differences; never alter expected values to force success.

Lock/model binding verification, **not test-output inventory verification**:

```bash
.venv/bin/python -B -m src.v5_missing_data.evaluate_policy_models_v2   --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json   --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts   --stage verify   --lock-dir output/experiments/v5_missing_data/model_comparison/v1/selection_locks/v2
```

Completed test-output inventory/content verification through the restored evaluator:

```bash
.venv/bin/python -B -m src.v5_missing_data.evaluate_policy_models   --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json   --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts   --stage verify   --output-dir output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2
```

A future evaluate would need that immutable lock, an unused output directory and explicit authorization. This guide intentionally supplies no runnable duplicate test or fit matrix. Completed outputs already contain 24 panels.

## Python contracts and errors

`SplitData.batch(positions)` returns inputs/targets/structured IDs for one declared split. `TrainingRegistry.local(node_index,rows)` produces immutable LocalData; `round_plan` binds rows, RNG seeds, batch sizes and steps. `GlobalModelState.initialize_client/capture_update` plus `aggregate_fedavg(...,expected_counts=...)` govern update provenance. `predict_and_score` accepts validation or authorized test, never training. `load_commits` checks actual state/trace hashes and previous links before deserialization.

Argparse validates stages/flags. v5 error prefixes and nonzero returns report failures; successful JSON status is stage-specific, not an HTTP code. Never equate exit 0, a directory or a completion marker with verified success. [Storage](Database.md) lists outputs; [Debugging](Debugging.md) lists recovery boundaries.
