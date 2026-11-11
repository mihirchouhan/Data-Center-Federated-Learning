# Tests and verification limits

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Existing test layers

| Layer | Existing paths | Tested intent / limits |
| --- | --- | --- |
| Original preparation/pipeline | tests/test_preprocess.py, test_multinode.py, test_pipeline.py, test_download.py, test_recovery.py | Causal bins/splits, synthetic acquisition and stage/recovery |
| v1 comparison | tests/test_experiments.py | Frozen masks, adapters and evaluator |
| FL data/state | tests/test_federated_data.py, test_fedavg.py | Node membership, training-only API, unequal weights/fixed state |
| FL rounds/integrity | tests/test_federated_rounds.py, test_federated_integrity.py, test_federated_stage2_integrity.py | Sampling, commits/resume, protected inventories |
| Central control | tests/test_centralized_sgd.py, test_centralized_integrity.py | Initialization/schedule/accounting and integrity |
| Locked evaluation | tests/test_federated_evaluate.py, test_centralized_evaluate.py | Selection lock, IDs, physical metrics and preservation |
| History/eligibility | tests/test_history_coverage_v4.py, test_v5_missing_data_eligibility.py | Gap-safe histories, observed targets, split/identity accounting |
| Current v5 | src/v5_missing_data/test_*.py | Fill/cohort/sampling/ledger/selection guards and evaluator-v2 discovery |
| Acquisition | tests/test_march_*.py | Loopback synthetic validators/resume/ownership/background/no-clobber |
| Synthetic multi-output | tests/phase1_3/test_pipeline.py | Shapes/scaling/windows; immutable output-fixture caveat |
| Phase 2 definitions | tests/phase2_multioutput/test_protocol.py | Pure NumPy temporal/client/target/scaler/metric/design contracts |

These describe inspected code and historical evidence, not freshly executed suites in this documentation task.

## Manual synthetic protocol check

The saved [Phase 2 test report](../output/phase2_0/test_report.md) records 77 tests with zero failures/errors/skips. The inspected suite uses fabricated arrays/temp fixtures, not models/GPU/archive requests or real scaler fitting.

```bash
cd /media/csedept/cse2014/Project/m100_training_template
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''   .venv/bin/python -B -m unittest discover   -s tests/phase2_multioutput -p 'test_*.py' -v
```

Do not blindly run all historical tests: some require exact environments/frozen data and Phase 1.3 uses immutable one-run fixtures in output/phase1_3. Inspect setup/teardown/destinations. Some official verify commands initialize/reload tensors, and some report stages execute inference. [API](API.md) distinguishes them.

## Required regression evidence for approved changes

| Area | Positive / negative cases |
| --- | --- |
| Windows | UTC grid, same node, no gap bridge, exact boundaries, duplicate IDs, hourly non-overlap when specified |
| Targets | Six original observed bins, no imputation, correct units/normalization |
| Policies | Original-source age/no renewal, split reset, nesting and common holdout IDs |
| Sampling | Independent streams, same-seed rows, distinct subsets, m_k/remainders/steps/exposures |
| FL | Immutable common snapshot, unequal weights, no global mutation, finite/name/shape/dtype/fixed-buffer validation and embedding behavior |
| Resume | Reject changed manifest/config/software/source/trace; preserve pending; compare uninterrupted/restarted GPU numerics before claiming exact equivalence |
| Selection/test | Earliest validation minimum, complete schedule, no test selection, real file-hash lock before test open, invariant IDs/targets/baseline |
| Metrics | Hand-computed physical MAE/RMSE, pooled vs equal-node differences, sample SD and every seed |
| Preservation | Actual SHA/size/mtime/mode/inventory before/after, no expected-value updates |

## Documentation checks and gaps

[DOCUMENTATION_AUDIT](DOCUMENTATION_AUDIT.md) records actual link/source/numerical/Mermaid/preservation checks for these new documents. No experiment test suite, training, model initialization/inference/scoring, data regeneration or official semantic replay was executed here.

No test coverage percentage was measured. Synthetic tests don't establish sensor calibration, future coverage, privacy, energy outcomes or cross-hardware reproduction. Phase 2 lock validation is metadata-only; real checkpoint-byte/test adapter remains missing. Fresh scoped byte preservation does not repair failed historical preflight. See [integrity review](../output/phase1_integrity_review/integrity_review.md).

Before completion claims, check approved scope, correct root/interpreter, unused outputs, bound protocol/source/inputs, observed targets/cohort identities, actual test counts/skips, resource scope and preserved old files. A historical passing log is not a fresh run; passing tests cannot grant next-stage authorization.
