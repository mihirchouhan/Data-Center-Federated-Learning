# Locked v3 centralized-SGD final test evaluation

Implementation stage only: no real test inference has been executed by the agent.
The validation report is a dated snapshot of saved scores, not a new evaluation.
All real inference commands below must be executed manually by the user.

## Fixed protocol and selections

`test_evaluation_v1.json` fixes seeds 42/43/44 and selected blocks **190/190/180**,
raw model hashes, training manifests/configuration snapshots, completed receipts,
the original frozen cohort and exact evaluation-source hashes. Selections were
made by the earliest minimum of the original validation RMSE candidates, not by
test scores. The control GRU and standardized outdoor-weather mask remain fixed.

The cohort is the existing `output/experiments/v1/frozen`: **75,566 test windows,
979 nodes**, original UTC identity/order, fourteen features, six ten-minute input
bins and next-hour mean-power target. Scalers, IDs, boundaries, targets and
persistence are unchanged. No freeze, training, tuning or model selection occurs.

The common evaluator is the same `src/experiment_eval.py` used previously. It
reports W, pooled MAE/RMSE, persistence errors, per-node metrics, coverage and
forecast plots on real UTC timestamps with breaks across temporal gaps. Outdoor
channels 7/8 alone are masked. The separate node embedding remains unchanged.
The three-seed summary uses mean individual scores and sample SD (ddof=1), not
ensemble predictions or a confidence interval from independent datasets.

The prior FedAvg test evaluation at
`output/experiments/v2_fedavg/test_evaluation/v1_retry1` is hashed and read only.
Its 44 artifacts, source snapshots, original three selected exports and cohort
identity are verified. It is NOT rerun. The earlier partial `v1` directory is
preserved. New results are compared with the stored FedAvg scores only after all
central predictions exist. Prior test exposure means this is a development
benchmark, not an untouched confirmatory test period.

## Verification modes and immutable manifest

`--verify-only` without an output directory checks all source/configuration,
model/receipt, schedule-proof, frozen-file and prior-result hashes; earliest
validation selection; finite selected states and fixed coefficients. It does not
map example arrays, create output, initialize CUDA or execute a model. Frozen
files are read as bytes for hashing. Checkpoint tensors are loaded on CPU.

Actual inference exclusively creates a new directory and writes read-only
`selection_lock.json`, `plan.json` and `manifest.json` before opening any test
arrays. ALL three exports are bound first. Execution-source snapshots are saved
and fsynced. The test-only loader maps test arrays and shared metadata only;
training/validation arrays cannot be requested. Each prediction's identity,
finiteness and unchanged model state is checked. Preflight runs again before
completion. No seed, split, checkpoint or hyperparameter override exists.

`--verify-only` WITH an existing output directory is postflight: it requires a
completed evaluation and checks all 48 report/source hashes, the selection lock,
saved score bindings, summary arithmetic and prior-result comparison. It does
not map examples, rerun predictions or produce metrics. The output itself has
49 required files including `completed.json`; an optional `.matplotlib` cache is
excluded from report hashing. No automatic resume or overwrite mode is provided.

## Manual commands (foreground, project root)

Use the existing `.venv`; no dependency installation is needed. Run each step
only after the preceding one succeeds. Stop and share any integrity error.

First run the delivery/preservation check. It includes every pre-existing
non-venv project file, including all completed v3 training runs and earlier test
artifacts, plus every new evaluator/report file. Original expected hashes are
never edited.

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B \
  output/experiments/v3_centralized_sgd/evaluator_implementation/v1/verify_preservation.py
```

Expected: no mismatches. Next perform the model/configuration preflight:

```bash
.venv/bin/python -B -m src.centralized_evaluate \
  --plan configs/experiments/v3_centralized_sgd/test_evaluation_v1.json \
  --data-dir output/experiments/v1/frozen \
  --verify-only
```

Expected `status: verified_only`, seeds 42/43/44, blocks 190/190/180, and
`test_evaluation_run: false`, `model_inference_executed: false`.

Only after preflight passes, run the actual test inference yourself:

```bash
.venv/bin/python -B -m src.centralized_evaluate \
  --plan configs/experiments/v3_centralized_sgd/test_evaluation_v1.json \
  --data-dir output/experiments/v1/frozen \
  --output-dir output/experiments/v3_centralized_sgd/test_evaluation/v1 \
  --device cuda:0
```

This exact output path must be unused. Do not pre-create it. CUDA cannot silently
fall back to CPU. Expected JSON: split `test`, units `W`, three fits, 75,566
samples per seed, 979 nodes, finite metrics and the locked selections. No accuracy
threshold is promised. `completed.json` must exist; test inference has not run
until YOU execute this command.

After successful completion:

```bash
.venv/bin/python -B -m src.centralized_evaluate \
  --plan configs/experiments/v3_centralized_sgd/test_evaluation_v1.json \
  --data-dir output/experiments/v1/frozen \
  --output-dir output/experiments/v3_centralized_sgd/test_evaluation/v1 \
  --verify-only

.venv/bin/python -B \
  output/experiments/v3_centralized_sgd/evaluator_implementation/v1/verify_preservation.py
```

Expected `status: verified_complete`, 48 artifact hashes verified, no inference,
and no preservation mismatches. Share results and verification output before
any later tuning, experiment, interpretation or additional test evaluation.

## Expected output files

The new directory contains the three immutable manifests, `source/` snapshots,
`test_window_identity.json`, `results.json`, `comparison_with_fedavg.json`,
`resources.json`, and `completed.json` written last.

Each `seed42/`, `seed43/` and `seed44/` contains `metrics.json`, `per_node.json`,
`per_node_comparison.json`, `comparison.json`, `coverage.json`,
`predictions.npz`, `forecast.png` and the common `evaluation.json` hash receipt.
`results.json` reports individual and mean/sample-SD RMSE/MAE plus persistence
comparisons. Cross-algorithm differences are explicitly signed as centralized
minus stored FedAvg (negative means smaller centralized error). No test score
influences the chosen exports. No new prediction ensemble is formed.

## Interruption/recovery

An existing output, even empty or partial, is never reused. There is NO
`--resume` flag for evaluation. A failed invocation retains its lock, partial
artifacts and `failed.json` when the exception can be recorded. A hard kill may
leave no failure marker. Absence of `completed.json` means incomplete; postflight
must not label it complete. Preserve all files and inspect the exact error.

If the error is an integrity, split or checkpoint discrepancy, stop and share it.
Do not change hashes, manifests, model files or settings. If it was solely an
interruption and preflight still passes, a fresh output suffix can rerun the
same fixed plan. This is a new full inference attempt, not continuation. After
confirming `v1_retry1` is unused, the manual retry command is:

```bash
.venv/bin/python -B -m src.centralized_evaluate \
  --plan configs/experiments/v3_centralized_sgd/test_evaluation_v1.json \
  --data-dir output/experiments/v1/frozen \
  --output-dir output/experiments/v3_centralized_sgd/test_evaluation/v1_retry1 \
  --device cuda:0
```

Use that SAME retry output for its postflight. Never delete an earlier attempt,
copy seed outputs between attempts, manufacture completion receipts, choose a
checkpoint from observed test scores, edit the original FedAvg plan/lock, or
rerun a completed evaluation unnecessarily. Keep exact code/dependencies/data;
cross-hardware bit identity is not promised. These are consistency checks, not
adversarial security or proof that humans never observed this benchmark.

## Tests

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_centralized_evaluate.py' -v
```

These tests use temporary synthetic CPU fixtures and fabricated histories, with
no optimization or real data inference. Separate implementation evidence records
real metadata-only preflight and before/after preservation. The validation
report remains immutable after future inference; create a new report version
for future test interpretation instead of editing the existing snapshot.
