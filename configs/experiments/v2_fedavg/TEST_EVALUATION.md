# Frozen FedAvg GRU test evaluation

This additive implementation does not change `federated.py`, model/training
sources, existing configurations, datasets or prior outputs. Implementation tests
use temporary synthetic data and fabricated checkpoint records. The real test
evaluation must be run separately after reviewing the implementation and tests.

The new `test_evaluation_v1.json` plan explicitly pins all three selected exports,
their selection receipts, training manifests, original configuration files and
completion events by SHA-256. It pins the evaluation execution sources as well.
No checkpoint is chosen using test scores, and no directory is selected by a
"latest" or "best test" heuristic.

## Seed 42 path resolution

The intended `output/experiments/v2_fedavg/main/gru_masked_seed42` directory was
absent during inspection. The existing `gru_masked_seed42b` directory contains a
seed-42 configuration, a complete round-0 through round-200 checkpoint chain,
the round-200 selection receipt and the corroborating complete event
`events/64ae616e051e467dbf4a6990b410f479.json`. Its current source hashes and source
snapshots match the recorded manifest. The plan explicitly binds this run;
there is no renaming, copying of model weights or silent path fallback.

| Seed | Directory under `output/experiments/v2_fedavg/main/` | Selected model SHA-256 |
|---|---|---|
| 42 | `gru_masked_seed42b` | `dabe9ae16022acaaae56ef54e082e52b3f516628349a04c74acc5053ed7e3aa3` |
| 43 | `gru_masked_seed43` | `50702e26a18edcbdd6eb7a69b3a908ad18c014d0bea730db5f0ba65f7d7f47cb` |
| 44 | `gru_masked_seed44` | `aab0f61ba0315476947119d1e46ee0ecb1b96bc0bd31e97119d1f03b8d0148f0` |

Each model file is `selected/model.pt`; each selected round is 200, established
from the earliest minimum recorded validation RMSE. Existing selection receipts
are not modified or re-signed.

## Reused code and safeguards

`FrozenData` loads the original arrays/scalers and verifies all frozen split
hashes, alignment, categorical node identity and chronological boundaries
read-only. A test-only facade exposes just `split('test')` to inference and the
common evaluator. Training and validation records are never used as test
predictions. No scaler is fitted. The CLI requires the original contract
`24b2224ea9a7e1f255faf6357ffe7a3536fcbcfd90ab8b73179798349736b722`,
75,566 test windows and all 979 nodes.

The model uses `federated_rounds.initialize_model`, strict state validation via
`GlobalModelState`, and the original `load_commits` / `load_selected` readers.
Construction is followed by loading the exact selected weights; there are no
optimizer steps. `experiment_models.neural_predict` applies the unchanged
weather mask and preserves batch order. Model-state fingerprints are compared
before and after inference. The feature and node-index order remain frozen.

`experiment_eval.evaluate` performs the existing inverse transformation:
`(scaled_target * y_std[0] + y_mean[0]) * 1000` watts. Its persistence definition
is the last completed ten-minute binned power value, the final history step in
input channel 0, held constant for the next-hour target:
`(X_test[:, -1, 0] * x_std[0] + x_mean[0]) * 1000` watts. It is not the average of
the whole input hour. The model forecasts the existing next-hour average target.

The new output directory is created exclusively. An immutable selection lock
binds all three models before any test examples are loaded for inference.
Canonical `(node, UTC forecast origin)` identities must be unique, complete and
identical in frozen order for every model and persistence. Every node is retained.
The evaluator rejects nonfinite predictions or changed model parameters.

The original common outputs include overall and per-node MAE/RMSE in watts,
per-node sample counts, identical-window persistence metrics, coverage,
timestamp-aware forecast plots and predictions with node/origin identities.
New comparison files add signed percentage improvements, counts of nodes beating
persistence, and across-seed mean and sample standard deviation (`ddof=1`).
Positive percentages mean improvement, negative mean deterioration. A zero
baseline with positive model error has undefined percentage, recorded as `null`;
both errors zero yields 0 percent. Seeds remain in declared order; no seed is
chosen using test scores and no ensemble is fitted. Standard deviation across
three seed metrics is not a confidence interval.

The root output includes `selection_lock.json`, `plan.json`, `manifest.json`,
execution-source snapshots, `test_window_identity.json`, `seed42/`, `seed43/`,
`seed44/`, `results.json`, resource measurements and an artifact-hash completion
receipt. A failure retains partial outputs and an error record; existing results
cannot be overwritten or automatically resumed. Do not tune from observed test
scores or manually change expected hashes to get past a failure.

This is the same previously examined development benchmark used by v1; it is
not a newly untouched time period. The evaluator adds no privacy mechanism.

## Commands

Run from `/media/csedept/cse2014/Project/m100_training_template` using the existing
venv; no dependency installation is needed.

Safe preflight (no examples loaded, predictions, scores or output writes):

```bash
.venv/bin/python -B -m src.federated_evaluate --plan configs/experiments/v2_fedavg/test_evaluation_v1.json --data-dir output/experiments/v1/frozen --verify-only
```

Synthetic tests and full historical preservation check:

```bash
M100_FEDAVG_EVALUATOR_BASELINE=output/experiments/v2_fedavg/evaluator_implementation/v1/baseline.json .venv/bin/python -B -m unittest discover -s tests -p 'test_federated_evaluate.py' -v
.venv/bin/python -B output/experiments/v2_fedavg/evaluator_implementation/v1/verify_preservation.py
```

Only after review, explicitly run the actual held-out evaluation:

```bash
.venv/bin/python -B -m src.federated_evaluate --plan configs/experiments/v2_fedavg/test_evaluation_v1.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v2_fedavg/test_evaluation/v1 --device cuda:0
```

Do not pre-create the final output directory. The implementation task does not
execute this command. The pinned plan identifies all three checkpoint paths and
hashes; no model path is inferred from a directory suffix. Explicit CPU inference
is available via `--device cpu`, recorded in the evaluation manifest; CUDA never
silently falls back. Keep code, configs and selected exports unchanged throughout
the final evaluation.
