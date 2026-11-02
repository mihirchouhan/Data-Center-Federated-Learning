# V3 centralized-SGD control

This is an additive, validation-only comparison against the three completed v2
FedAvg runs. It does not replace the historical centralized Adam benchmark.
No protected source, configuration, scaler, dataset, expected hash or old output
is changed. The v1 dynamic execution-source glob is unchanged: neither new module
matches `src/experiment_*.py`.

## Research protocol

The control uses the same frozen 4,506,133 training windows, 199,670 validation
windows, 979 nodes, fourteen channels, weather mask at standardized channels 7/8,
one-hour history, next-hour mean-power target, GRU architecture and node embedding.
It uses the original model factory and fresh seeded initialization, then requires
exact equality with the saved round-zero state. A trained checkpoint is never
loaded into the new control model. `baseline_coefficients` remain fixed.

SGD learning rate is 0.01, with zero momentum/weight decay, batch 512,
`drop_last=False`, and one pass through each selected client's windows. In each
block, the exact recorded 98-client order and local shuffle seeds are replayed.
One continuing centralized model takes every minibatch step. There is no local
model reset or FedAvg aggregation. A fresh stateless SGD object per block has
identical semantics to keeping zero-momentum/zero-decay SGD alive across blocks.
The loss is the original mean MSE in standardized target units.

Main runs use 200 blocks; validation occurs at 0,10,...,200, with the earliest
strict minimum pooled validation RMSE W selecting the model. The initial
zero-correction persistence candidate participates. There is no patience stop.
All eligible nodes remain in the frozen training cohort, with precisely the
historical repeated exposures. Budgets are:

| Seed | Authoritative reference | Examples | SGD steps |
|---|---|---:|---:|
| 42 | `output/experiments/v2_fedavg/main/gru_masked_seed42b` | 90,220,299 | 186,208 |
| 43 | `output/experiments/v2_fedavg/main/gru_masked_seed43` | 90,210,631 | 186,192 |
| 44 | `output/experiments/v2_fedavg/main/gru_masked_seed44` | 90,221,096 | 186,254 |

The smoke config is a diagnostic three-block PREFIX of seed 42's main schedule,
using the same 98 clients per block, architecture and complete validation cohort.
It adds validation at its final block 3 (main validation is at block 200). Smoke
scores are not full-budget results and must not be used for test-based tuning.
Stopping after block 1 and restarting with `--resume` exercises a process restart.
Main configurations are supplied for reproducibility, but no main runs are
authorized or launched by this implementation stage.

## Evidence and limits of schedule matching

Startup verifies the reference manifest/config, source hashes and snapshots,
Python/distribution/CUDA-build versions, frozen node mapping, row-ID hashes,
counts, complete checkpoint/receipt chain, selection export and completion event.
All recorded client orders, zero initial local optimizer states and shuffle
seeds must agree with the original deterministic procedure. Reference validation
scores verify existing provenance only; they never select a control checkpoint.

Historical per-batch row IDs were NOT stored. Batches are reconstructed with the
recorded seed, original NumPy `default_rng(seed).permutation(n)` and immutable
client-local row order. Synthetic tests compare the actual arrays/IDs observed
by the original `train_client` function with the control, including a 512+1
remainder. This demonstrates replay under the recorded implementation and
environment; it cannot create an independently recorded historical batch trace.
Each new block logs row-order hashes and batch sizes to improve future evidence.

The same nominal SGD learning rate and number of minibatch steps do not imply
the same effective global update. FedAvg averages local models and dilutes
client-specific embedding-row updates; centralized SGD applies steps sequentially.
This is a node-grouped schedule control, not a conventional pooled-shuffle control
or a claim that either algorithm has been optimally tuned. Seeds share one
dataset and do not provide independent-dataset confidence intervals.

## Manual smoke pilot (foreground, project root)

Use the existing `.venv`; do not install or upgrade packages. The CLI permits
only the original M100 training/validation cohort and v3 output/config paths.
It never opens test arrays. Initializing CUDA is explicit; CPU fallback is forbidden.

Run these commands yourself, in order. A failing command means stop and inspect
its output. Fresh runs refuse any existing output path. Do not pre-create it.

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B output/experiments/v2_fedavg/evaluator_implementation/v1/verify_preservation.py
```

After the integrity check passes:

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B -m src.centralized_sgd \
  --config configs/experiments/v3_centralized_sgd/gru_masked_seed42_smoke.json \
  --data-dir output/experiments/v1/frozen \
  --output-dir output/experiments/v3_centralized_sgd/pilots/gru_masked_seed42_smoke_v1 \
  --stop-after-block 1
```

Expected: JSON `status: stopped`, `completed_block: 1`, and no `completed.json`.
The block-one counters must be **452,983 examples and 936 SGD steps**.
After this succeeds (or after an interruption with a valid manifest), resume
the same run with the exact same configuration, code, dependencies and device:

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B -m src.centralized_sgd \
  --config configs/experiments/v3_centralized_sgd/gru_masked_seed42_smoke.json \
  --data-dir output/experiments/v1/frozen \
  --output-dir output/experiments/v3_centralized_sgd/pilots/gru_masked_seed42_smoke_v1 \
  --resume
```

Expected: `status: complete`, `completed_block: 3`, finite validation metrics,
`export_reload_exact: true`, **1,354,678 examples and 2,796 SGD steps**.
The selected block may be 0 or 3; no particular accuracy is promised.
To verify the saved run without model inference:

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B -m src.centralized_sgd \
  --config configs/experiments/v3_centralized_sgd/gru_masked_seed42_smoke.json \
  --data-dir output/experiments/v1/frozen \
  --output-dir output/experiments/v3_centralized_sgd/pilots/gru_masked_seed42_smoke_v1 \
  --verify-only
```

Expected `status: verified_complete`, `completed_block: 3`,
`model_inference_executed: false`. Verification reads training/validation data
and stored weights, but performs no prediction or scoring. Re-run the integrity
checker after the pilot. Share the terminal outputs before any full experiment.

## Outputs, interruption and integrity

Each fresh run exclusively creates `manifest.json`, `configuration.json`,
`reference_lock.json` and `source/` snapshots. Immutable `blocks/0000` through
`blocks/0003` (smoke) or `0200` (main) contain `state.pt` and hash-linked
`receipt.json`. Model/best weights, selection, reference binding and counters are
checked before resuming. Failed blocks are replayed in full from the last commit;
no participants or weights are silently substituted. Hidden `.pending-*`
directories are retained and ignored. Operational `.run.lock` and new invocation
events are mutable/new bookkeeping, not model checkpoints.

`final/model.pt`, `final/receipt.json`, and `final/validation/` contain the selected
model, reload check, common-evaluator pooled/per-node metrics, coverage,
`predictions.npz`, `forecast.png`, and `evaluation.json`. `completed.json` is
written last. `events/` records stopped/completed/aborted invocations and their
resource scope. A final export interrupted before its atomic rename is replayed;
a valid existing final export is verified and reused, never overwritten.

If interrupted before the manifest/source snapshot is fully written, automatic
resume may correctly refuse the incomplete run. Preserve it and select a NEW
output suffix after review; never delete files or alter expected hashes. Resume
requires identical source/config/environment and dataset. CPU bitwise restart
equivalence is tested in a separate Python process. CUDA restart equivalence is
not asserted before the user-run GPU pilot; cross-hardware bit identity is not
promised. Three smoke blocks cannot establish full-budget model stability.

RSS is a process-lifetime high-water mark; sampled GPU memory may miss brief
peaks. Event wall time excludes initial data/reference loading. Shared GPU load
can affect runtime. No new central runtime or accuracy is measured until you run
the pilot. Existing FedAvg runtimes (roughly 33–57 min/seed) are planning context,
not a prediction or measurement for this control.

Existing test results have already been examined. Nothing here establishes an
untouched test period, independent data centers, secure aggregation, differential
privacy, energy/cooling savings, PUE improvement or carbon reductions.

## Synthetic tests

```bash
cd /media/csedept/cse2014/Project/m100_training_template
.venv/bin/python -B -m unittest discover -s tests -p 'test_centralized*.py' -v
```

All optimization in these tests is on temporary synthetic CPU fixtures. No real
training, GPU pilot, dataset freezing, v1 report generation or test evaluation is
invoked. The implementation evidence directory separately records preservation
of every pre-existing non-venv project file and read-only reference preflight.
