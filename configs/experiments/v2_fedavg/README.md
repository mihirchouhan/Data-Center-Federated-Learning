# Stage 2 FedAvg GRU

This runner simulates one compute-node client per frozen node ID in a single
process. It provides no secure aggregation, differential privacy or OS-level
client isolation. It never opens frozen test arrays or invokes v1 reporting.

`gru_masked_seed42.json`, `gru_masked_seed43.json` and `gru_masked_seed44.json`
declare the future main protocol: 200 rounds, 98 clients sampled uniformly without
replacement each round, one complete local epoch, batch size 512 with the partial
last batch retained, fresh SGD with lr 0.01, no momentum or weight decay. All 979
clients and 4,506,133 training windows remain eligible. Main runs are not launched
by installation or tests.

`gru_smoke_seed42.json` is explicitly a diagnostic pilot: three rounds with three
clients per round, validation every round. It keeps the main architecture,
preprocessing, weather mask, client eligibility and optimizer. This pilot is not
an accuracy comparison and must not be used to tune settings on test results.

The GRU has two 64-unit recurrent layers and a separate eight-dimensional shared
node embedding. The residual head starts at zero (the initial persistence
candidate); no trained checkpoint is imported. Frozen `baseline_coefficients`
remain fixed. The complete embedding table uses ordinary sample-weighted FedAvg,
including dilution of a client's own embedding-row update by its aggregation
weight. There is no row-specific reweighting.

Initialization, sampling and local shuffling have separate deterministic random
domains. Sampling is indexed by round; local permutations by round and frozen
node index. There is no mutable sampling cursor, stochastic augmentation or GRU
dropout. Optimizer state is discarded after each client. CUDA uses deterministic
algorithms, no TF32 and a fixed cuBLAS workspace setting. Exact reproducibility
requires the same software/device environment; cross-hardware bit identity is
not promised.

Each run exclusively creates a manifest, configuration snapshot and execution
source snapshots. Immutable per-round directories contain CPU model states,
selected-validation states, local counts/steps/participation/shuffle seeds,
validation scores and timings. Hash-linked receipts protect checkpoint bytes.
The coordinator commits a round only after all selected clients and scheduled
validation succeed. Incomplete hidden `.pending-*` directories are retained and
ignored on resume. A failed round is replayed in full from the previous commit;
no participant is silently removed or replaced. Do not edit manifests or receipts.

Main validation runs on the complete frozen cohort at round zero, every ten
rounds and round 200. Selection is the earliest strict minimum pooled validation
RMSE in watts. Round-zero persistence participates in selection. Metrics use the
existing common evaluator's physical-unit conversion and `metric_pair`; no final
report or test evaluation is generated. Pilot validation also covers the entire
cohort. A selected export is created only at the configured last round, and its
reloaded predictions are checked against the selected checkpoint on validation.

Run the approved smoke pilot from the project root, stopping after round one to
exercise an actual process restart:

```bash
.venv/bin/python -B federated.py --config configs/experiments/v2_fedavg/gru_smoke_seed42.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v2_fedavg/pilots/gru_smoke_seed42_stage2 --stop-after-round 1
.venv/bin/python -B federated.py --config configs/experiments/v2_fedavg/gru_smoke_seed42.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v2_fedavg/pilots/gru_smoke_seed42_stage2 --resume
```

The first command requires a new output directory. `--resume` checks the exact
manifest, source snapshots, environment, frozen partitions and committed chain.
Preserve completed outputs; choose a distinct directory for any new pilot.

Tests create temporary synthetic fixtures and never run main experiments:

```bash
M100_FEDAVG_REAL_FROZEN="$PWD/output/experiments/v1/frozen" M100_FEDAVG_BASELINE=/tmp/m100-fedavg-stage1/baseline.json M100_FEDAVG_STAGE2_BASELINE=/tmp/m100-fedavg-stage2/baseline.json .venv/bin/python -B -m unittest discover -s tests -p 'test_fed*.py' -v
```

RSS is a process-lifetime high-water mark, including data loading. GPU allocator
peaks and sampled process GPU memory have different scopes and are reported
separately. Internal wall times cover orchestration after loading; an external
`/usr/bin/time -v` measurement includes imports and frozen-data verification.
