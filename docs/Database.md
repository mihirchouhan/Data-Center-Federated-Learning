# Artifact storage and persistence

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

[NOT APPLICABLE] No traditional database, ORM or migration framework was found in the inspected research application. Storage uses TAR, Parquet, NPY/NPZ, model exports, JSON receipts, CSV tables and logs. The following is an artifact catalog, not a fictional SQL schema.

## Storage map

| Role | Project-relative location | Format / authority |
| --- | --- | --- |
| Jan–Feb archives | `data/archives/22-01.tar`, `22-02.tar` | Published checksums and extraction receipts |
| Extracted source | `data/extracted/b067a6ea8a820f12dbab8ee65f0d3fb862e3f6ea0a8646614d09247f7c71b357/` | `year_month=YY-MM/plugin=.../metric=.../*.parquet` |
| Original prepared data | `output/m100_all_nodes/prepared/` | `dataset.npz`, `preprocessing.json` |
| One-hour frozen cohort | `output/experiments/v1/frozen/` | Hashed NPY arrays, scalers and contract |
| v1 comparison | `output/experiments/v1/comparison/` | Fit manifests/receipts, comparison.json |
| Historical FedAvg | `output/experiments/v2_fedavg/main/` | Authoritative seed42 folder ends `seed42b`; round ledgers |
| Historical central control | `output/experiments/v3_centralized_sgd/main/` | `gru_masked_seed{42,43,44}_schedule_matched`; block ledgers |
| v5 cohorts | `output/experiments/v5_missing_data/model_comparison/v1/cohorts/` | Retained bins/ages/power, native/common IDs and targets |
| v5 runs | `.../model_comparison/v1/full_runs/` | Flat historical names; canonical identity in manifest |
| v5 lock / test | `.../model_comparison/v1/selection_locks/v2/`, `test_evaluation/v2/` | Locked selections; 24 prediction/metric panels |
| Saved analysis | `.../model_comparison/v1/results_analysis/` | Sealed reports, plots, provenance and analysis scripts |
| Phase 2 design | `configs/experiments/phase2_multioutput/v1/` | JSON/matrix/design manifest, not March model data |

Ellipses abbreviate `output/experiments/v5_missing_data`; do not paste them into commands.

## Data schemas

Raw selected metrics contain node, timestamp and value. Target samples recorded in [Phase 1.4](../output/phase1_4/temperature_semantics.md) have `node:string`, `value:float 32`, `timestamp:timestamp[ms,tz=UTC]`. Unit metadata is not inferred solely from raw numeric values. Weather is a site-level series, distinct from node-inlet temperature. Inspect each new source schema instead of assuming historical compatibility.

The original frozen arrays are `X_{train,val,test}.npy`, `y_*.npy`, `node_*.npy`, `origin_*.npy`, plus `node_names.npy` and `power_baseline_coefficients.npy` (14 total). Contract records shapes, dtypes, hashes, feature order, cuts and scalers. Inputs have trailing shape `(6,14)`, targets `(1,)`, origins UTC-interpreted datetime64[ns].

The [v5 contract](../output/experiments/v5_missing_data/model_comparison/v1/cohorts/contract.json) defines:

- `bin_contexts`: e.g. `bins/train_features.npy`, `train_fill_ages.npy`, `train_observed_power.npy` for historical values, original-source ages and separate observed labels.
- `cohorts`: native policy train IDs plus shared `common/all_policies/test/{window_ids.npy,targets_kw.npy,targets_scaled.npy}`.
- Structured IDs with `node`, `node_index`, `origin`; string/index must agree with mapping.
- `client_budgets.json` native counts/reference `m_k`, `scalers.json` transforms, mapping and completion inventory.

These are inspected file interfaces, not proposed database tables.

## Ledger and evaluation schemas

`training_io.open_run` creates `manifest.json`, `configuration.json`, `source/`, `rounds/`. Numbered `rounds/0000` through `0200` contain `state.pt`, `sampled_rows.npz`, `receipt.json`. Trace fields include node_indices, offsets and rows. Receipts bind previous receipt, manifest, state and trace hashes. `selected/{model.pt,receipt.json}` stores the validation-selected export; `completed.json` binds selected/final receipts.

v5 evaluation stores per-identity `metrics.json`, `per_node.json`, `predictions.npz`. Prediction fields are `predictions_w`, `targets_w`, `persistence_w`, `ids`. Root results.json, copied selection_lock.json, manifest.json and completed.json bind outputs. A filename or completion flag is insufficient without exact inventory and actual hash checks.

## Lifecycle and access

Raw → prepared → frozen contract → bound fit → validation-selected export → pre-test lock → predictions → saved-artifact reports. Preserve every predecessor. No destructive migration or garbage collection is specified. Exclusive creation/fsync/read-only modes/atomic publication are application safeguards, not hardware immutability or independently signed provenance.

Never add files inside sealed directories: exact inventory checks can reject additions. Use unused siblings. Do not delete/rewrite `data/.cache.lock` or other operational locks to satisfy historical inventory. Trusted local checkpoint loading uses `weights_only=True`; do not treat arbitrary uploaded checkpoint files as safe.
