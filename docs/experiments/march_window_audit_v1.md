# March streaming coverage audit v1

Additive coverage-only implementation; no training, inference, scoring, scalers or dataset freeze. Uses the unchanged Phase 2 v1 protocol, original 979-name mapping and fourteen inputs. Jan--Feb, March-only and combined availability use identical fixed splits: March-only is test-only, Jan--Feb has no March test, and only the combined scenario can support the declared train/validation/test design. No chronological cuts are recomputed.

`src/march_window_audit_v1.py` uses bounded Arrow batches and temporary node shards, stable keep-last deduplication, existing range checks, `multinode.weighted_bins`, `preprocess._weather_asof` and `history_data_v4.node_features`. Raw contributors are restricted to each declared split before binning. Whole histories and targets remain inside that split; hourly origins use six observed target bins. No missing CPU/memory/weather feature is dropped. The weather mask never expands eligibility.

Memory is the original ratio of separately time-weighted free/total means aligned on the same node/UTC bins, not mean-of-ratios or an invented synchronous raw join. Invalid or nonpositive denominators and fractions outside [0,1] are rejected. Simultaneous raw support within a bin is not established.

New test cases cover coverage/holds/ranges, null duplicates, weather cutoff/age/reset, aligned memory, gap-safe histories, targets/splits, masks, eligibility, identity/order, node separation, bounded shards, unsafe paths, tamper/no-clobber and an end-to-end synthetic Parquet audit. Actual execution results and preservation evidence belong in the new audit report; this guide does not claim an unrun test passed.

## Commands

Run from `/media/csedept/cse2014/Project/m100_training_template` with the existing venv.

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_march_window_audit_v1.py' -v
.venv/bin/python -B -m src.march_window_audit_v1
```

Default is read-only preflight. Explicit `--execute-audit` writes only to the unused `output/march_window_audit/v1` destination; an occupied path is rejected. This user task authorizes the coverage audit, not model operations. A repeat requires a reviewed unused versioned destination.

```bash
.venv/bin/python -B -m src.march_window_audit_v1 --execute-audit
.venv/bin/python -B -m src.march_window_audit_v1 --verify-only
```

Verification checks exact artifact inventory/hashes, saved window identities/targets/splits/order and shared-scenario equality. It does not run a model or certify historical semantic reproduction. It verifies this audit's frozen metadata rather than granting a training permission. On failure preserve the partial output; no automatic overwrite/resume is implemented. Temporary owned scratch spools are removed when their stage exits; source data and failed output reports remain.

## Resource bounds and evidence

At most one metric is spooled at a time, <=65,536 Arrow rows per batch, <=16 nodes per shard, <=1,000,000 raw rows per node and <=12 GiB spool payload. Per-split operational bins use disk-backed arrays; dense 36-step training datasets are never built. Free-space preflight is not authoritative quota confirmation; I/O/quota failure aborts without modifying source data. Resource measurements are recorded after the actual run.

Historical global preservation preflight remains FAILED and is neither executed behind its gate nor repaired. Fresh scoped preservation checks are separate. March is retrospective data, not automatically untouched confirmatory evidence. Sensor calibration, hardware continuity, privacy and energy benefits remain unproven. [Original design](phase2_multioutput/v1/PROTOCOL.md), [methodology](../Design.md), [source integrity investigation](../../output/phase1_integrity_review/integrity_review.md).
