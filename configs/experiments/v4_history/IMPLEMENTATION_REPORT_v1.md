# History-coverage audit v4: implementation v1

25 October 2026. Implementation and synthetic verification only. The real M100 coverage audit has NOT run. No real longer-history counts or forecasting results are claimed.

## Files created

- `src/history_data_v4.py`: bounded raw reading, protected causal binning/weather helpers, fixed features, exact reference replay and a copy-only variable-history weather mask.
- `src/history_coverage_v4.py`: coverage CLI, reference gate, cohorts, diagnostics, snapshots and verification.
- `tests/test_history_coverage_v4.py`: synthetic tests using temporary Parquet/array fixtures, without models.
- `configs/experiments/v4_history/coverage_audit_v1.json`: new protocol and preservation inventory; read-only file permissions.
- `configs/experiments/v4_history/IMPLEMENTATION_REPORT_v1.md`: this report.

All destinations were unused before creation. `output/experiments/v4_history/coverage_audit/v1/` remains absent. No existing protected source, configuration, dataset, checkpoint, result, receipt or historical expected hash was modified. New source names avoid the protected `src/experiment_*.py` glob.

Configuration SHA-256: `778fa2aedb4d43711ff60f2b0b7d523d0f189bd8b47496ddfc5158c03367b8d9`.

## Protocol and safeguards

Histories are 6/36/144 ten-minute steps; targets remain six future ten-minute power bins. The 979 original node strings and indices remain fixed; no node is filtered or renumbered. Fourteen feature definitions, stable duplicate handling, 300-second causal holds, 50% bin coverage, 30-minute weather as-of age, calendar features and original UTC cuts are inherited from hash-bound metadata/source. Histories may use earlier-split observations available before the origin; targets must remain entirely inside their split, including the test upper boundary. Missing bins stay on the grid.

The original float64 target cumulative sum, float32 cast BEFORE normalization, scalers and persistence coefficients are preserved. Scalers describe the original pooled training-window population, not scalers fitted on the new matched cohort. Weather must be valid before eligibility; masking cannot recover lost windows. The audit creates no masked model tensors; its tested copy-only mask zeros standardized channels 7/8 and leaves inlet temperature, calendar and identity unchanged.

Before accepting longer-history summaries, the real run must reproduce every original one-hour identity, count, ordering, split, scaled target AND six-step input byte across all splits. A mismatch aborts without a completion receipt. **Real replay is pending; only synthetic replay has passed.**

Native cohorts contain all valid windows for each history on the fixed mapping. The primary cohort is their per-split intersection. Eligibility nesting is asserted. `matched_one_hour_reference.json` describes the one-hour reference restricted to those same identities/targets; it contains no forecasting scores. One history-independent target calculation is shared by all cohorts.

Identity is `(node string, forecast-origin UTC datetime64[ns])`, ordered by frozen node index then increasing origin. Binary inventories store little-endian `node_index:u4, origin_utc_ns:i8, target_scaled:f4` (16 bytes/record); `node_mapping.json` resolves the original strings. Ordered identity hashes use the historical M100 encoding; ordered scaled-target hashes are separate.

Every split-grid origin for every original node enters exclusion accounting, including initial-history failures. Mutually exclusive primary precedence: insufficient initial history, target crossing split end, invalid target power, incomplete history. Accepted plus primary exclusions must equal candidates. Additional overlapping counts identify invalid history channels. Marginal raw duplicate/nonfinite/range/coverage diagnostics are not claimed to uniquely explain every excluded window.

Outputs report native and matched counts, pairwise overlaps, retention versus native and original one-hour cohorts, per-node quantiles/zero coverage, origin/date coverage, and training eligibility (1,024 windows and seven training days). Eligibility is reported without filtering. `model_comparison_ready` requires nonempty matched splits and every original client meeting matched training thresholds; it is not permission to train.

## Verification actually performed

| Check | Result |
| --- | --- |
| Synthetic command | `.venv/bin/python -B -m unittest discover -s tests -p 'test_history_coverage_v4.py' -v` |
| Final project test run | 26 passed; zero failures/skips; 2.072 seconds |
| Existing thesis verifier, before and after | 3,713 pre-existing files, 22 thesis artifacts, 82 direct sources; zero mismatches |
| Fresh preservation inventory, before/after | 3717/3717 original files; zero content/size/mtime/mode mismatches |
| Real-input preflight only | 14 frozen arrays, 20 raw Parquet hashes, 3717 protected files, 8 source/test fingerprints; zero mismatches |
| Documentation checks | Paired code fences, expected files present, future audit output absent |

The preservation inventory excludes directory names `.venv`, `.git`, `__pycache__`, `.pytest_cache`; access times are not compared. No unexpected project additions were found. New configuration hashes record existing provenance without changing historical receipts.

Tests demonstrate original-window parity for all lengths; gaps; node separation; target boundaries and past context; causal inputs; invariant targets; nesting/intersections; reconciled reasons; masking/scalers; identity encoding/order; zero coverage and eligibility; bounded-reader stable duplicate/null parity; resource-limit aborts; raw/frozen/protected hash failures; reference-gate failure; saved matched targets; partial outputs; output tampering; verification without rebinning; and preservation.

The real-input preflight only hashes/checks inputs and schemas; it does not bin raw data, calculate real coverage, run inference or score predictions. Synthetic runtime is not a real-data benchmark.

## Estimated resources and limits

One ordered scan per local metric spools temporary raw shards; one node is stable-sorted at a time. Settings: 32 nodes/shard, at most 262,144 Arrow rows/batch, 20,000,000 rows/shard and 2,000,000 rows/node. Exceeding a limit aborts rather than truncating or filtering. The eight-metric float64 grid is disk-backed: approximately 508 MiB for 979 nodes and 8,496 bins. No dense 36/144-step model dataset is created.

The largest metric's existing Parquet footers total approximately 249.6 million rows, or 4.65 GiB of 20-byte raw records before filtering. Estimated peak scratch/grid/identity storage is around 6 GiB; allow at least 10 GiB free disk. Four full-cohort identity inventories would occupy at most approximately 0.285 GiB before summaries/snapshots. Several GiB of available RAM are advisable; 4–8 GiB is a planning allowance, not a measured peak or guaranteed memory cap. Mapped pages, Arrow buffers and sort copies contribute to RSS.

Real runtime remains unmeasured. Decoding, sorting, temporary writes and full-project integrity hashing dominate. History masks reuse the binned grid. `resources.json` will record actual Linux process peak RSS and audit wall time; wall time starts after initial preflight, whereas RSS covers process lifetime. No GPU is needed.

## Manual commands — real run requires review/approval

Run from the project root using the existing environment, in the foreground. Read-only preflight:

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B -m src.history_coverage_v4 \
  --config configs/experiments/v4_history/coverage_audit_v1.json \
  --reference-data-dir output/experiments/v1/frozen \
  --raw-data-dir data/extracted/b067a6ea8a820f12dbab8ee65f0d3fb862e3f6ea0a8646614d09247f7c71b357 \
  --output-dir output/experiments/v4_history/coverage_audit/v1 \
  --preflight-only
```

After approving the implementation and successful preflight, execute the actual audit:

```bash
.venv/bin/python -B -m src.history_coverage_v4 \
  --config configs/experiments/v4_history/coverage_audit_v1.json \
  --reference-data-dir output/experiments/v1/frozen \
  --raw-data-dir data/extracted/b067a6ea8a820f12dbab8ee65f0d3fb862e3f6ea0a8646614d09247f7c71b357 \
  --output-dir output/experiments/v4_history/coverage_audit/v1
```

Only after completion, verify saved artifacts without rebuilding bins:

```bash
.venv/bin/python -B -m src.history_coverage_v4 \
  --config configs/experiments/v4_history/coverage_audit_v1.json \
  --reference-data-dir output/experiments/v1/frozen \
  --raw-data-dir data/extracted/b067a6ea8a820f12dbab8ee65f0d3fb862e3f6ea0a8646614d09247f7c71b357 \
  --output-dir output/experiments/v4_history/coverage_audit/v1 \
  --verify-only

.venv/bin/python -B docs/thesis/v1/code/verify_thesis.py
```

Expected statuses: `coverage_complete`, then `verified_complete`; `reference_replay.json` must contain `exact_match`. Inspect `summary.json`, `matched_one_hour_reference.json`, `per_node_coverage.csv`, `origin_coverage.csv`, `date_coverage.csv`, `exclusions.json`, `source_quality.json`, `resources.json`, `preservation.json`, manifest/config/source snapshots, binary inventories and `completed.json`.

## Recovery, limitations and stop conditions

No resume option exists. Occupied completed/partial directories are rejected before preflight. Preserve interrupted outputs and share the error; select a separately checked unused version for an approved retry. Ordinary exceptions/Ctrl-C clean only newly created temporary scratch; SIGKILL may leave scratch. Do not delete historical artifacts or edit expected hashes.

Stop for hash/metadata mismatches, any one-hour replay discrepancy, changed targets/mapping/cuts, duplicate/unordered identities, non-nested eligibility, unreconciled exclusions or a reading-limit violation. Empty or inadequate matched cohorts require a reviewed protocol decision before fitting. Synthetic success does not establish real 24-hour retention. The test period has already been observed and coverage matching is retrospective. No improvement, energy saving or privacy guarantee is claimed.
