# Report Validation — Updated Version v2

**Review date:**17 November 2026, Asia/Calcutta. **Report decision:** Updated documentation published with explicit research/provenance caveats. This is not a training-readiness approval or a historical reproduction certificate.

## Starting document, backup and preserved structure

The source document is the existing Markdown report `output/final_endsem_report/FINAL_PROJECT_REPORT.md`, not the older Desktop project. Its 16 numbered chapters, existing academic discussion and references, separate result panels, and figures 1–8 are retained. Original root reports/registers/inventory were copied to `BACKUP/` before editing the new workspace. The original package's41 inventoried file hashes match. Original model/data/experiment files were not overwritten.

The new version is approximately 14,100 words, reflecting additions to the original approximately 12,000-word document. It has 20 chapter-numbered tables, 12 rendered PNG/SVG figure pairs, editable Mermaid workflows, 190 experiment-status rows and 1,162 evidence/claim rows. The extra length preserves the requested starting document rather than rewriting it to a new outline. `CHANGELOG.md` and the unified diff identify the modified chapters. `BIBLIOGRAPHY.md` retains the seven previously verified references; only spacing defects were repaired. No new external reference was invented.

## Repository discovery and current status

A new read-only scan records 174,775 pre-existing regular files and reads/hashes 279 source/config/test/tool/document files. AST inspection records interfaces without importing project model or training modules. The full old inventory remains a dated v1 snapshot; additions since it are recorded in `EVIDENCE/UPDATE_V2/new_files_since_previous_snapshot.csv`,61 paths outside the original report package. They include 9 new source/test/tool files and newer coverage/preflight, T6 and T7 artifacts.

The update retains legacy LSTM, historical model/weather and matched-controls,24-fit v5, B1 two-output/persistence,older history/temporal work and T4/T5/T6 as separate studies. T7 is newly incorporated with 99 saved evaluations of three frozen models, not 99 fits. New T1.1v2 coverage and T1.2preflight artifacts are audited as saved records with unresolved manifest/eligibility issues. The new two-layer residual history implementation has partial external execution evidence, not a completed result matrix. No complete centralized B1 March comparison, FedProx matrix, participation sensitivity, validated action-response simulator or energy/SLA policy evaluation was found.

## Fresh verification results

|Layer|Actual outcome|Scope|
|---|---|---|
|Report consistency/preservation-scope checks|22 PASS,0 unexpected failures|Structure, backups, references, tables, figures, bound source contents and original package|
|New saved numeric/count checks|11 PASS|33 groups ×4 metrics ×mean/sampleSD =264 summary equalities,99 paired degradation differences,25 missingness equalities, selected node summaries and client totals|
|Existing/new manifest-content comparisons|638 PASS,9 FAIL out of647|Nine failures are in the two **new history-audit/preflight manifests**, retained below|
|Original report package|41 inventoried artifacts unchanged|Actual bytes; original root manifest remains untouched|
|Original source/result baselines|270 source/config/test/doc hashes and116 result-source hashes unchanged|Bounded content scope, not full-project binary integrity|
|Current whole-tree metadata equality|FAIL:1 concurrent source edit|174,775 existing files compared; no baseline rewritten|
|Current final source snapshot|279 audited bindings stable at final verification|Initial and later source hashes are separately retained|

The 638 matching comparisons include the 560 prior core checks and T4's 22 manifest/six checkpoint bindings, plus newer checks. These counts include overlapping references and are not counts of unique models. New files were also checked against their manifests where present. No failed comparator was converted into a PASS or removed from evidence.

The 11 new arithmetic checks use stored metric scores, node aggregates and count tables. They do not recompute predictions from models or decode raw telemetry. Sample SD uses ddof=1. Summary agreement allows only the 5e-5 W rounding interval for existing four-decimal CSV fields; no scaler or scientific acceptance tolerance is changed.

## Nine failed new-manifest comparisons

|Referenced path|Expected by new manifest|Actual byte SHA-256|
|---|---|---|
| output/experiments/t1_1_common_history_coverage_audit_v2/progress.log | `4f6c449c25227aa71c26f04c64bf2e5ce6c94ad515cb6b3332ebbc61d3680e9a` | `d35dc4c966b8b2522ef1d3d9fcc425846f1f4bbd471e0c448d26eaff128eb714` |
| src/t1_1_common_history_coverage_audit_v2.py | `0eaae89ff30ae73397984f478a87b140ad517dcbdfd2ec3eb5e57eeef621f2bb` | `1ec608c0400cbc6b481705603347eb9ef645fd8ae9c2f3ef83c64afddb0f24ac` |
| tests/test_t1_1_common_history_coverage_audit_v2.py | `b693fa95f32cb071e7a5d97aa356df3c082725514f762095f7004f13454ae7c6` | `55e04eef7d0884e2686dfaeca629aa79518a08b604f9a76c41e4653a9f8af709` |
| output/experiments/history_coverage_audit_v1/coverage_by_history.csv | `ecda9e97f02bfa450d9dd1367015db6ea07ee974f85e49f652bfa3240eb0f7c2` | `bff5eeef9c70d9b1bda7136aff1a99a880d3f0a24c9c4b8a14e0d0186ec40056` |
| output/experiments/v5_missing_data/model_comparison/v1/cohorts/bins/train_features.npy | `fe090fa7d10e543a6d1a938c0fa370ca9b626f2f252cf1b15ecb3aa4c2ffdb1d` | `d97466aba74227f18f948998281dbd9c23d53468285640874a5f7b710e51fe72` |
| output/experiments/t1_2_history_length_preflight_v1/progress.log | `6b9760773d57f005fbc30f9a2656910fc9ba278297b8f95c510803407e7e651e` | `fa602e16ca6ddf7b8910b770b105abb485eb94cc5e2bd82bb3669f1bf29806a2` |
| src/t1_2_history_length_preflight_v1.py | `0d68db0bfd3989c9918fc9d10787e7f72e90c888e2c0205ae7d62025175cfc2c` | `da28069c0b7f049f5068f1011c84029f07dbafeb845151830ea09817594c4094` |
| tests/test_t1_2_history_length_preflight_v1.py | `c76fb97fa5290b0e513d3ae071fdfc745bc09f187a0ca3e7e814d45d7d3d750c` | `dc24dae78c2e4b012f5e53a51bf66857732b14eb57998b2827a6a866af1fb4d1` |
| output/experiments/v5_missing_data/model_comparison/v1/cohorts/bins/train_features.npy | `fe090fa7d10e543a6d1a938c0fa370ca9b626f2f252cf1b15ecb3aa4c2ffdb1d` | `d97466aba74227f18f948998281dbd9c23d53468285640874a5f7b710e51fe72` |

These are nine comparisons over eight unique referenced files. The historical coverage CSV's actual hash matches the original report's `result_source_hashes.json`. The training array's actual hash matches `cohorts/completed.json#/artifacts/bins/train_features.npy`. Thus the new manifests disagree with independently recorded historical bindings; this check does not show that those two historical inputs changed.

The exact reason for the new source/test/log expected hashes is **UNKNOWN**. Their corresponding historical execution snapshots have not been recovered, and no intent is inferred. The new coverage/preflight execution provenance remains unverified. Count arithmetic can be inspected and quoted as a saved record without treating the manifest as verified, but it cannot approve a new fit. No new baseline, expected hash, lock or receipt was regenerated. The original historical preflight remains FAILED and semantic reproduction UNKNOWN.

## Current execution and concurrent preservation qualification

At the start of this review, the new controlled-history output was absent. During this task, `src/t1_2_controlled_history_benchmark_v1.py` changed from 51,971 to 52,528 bytes, and a runtime directory/log appeared. Its initial SHA-256 is `5bf10f0969ae2c88fe557a1d29bd8349bd2d08c43dcf08ef92f41f56c86af65b`; the later audited value is `6d71bcf516b20a6edb7d2a2df828b74d3aade54b2d74daf719e1bf4981584022`. Metadata details are retained in the current preservation JSON. The exact writer/invocation is UNKNOWN. This task issued no source edit, training/model command, process signal or GPU operation.

The fixed log snapshot at **2026-11-17T19:52:19.224609+05:30** records **H=6 (one-hour), seed42**, through round **34/200**. This is a progress observation, not a completed checkpoint/evaluation claim. The logger's own UTC label was not independently validated; the snapshot timestamp uses the client timezone. Provisional validation scores are excluded from final numerical result tables. Later runtime progress is outside this fixed snapshot; the report grants no execution authorization or retrospective approval.

Accordingly, full-tree zero-change preservation is not claimed. The scoped old source/result and original-report byte checks pass; this single newly added runner changed concurrently. There were no additional unexpected pre-existing metadata changes at final verification. Atime, directories, `.git`,`.venv`, `__pycache__` and `.pytest_cache` are outside the regular-file metadata comparison. New external runtime files are not retroactively inserted into the initial preservation baseline.

## Scientific/reporting corrections

- T7 clean canonical RMSE is 57.5502 W in seed/summary CSVs, not 44.97 W in the Markdown. CSV values take precedence; the old report/artifact is not edited.
- T7 uses FF30 then standardized-mean **input imputation** while targets remain observed. Degraded-input persistence has a changed anchor; it must not be mislabeled unchanged clean persistence.
- Fixed seed42 masks are shared across model seeds. Reported seed SD is not uncertainty across independently simulated outage realizations.
- Burst start positions are distinct, but intervals can overlap. Requested dropout rate is not realized removed-observation rate.
- The missingness denominator is 7,154,532 node-grid/channel values, not the narrative's 7,138,530 or windows×history count.
- The current T7 file defines 4 test methods; its claimed 21 passed tests are not established. Two instantiate a manager that reads real arrays and recomputes training statistics, so that suite was **not executed** during this documentation task.
- `tools/audit_t7_integrity.py` checks saved metric arithmetic and checkpoint hashes; its broad prior-baseline check is artifact existence, not a content audit of all prior outputs. It also writes the historical report, so it was inspected, not run.
- The new preflight's 979 active clients have 12–985 common training windows; all 979>=1,024 eligibility flags are False. Activity and eligibility are distinct; the source does not establish the claimed seven-day condition.
- The new history runner uses MSE; T4's winning model uses L1. Calling those identical is false. Its new two-layer residual experiment must remain separate from the older completed one-layer/direct-power history panel.
- FF30 is the largest saved common cohort here, not the only nonzero one. No-fill/FF10/FF20 also have positive intersections. A Ganglia-daemon failure is not proven by missingness counts alone. Three historical FF10 validation rows additionally label146 common windows across979 common nodes; those node counts are impossible and are not used.
- T4 pooled 24.8133 W MAE still pairs with pooled persistence 33.0661 W; macro 25.2242 pairs with macro 33.3504 W. Earlier mixed-aggregation and population-SD limitations remain documented.
- T5/T6 measured outcomes remain conditional: no universal irrelevance of weather or architecture/Pareto superiority is established. No Transformer result is present in the verified T6 matrix.

`UNVERIFIED_CLAIMS.md/.csv` lists 19 unsupported or contradicted claims, evidence and how the report handles each. The old report's other qualifications remain in its backed-up validation record.

## Paper and bibliography

The same actual official arXiv v3 PDF used in v1 remains available in the isolated temporary workspace. Its 1,069,514 bytes and SHA-256 `c081a18f0a7ac2ea920698904734f667981fe2f707fbb5d1d869efe8ec4d8aef` match the earlier identity record. Its page-level mathematical and numerical analysis is retained; no new network request was needed. The specifically named uploaded copy remains unavailable for byte comparison, and that limitation is preserved.

All seven base-paper comparison tables remain. T7 is characterized as input-corruption forecasting analysis, not the paper's adaptive model-acceptance or physical optimization framework. R²/cost error/runtime are not ranked against watt MAE/RMSE. Privacy, cryptographic and cybersecurity mechanisms remain out of comparison scope.

## Commands, outputs and forbidden operations

Read-only inspection used bounded `rg`, `cat`, `head`, `tail`, Python JSON/CSV/AST parsing, metadata inspection and SHA-256 reads. Temporary assembly ran `python3 -B /tmp/endsem_update_audit.py`, `/tmp/endsem_update_v2_content.py`, `/tmp/endsem_update_v2_docs.py`, `/tmp/endsem_update_current_status.py` and `/tmp/endsem_update_v2_verify.py`. Initial assembly detected a relative-path handling issue and stopped; only new temporary reporting code was corrected. Manifest failures were retained and investigated, not fixed.

Static figures ran with `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B /tmp/endsem_update_v2_figures.py`, matplotlib Agg and configuration under `/tmp`. Four new PNG/SVG pairs were created from saved metrics/counts, alongside the eight unchanged originals. Regeneration scripts are report utilities, not experiments; copy an immutable package into a new workspace before regenerating.

The original report package and experiment files remain untouched by this task. Only the new `output/final_endsem_report_v2/` package is published, through a fresh owned staging directory and atomic no-replace rename. Its generated-file inventory excludes its own recursive hash; the original inventory remains under `BACKUP/`.

Not executed by this task: training, model initialization, forward/backward/optimizer calls, inference, prediction scoring, GPU diagnostics/workloads, scaler fitting, raw preparation, archive downloading/extraction, historical semantic replay, baseline/hash repair, lock/receipt creation, process launch/resume/termination or protocol approval. Reading externally written progress is not running that experiment.
