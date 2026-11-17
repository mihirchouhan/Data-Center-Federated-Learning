# Report Validation and Research-Integrity Record

**Evidence snapshot:** 17 November 2026. **Publication status:** additive report package; no experiment executed by this task. **Overall reporting decision:** usable with the explicit scientific and concurrent-write qualifications below. This is not a historical reproducibility certificate.

## Scope and inventory

The actual repository is `/media/csedept/cse2014/Project/m100_training_template`; the supplied `csdept` spelling does not resolve. Existing repository rules and the principal overview documents were read. The inventory records 174,672 pre-existing regular files, excluding `.git`, `.venv`, `__pycache__` and `.pytest_cache` and not following symlinks. This count is not a count of fully read or content-verified files.

270 source/configuration/test/tool/documentation text files were content-hashed and source interfaces parsed into 3,165 AST function/class records. The core result extraction read and hashed 116 evidence files. Additional late T5/T6 result/source hashes are recorded separately. Relevant source, config, metrics, manifests, logs and checkpoint bindings were examined. Checkpoints were hashed as bytes, never loaded into a model. The report does not claim every one of the 174,672 files received semantic review.

The main report is approximately 12,000 whitespace-delimited words, including tables, equations, references and Mermaid labels. It has a dedicated seven-table, 25-dimension comparison chapter. The package includes eight PNG/SVG figure pairs, two Mermaid sources, 559 claim rows and an experiment-status register. Reused T4 references are explicitly distinguished from new T5/T6 fits.

## Actual fresh checks

|Check|Observed result|Meaning/limit|
|---|---|---|
|Core actual content-versus-recorded SHA-256 bindings|560/560 match|Includes v5 selected run/evaluation bindings, historical v1 evidence and B1 selected/evaluation artifacts; entries may overlap|
|T4 manifest-listed artifacts|22/22 match|Actual bytes, not regenerated manifest|
|T4 final checkpoints|6/6 match|No deserialization or forward execution|
|Additional T5/T6 bindings|49/49 match|Includes repeated seed/period checkpoint references; not 49 unique models|
|v5 expected identities|24 unique expected records|2 methods × 4 policies × 3 seeds|
|v5 saved target/ordered-identity digests|Common across all 24|Evaluation artifact bytes checked separately; no raw replay|
|v5 method differences|Central lower pooled MAE and RMSE in 12/12 pairs|24 inequalities on one narrow test population|
|B1 aggregate arithmetic|8/8 mean/sample-SD equalities match|Recomputed from three saved seed scores, not fresh predictions|
|B1 population|194,212 saved windows and 976 node records per seed|Common recorded identity digest|
|History execution|9 fit JSONs and 9 seed rows|FF30 only; no-fill matrix not established|
|T4 metric reconciliation|18 stored PASS rows|Historical audit output; audit tool not rerun|
|Claim sources|559 rows have source paths and hashes|Metric-field and bounded source evidence|
|Report structure|Links, figures, fences, placeholders and seven interpretations checked|Rendering of Mermaid depends on viewer support|
|Published-paper arithmetic|Table III rounding and 5.30/.61 ratio consistent|Recalculation from displayed values, not paper reproduction|

The automated report check record contains **26 PASS and 1 FAIL**. The FAIL is the full pre-existing metadata equality check described next; it has not been relabeled PASS. No new model or historical pipeline test suite was run, because relevant suites/tools can initialize models or infer. Existing test outcomes are identified as historical recorded results, never as fresh results of this task.

## Preservation result: qualified, not zero mismatch

All **270 code/config/document text hashes** and **116 core inspected result-source hashes** match their before values. No project source, configuration, prepared dataset, scaler, checkpoint, manifest, lock or historical result was written by this task. All report drafting, summary calculations and plotting took place in an isolated `/tmp` workspace before additive publication.

The metadata comparison of 174,672 pre-existing files found exactly two differences, both T6 progress files:

|File under `output/experiments/t6_architecture_benchmark_v1/`|Before bytes|After bytes|Fields differing|
|---|---:|---:|---|
|`fedavg_lstm_seed43_progress.csv`|14,384|15,450|bytes, mtime_ns|
|`progress.log`|73,929|115,198|bytes, mtime_ns|

Exact nanosecond values are retained in `EVIDENCE/preservation_verification.json`. The T6 log records suite progression and completion; T6 reports/checkpoints appeared during this task. A host read-only process query found an independent `tail -f` monitor, but not a surviving T6 training writer at inspection time. The writer's exact invocation and identity are **UNKNOWN**. The observation is consistent with an independently progressing suite; it is not evidence that this reporting task launched it. No process was stopped, resumed or signaled. No baseline or old file was repaired.

Accordingly, **full-tree zero-change preservation FAILED**. That result is preserved separately from the matching inspected source/result content and from this task's no-existing-file-write scope. Metadata equality does not prove all binary content integrity, and atime was not compared. The report neither deletes this exception nor treats progress-file existence as verified model completion. T6 has its own inspected saved metrics/checkpoint bindings and limitations.

## Important reconciliations and unresolved points

1. **Legacy model identity:** 27.934 W MAE/54.884 W RMSE belongs to the shared LSTM, not FedAvg-GRU. Duplicate `gpu1_test` metrics are not independent evidence.
2. **Separate contracts:** one-hour legacy/v1/v2-v3, six-hour v5, two-output B1 and Candidate1/T4/T5/T6 remain distinct. Different windows, targets, fill rules, scaling populations and optimizer budgets prohibit a merged ranking.
3. **T4 pooled/macro headline:** pooled FedAvg MAE 24.8133 W pairs with pooled persistence 33.0661 W. Macro FedAvg MAE 25.2242 W pairs with approximately 33.3504 W macro persistence. The repeated 24.81 versus 33.35 statement mixes aggregations.
4. **Standard deviations:** some T4 summary CSVs use population SD; this package explicitly computes sample SD, ddof=1, from saved rounded seed scores. This does not alter historical records. Seeds share one dataset.
5. **T4 node exports:** 3,908 rows contain seed42 only for canonical/March views; all-seed aggregate node-win counts are separately stored. Full all-seed prediction/node reconstruction was not performed here.
6. **T4 selection:** pilot candidate sorting uses MAE then RMSE, but within-fit checkpoint updates use strict MAE improvement. Do not claim an implemented RMSE tie-break for every checkpoint selection.
7. **T4 efficiency:** `src/t4_model_improvement_v1.py:526,576` assigns fixed peak-VRAM values; these are not measured GPU memory. Communication figures are model-state payload estimates; binary MiB/GiB were labeled MB/GB in some old text. No measured wire traffic or deployed network savings are established.
8. **Central budgets:** B1 uses exact schedule replay; T4 centralized 20 full epochs are only approximately balanced against a nominal FedAvg exposure estimate. Actual exact trajectory/budget equivalence is not proven by the multiplication.
9. **Candidate1 history:** executed common counts differ from protocol-design counts. The loader gates on maximum 144-bin history, uses finite original training-bin scaling and zeros outdoor channels. This report documents rather than repairs the discrepancy.
10. **History prose:** the saved 1h→6h MAE change is approximately 1.96%, not 3.1%; one 1h seed beats persistence MAE. The stated 200.7→365.7 MB is about 1.82×, not triple. Runtime summaries have inconsistencies; no rigorous Pareto optimum is claimed.
11. **T5 availability:** actual weather-ablation metrics exist and are included. The no-weather condition reuses T4; nine other conditions/seeds are new fits. T5's manifest repeats a reference round198 despite T4's actual 197/198/200 seed selections. The registry hashes, not that shorthand, identify the models.
12. **T6 availability and limits:** reports appeared during this task; the numerical table and model hashes were checked afterward. Reused GRU, newly trained LSTM and centralized XGBoost are distinguished. LSTM has more parameters. GRU's lower MAE is not lower error on every RMSE panel. Raw XGBoost `predict` does not explicitly constrain the iteration range; interaction with early stopping is not independently resolved. No new inference was performed to investigate it.
13. **Broad target audit:** facility cooling power, energy-related registers and PUE channels exist upstream. Their node-level model readiness is not established. Solar radiation was not found; wet-bulb derivation would require assumptions. SLA proxies are not a defined SLA outcome.
14. **B1 current control:** three centralized fits and selected exports exist. The March evaluator adapter exists, but its runtime preflight/authorization/evaluation outputs were not verified present. No centralized B1 March score is invented.
15. **Retrospective periods:** March has been inspected and evaluated. T4 extended and March views overlap the canonical period hierarchy. Large window counts and small seed SD do not establish independent or confirmatory evidence.

## Base-paper verification and limitation

The specifically named uploaded PDF could not be found locally. The official versioned arXiv PDF was fetched into `/tmp`, not into the dataset. Its actual 14 pages were read; title, author list and v3 date match the request. The PDF SHA-256 is recorded in `EVIDENCE/base_paper_identity.json`. Equality with the unavailable uploaded copy is unverified. The report explicitly discloses this substitution.

Page-level checks cover Eq. (1)–(4), Eqs. (5),(10)–(15), Algorithm1, Figures1–8 where relevant, TableII and TableIII. The comparison excludes privacy, key management, cryptographic protocols, cybersecurity and security mechanisms. The printed momentum direction in Eq. (13) is identified as a reproduction question, not silently corrected or used in this project. No paper code, optimization solver or numerical experiment was run.

## Execution and regeneration scope

Read-only commands included repository `rg --files`, bounded `rg -n`, `cat`, `head`, `tail`, Python JSON/CSV/AST parsing and SHA-256 reads, `pdfinfo` and page-bounded `pdftotext`, and read-only process metadata. The official PDF request initially failed in the network-restricted sandbox; a reviewed request fetched only the paper PDF. No dataset/archive network request occurred.

Temporary analysis commands were `python3 -B /tmp/endsem_probe.py`, `python3 -B /tmp/endsem_build.py`, `python3 -B /tmp/endsem_finalize.py` and `python3 -B /tmp/endsem_late_results.py`. Two initial assembly checks caught a history CSV field-name mismatch and broken figure paths introduced by formatting; these were corrected only in new unsealed temporary documents/scripts. Preservation FAIL evidence was retained and investigated, rather than hidden.

Figure generation used `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B /tmp/endsem_figures.py`, with matplotlib Agg and configuration confined to `/tmp`. The portable copy `FIGURES/generate_figures.py` reads only included metric summaries. Do not rerun it inside an immutable published package; copy the package to a new workspace first.

Historical verifier scripts were inspected, not executed. In particular, `tools/audit_t4_integrity.py` initializes models, performs inference and recalculates training statistics; running it would violate this task. Stored PASS audit statements are clearly separated from fresh hash/arithmetic checks.

## Publication and exclusions

The destination `output/final_endsem_report/` was checked unused. The completed package is copied into a new owned staging directory on the output filesystem and renamed with no-replace semantics. `generated_files.json` hashes all package files except itself; its self-hash is deliberately nonrecursive. No old report or artifact is overwritten.

Not executed by this task: training, model initialization, checkpoint loading, inference, prediction scoring, validation/model reselection, GPU computation, scaler fitting, raw-data preparation, archive acquisition/extraction, historical semantic reproduction, baseline repair, selection-lock creation or experiment authorization. Summary arithmetic and plots use existing metric records exclusively.

Historical strict preflight remains **FAILED**. Semantic reproduction remains **UNKNOWN**. No future experiment or operational benefit is approved by this report.
