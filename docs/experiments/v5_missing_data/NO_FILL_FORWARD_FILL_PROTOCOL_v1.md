# V5 protocol: no-fill and bounded forward-fill for six-hour GRU forecasting

Version 1 — 26 October 2026 — **proposed experiment, not implemented or launched**.

Project root: `/media/csedept/cse2014/Project/m100_training_template`.

This document distinguishes verified saved evidence, deductions from the inspected algorithm, and proposed future work. This documentation task performed no raw-data reconstruction, model training, inference, or prediction scoring. It creates this document only. All historical sources, configurations, arrays, checkpoints, reports and expected hashes remain unchanged.

## 1. Decision and research question

The existing training pipeline cannot run a fair six-hour comparison unchanged. New additive data/runner/evaluation adapters are required. The proposed primary setting is a six-hour history and a 20-minute forward-fill limit, with no fill as reference and 10/30 minutes as sensitivity settings. The choice is informed by already-observed coverage, not demonstrated predictive superiority; it requires review before implementation.

**A fully matched-window comparison would be degenerate.** Let `K(g,s)` be the eligible `(node string, origin UTC ns)` identities for gap limit `g` and split `s`, with `g=0` meaning no fill and ignored weather. The audited implementation asserts `K(0,s) ⊆ K(1,s) ⊆ K(2,s) ⊆ K(3,s)`. Where two policies are both eligible, their non-weather input values are identical: both use the same last original finite observation, and a larger limit only fills additional, previously invalid bins. Weather is masked identically. On `K(0,s)`, no required channel needs filling at all.

Consequently, matching training, validation AND test windows across all policies, and also matching scaling, initialization and schedules, supplies identical tensors to all four fits. It cannot estimate an imputation benefit. Matching only filled policies has the same issue on their shared eligible windows. Fifteen small in-memory synthetic equality checks passed during this documentation task; this is also a direct consequence of `causal_forward_fill()` and `eligible_origins()`, not a measured forecasting result.

**Recommended estimand:** the effect of access to each policy's enlarged training pool under a fixed computational budget, evaluated on identical no-fill validation/test windows. This intentionally varies training-pool membership and training-input imputation together. It does not isolate the numerical accuracy of the imputer, or measure the effect of simply adding more SGD steps. Recovered-window coverage and conditional forecasting are a separate secondary analysis.

Hypotheses, to lock before fitting:

- H1: access to the 20-minute training pool changes pooled next-hour RMSE on the common no-fill test cohort relative to no-fill training, under matched budgets. Improvement is a hypothesis; a null or adverse result is valid.
- H2: 10/30-minute policies may trade recovered training coverage against stale-input bias. Their predictive ordering is unknown; report both without selecting a test-winning limit.
- H3: FedAvg and a matched centralized-SGD control may respond differently to training-pool access. Estimate paired contrasts within each algorithm; do not presuppose a federated advantage.

## 2. Verified evidence and scope

Completed inputs:

- `output/experiments/v5_missing_data/diagnostic/v1/{summary.json,masked_coverage_by_node.csv}`.
- `output/experiments/v5_missing_data/forward_fill_audit/v1/`, including source/config snapshots, reference replay, per-node counts, source quality, resources and completion receipt.
- Original reference: `output/experiments/v1/frozen/{contract.json,preprocessing.json}` and its fourteen array files.

Read-only `forward_fill_audit --verify-only` passed: ten recorded artifacts, current execution sources, baseline diagnostic binding and original-input preflight. The preflight checks fourteen frozen arrays, twenty raw-input hashes, eight v4 source/test fingerprints and 3,717 protected files. No raw bins were rebuilt. The two CSVs contain 8,811 and 26,433 unique node/history/split/policy rows, respectively; their node strings/indices match the original mapping, and their aggregated counts reconcile with the saved summaries. Original strict replay is recorded as exact: 4,506,133 training, 199,670 validation and 75,566 test windows, including target/input bytes.

Six-hour native coverage, recomputed from saved CSV counts only:

| Policy | Training windows | Validation windows | Test windows | Train nodes | Validation nodes | Test nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0: no fill, weather ignored | 2,310,256 | 31,448 | 6,590 | 979 | 929 | **570** |
| G1: maximum 1 missing bin | 2,996,160 | 80,679 | 10,379 | 979 | 979 | 762 |
| G2: maximum 2 missing bins | 3,459,118 | 139,874 | 11,821 | 979 | 979 | 802 |
| G3: maximum 3 missing bins | 3,867,660 | 196,149 | 13,184 | 979 | 979 | 827 |

These are availability findings, not errors or model results. In particular, neither 6,590 nor 11,821 is the historical 75,566-window test cohort; historical RMSE/MAE values must not be substituted as baselines here.

### 2.1 Shared cohorts and what is not yet materialized

The following intersection sizes follow exactly from saved native counts and the hash-bound runtime subset assertions. They have **not** been independently enumerated or frozen as new ID arrays in this task; the coverage audit saves counts, not per-origin membership lists.

| Policies intersected, per split | Shared training | Shared validation | Shared test | Shared validation/test nodes |
| --- | ---: | ---: | ---: | --- |
| G0/G1/G2/G3 | 2,310,256 | 31,448 | 6,590 | 929 / 570 |
| G1/G2/G3 | 2,996,160 | 80,679 | 10,379 | 979 / 762 |
| G2/G3 | 3,459,118 | 139,874 | 11,821 | 979 / 802 |

The future builder must explicitly join, sort and hash `(node string, forecast-origin datetime64[ns] interpreted as UTC)`, verify target bytes/split assignment, and reproduce these counts. Count equality alone is insufficient to establish identical windows. Preserve frozen node order, then increasing origin within node. Do not join by array row number across native policies.

Primary validation/test are `V0=K(0,val)` and `T0=K(0,test)`, for all policies and both algorithms. Shared training `K(0,train)` supplies reference client budgets; native `K(g,train)` supplies each policy's intentionally different training pool. This distinction must appear in every report.

## 3. Experimental matrix and controlled budget

Proposed matrix: two algorithms × four policies × seeds 42/43/44 = **24 fits**, not counting future engineering smoke tests. Primary comparisons are G2 versus G0 for both algorithms (12 fits). G1/G3 are predeclared sensitivities (12 additional fits). Do not launch either stage without separately approved implementation, tests, smoke results and cost review. Do not run duplicate all-common-training fits merely to label them different gap policies.

| Factor | Proposed locked setting |
| --- | --- |
| History/target | 36 completed ten-minute bins; next six observed power-bin means |
| Stored input | Original fourteen channels; node index at 13; standardized weather 7/8 zeroed |
| Model | Existing residual `SequenceRegressor(model='gru')`; two 64-unit GRU layers, 979×8 node embedding, linear head; 21 recurrent input values |
| Initialization | Fresh seeded parameters, zero residual correction head; exact paired initial-state fingerprints; no trained checkpoint import |
| Optimizer/loss | SGD lr 0.01, momentum 0, weight decay 0; mean squared error of standardized targets |
| Batch sizes | Training 512 with partial batch retained; validation 1,024 |
| Clients | 979 original compute-node identifiers; 98 sampled uniformly without replacement per round/block |
| Cap/validation | 200 rounds/blocks; validation at 0,10,…,200 on all of V0 |
| Selection/stopping | Earliest strict minimum pooled validation RMSE W, including initial persistence candidate; no patience termination; complete fixed cap |
| Seed streams | Existing initialization/client-sampling/local-shuffle domains 0/1/2; new separate domain 3 for drawing a round's training subset |
| Fixed state | `baseline_coefficients` unchanged; ordinary full-table node-embedding aggregation |

### 3.1 Explicit proposed departure from the historical full-local-epoch protocol

Using one complete native local epoch per selected client would increase steps/exposures as gap limits increase and confound training access with compute. Instead, define `m_k = |K(0,train) for node k|` and lock these counts before fitting. On each selected client session:

1. G0 visits its `m_k` complete no-fill rows once in shuffled order.
2. G1/G2/G3 uniformly sample `m_k` distinct rows from their own native training pool, without replacement within that session, then shuffle them. Repeat sampling independently across rounds using locked deterministic seeds.
3. Record native pool count, processed count `m_k`, ID-order hash, sample/shuffle seeds, batch sizes and steps. The session is a **reference-size sample pass**, not a full epoch over the enlarged native pool.

Use the same sampling/shuffle code path for G0 too, sampling all its own rows. Identical candidate pools must yield identical plans; otherwise a difference in scheduling would undermine the common-input invariance control.

`m_k` ranges from 1,598 to 3,396. Each policy has at least this many rows for every node. This design deliberately changes the historical full-native-pass protocol and requires explicit methodological approval. It studies training-pool access under fixed compute; it can replace some complete training examples with recovered/imputed examples in a session, so it is not a pure causal estimate of imputation quality.

For each seed, all policies/algorithms use the same ordered sampled clients and `m_k` batch sizes. Within a policy, the centralized control and FedAvg replay identical sampled row IDs and ordering. Across policies, row IDs intentionally differ, but random streams and counts are paired. Preserve all remainders (`drop_last=False`).

FedAvg: start every client from the same immutable round state; reset local SGD; weight returned states by processed `m_k`, not the larger native pool count. The unchanged generic `aggregate_fedavg()` can validate against an explicit frozen per-round subset-count manifest; do not mislabel native membership as processed counts. Centralized SGD: one continuing global model processes the same planned client/batch sessions sequentially without averaging. Initialization, sample plan, optimizer settings, exposures and validation schedule match; optimization trajectories need not match.

Lightweight calculations using the inspected sampler, original node order and saved no-fill client counts give these **proposed budgets, not executed runs**:

| Seed | Example exposures per method/policy | SGD steps per method/policy |
| --- | ---: | ---: |
| 42 | 46,268,600 | 100,038 |
| 43 | 46,238,350 | 100,014 |
| 44 | 46,275,837 | 100,074 |

The future planner must independently reproduce these numbers before fitting. Distinct training rows, repeated exposures and optimizer steps are different quantities. The same fixed cap does not force selected checkpoints to have identical effective exposure; report selected round and its counters separately.

### 3.2 Nodes with insufficient coverage

Keep all 979 original training nodes and the full original embedding mapping. All currently exceed 1,024 six-hour no-fill training windows. On a unique ten-minute origin grid there are at most 144 origins/day; the minimum 1,598 rows implies at least twelve training days, satisfying the historical seven-day threshold without inventing exact day counts. Confirm actual days during materialization.

Do not remove training nodes because they lack validation/test windows. V0 has 929 evaluated nodes and T0 has 570; report the remaining 50/409 as unevaluated, not zero-error nodes. Permit any positive evaluation count and publish it alongside each node metric; sparse nodes are descriptive and uncertain. If reconstruction breaks the training thresholds or audited counts, stop for review rather than silently filtering, renumbering, or lowering thresholds.

## 4. Preprocessing, targets and leakage safeguards

- Preserve UTC grid 2022-01-01 to 2022-03-01 exclusive. Train ends 2022-02-11 07:10; validation ends 2022-02-20 03:30. Full target intervals must lie within their assigned split. Histories may precede split starts, without accessing observations at/after their origin.
- Reuse original stable duplicate handling, range rules, causal 300-second sample holds, 50% bin coverage and weather as-of age. A finite target bin has the original coverage-controlled definition; this is not a claim of uninterrupted per-second sensor observations.
- Fill raw historical feature bins only, from the same node/channel's last original finite bin. Fill at most the first 1/2/3 missing bins; imputed values do not renew age. Reset state at both cuts, never fill leading missing values, never interpolate backward. Histories may include prior-split bins that were filled causally within that earlier split.
- Keep a separate immutable observed-power vector. Require six original finite future power bins and compute their mean using unchanged v4 cumulative sums and float32 conversion before normalization. Filling input channel 0 must never alter a target or its validity.
- Ignore weather 7/8 for history eligibility under every policy, including G0. Never forward-fill weather; mask standardized batch copies to zero. Indoor inlet temperature remains a feature. Do not add imputation indicators to the fourteen model channels in this primary study; retain masks/ages separately for provenance.
- Reuse exactly the original frozen input/target scalers and persistence coefficients for every policy, fitted on the original pooled training-window population. This uses training-period information only, but is not a new fit on the six-hour or matched cohort. No learned transformation may use validation/test observations. A different scaler strategy would require a separately approved protocol and new artifacts.
- Reproduce strict one-hour reference first using `audit_node(..., histories=(6,))` and `ReferenceData.verify_node(index, original, edges, selected[6], targets)`. Require byte-exact original identities/order, targets and inputs; do not change expected hashes to pass.
- Restrict workers to training-only capabilities; keep V0 coordinator-only and test data unavailable to fit/selection code. Freeze cohort IDs, transforms, source snapshots, sample plans and configuration before fitting. Save imputed input masks/ages independently from observed labels.
- Lock all selected model/configuration/checkpoint hashes from validation receipts before final test inference. Include all seeds and policies; no best test seed, threshold, node subset or round may be selected. Test reporting is a separate, explicitly authorized command.

The test period and its coverage have already been observed repeatedly. This is an exploratory development benchmark with retrospectively coverage-matched cohorts, not untouched confirmatory evidence. Prospective confirmation requires a newly reserved time period or dataset. Predeclaring this plan cannot erase past exposure.

## 5. Persistence, metrics and statistical reporting

On V0/T0, every method/policy uses exactly the same original target vector and input histories. Persistence predicts the last completed ten-minute power-bin mean for the next-hour mean target (`X[:, -1, 0]` inverted to W), not the previous hour's average. It is finite and observed on this no-fill cohort; one shared persistence vector must have the same hash for all fits.

Report in W: pooled `MAE=mean(|prediction−target|)` and `RMSE=sqrt(mean((prediction−target)^2))`, after inverse target scaling and ×1,000. `src/experiment_eval.py::metric_pair/evaluate` define these conventions. Checkpoint selection uses pooled V0 RMSE; MAE is a secondary reported metric. Initial residual persistence can differ by tiny float32 rounding from the separate float64 physical baseline; keep both conventions explicit.

For every fit publish pooled and per-node MAE/RMSE, persistence comparisons, counts, selected round, resource/accounting summaries, finite/alignment checks and window/target hashes. Give equal-node mean MAE AND mean node RMSE separately. Pooled MAE weights by node-window counts; pooled RMSE is the square root of the count-weighted mean node squared RMSE, not the mean node RMSE.

Across seeds report every score, mean and sample SD (`ddof=1`), and paired G2−G0 differences within each algorithm; G1/G3 contrasts are sensitivities. Average scores, not prediction vectors. Three seeds share one dataset and are not three independent datasets; their SD is not a confidence interval. Do not treat overlapping windows or 570 nodes as independent replicates for a naive significance test. Time-block bootstrap would require a predeclared block scheme and enough audited distinct dates; the current aggregate artifacts do not establish that condition. The minimal protocol therefore makes descriptive, not significance, claims.

Never compare scores from different native cohorts as if matched. Historical v2/v3 75,566-window scores remain context, not numerical controls for T0.

### 5.1 Secondary deployment coverage

For G2, `K(2,test) \ T0` contains 5,231 recovered windows; per-node count differences show recovered windows on 802 nodes, including 232 nodes with no T0 windows. This is separate from the 6,590-window primary comparison. If later authorized, evaluate the already validation-selected G2 model and persistence on exactly that recovered cohort, without retuning. G0 abstains there; it has no valid finite input under its declared policy. Do not fabricate a G0 error or compare recovered-cohort RMSE with T0 RMSE as an imputation improvement.

On recovered windows, the last input power bin may itself be filled. Report its observation flag/age and define persistence as the same causal carried-forward power used by the declared policy; do not claim a missing last bin was observed. Where multiple fill policies are compared, restrict prediction comparisons to their explicit shared IDs and report coverage separately. Existing same-window input equality still applies; differences across trained models would arise from training-pool access.

## 6. Verified 24-hour limitation

All three fill limits have zero 24-hour test windows. The saved CSV yields the same reconciled test exclusions for G1/G2/G3:

| Quantity, 979-node test grid | Windows/origins |
| --- | ---: |
| Candidate node/origin pairs | 1,248,225 |
| Insufficient initial history | 0 |
| Target crosses test end | 4,895 |
| In-split target contains unobserved power | 655 |
| Remaining observed-target candidates with invalid input history | **1,242,675** |
| Accepted | **0** |

Thus the global history start and target-boundary rule do not explain the zero cohort. Observed future targets exist for 1,242,675 candidates; every one has an invalid required 144-bin history. Under G3, aggregate residual missing test bins include 1,086,870 CPU-user and 1,091,602 memory-used-fraction bins (raw missing counts minus filled-input counts). These feature counts overlap and must not be added as window exclusions.

Given the causal age-limit algorithm, remaining invalid bins are leading missing values since a split reset or missing runs extending beyond three bins. **The saved artifacts do not retain run-length/last-observation masks, so the exact contribution of long gaps versus split-leading gaps, or the blocking channel at every origin, is unresolved.** Do not assert that all failures are long gaps or all are caused by resetting state. No raw scan was rerun to obtain a finer answer.

Exclude 24 hours from model fitting in this protocol. G3 has some training/validation availability (906,775/19,909 windows) but no test cohort; fitting cannot establish held-out 24-hour forecasting here. Do not impute targets, relax completeness, or import earlier values across reset boundaries to manufacture a test set. During a separately approved future preparation pass, retain per-channel run lengths, leading-gap flags and observed-target validity to decompose the cause. An independently collected more continuous period is the defensible route to a 24-hour study.

## 7. Current implementation and minimum additive changes

Exact inspected entry points/interfaces:

| Existing path / interface | Behavior and consequence |
| --- | --- |
| `experiments.py::main/validate_config` | Existing `freeze`, `fit`, `report`, `matrix --stage fit|report`; v1 schema is embedded here, not a separate configuration module. Common fields: version/kind/model/weather/seed/device/num_threads/learning_rate/patience; neural fields: epochs/batch_size/eval_batch_size/hidden_size/num_layers/node_embedding_size; pilot adds train_limit. `fit_neural()` uses Adam, not the required SGD. |
| `src/multinode.py::weighted_bins/node_windows/prepare_multinode` and `src/preprocess.py::_range_mask/_weather_asof` | Original causal recipe, generic history construction and training-only scaler fitting. Preparation writes outputs, refilters nodes per split, renumbers them and expands dense windows; do not invoke unchanged for this comparison. |
| `src/history_data_v4.py::build_binned_grid/node_features/ReferenceData/mask_weather` and `src/history_coverage_v4.py::audit_node/preflight` | Reusable bounded raw/binning, strict original replay and variable-history masking helpers. They do not provide a new trainable six-hour contract. |
| `src/v5_missing_data/eligibility.py`, `diagnose.py::masked_coverage`, `forward_fill_audit.py::causal_forward_fill/eligible_origins/replay_reference_node` | Audited no-fill/filled eligibility, split resets, original-power target protection and exact reference replay. |
| `src/experiment_data.py::freeze/FrozenData.verify/weather_view` | Original dataset hash pinned; six-step shapes. New cohorts cannot be disguised as v1 arrays. |
| `src/federated_data.py::_FrozenContext/split_arrays/ClientTrainDataset/build_client_index/load_coordinator_validation` | Six-step shapes and original contract; strict training/client capabilities. Validation expects all mapped nodes; V0 has only 929 evaluated nodes. New adapters must preserve 979 IDs while permitting zero evaluation coverage. |
| `federated.py::main`, `src/federated_rounds.py::validate_config/run_rounds/train_client` | v2 config/output path restrictions, fixed main contract/979 clients/4,506,133 rows; fixed full-local-pass semantics. Cannot run this proposal by editing just history length. |
| `src/centralized_sgd.py::validate_config/verify_reference/plan_block/run_from_paths/main` | v3 schema extends v2 with `reference_run`/`reference_manifest_sha256`; requires original counts and historical v2 schedule. New row pools cannot replay old membership/schedules. |
| `src/model.py::SequenceRegressor` and `src/federated_train.py::GlobalModelState/aggregate_fedavg` | GRU already accepts variable sequence length; generic state/aggregation helpers can be reused with correct new subset-count manifests. Protected model source need not change. |
| `src/experiment_eval.py::metric_pair/evaluate`, `src/forecast_plot.py::gap_series/plot_forecast` | W metrics, node/origin alignment, last-bin persistence, UTC plots with gap breaks. Preserve mathematical contract; new data receipts/adapters are necessary. |
| `src/federated_evaluate.py::verify_header/identity_record`, `src/centralized_evaluate.py::TestCohort` | Locked old six-step/cohort selection contracts; cannot evaluate new 36-step cohorts unchanged. New test-only adapter must accept original mapping with 570 evaluated nodes. |

V2 configuration schema is exactly `src/federated_rounds.py::DEFAULTS`, including all fixed settings in Section 3 plus `version`, `kind`, `device`, `num_threads`, `minimum_windows`, `minimum_days` and `frozen_contract_sha256`; unknown fields are rejected. V3 adds the reference bindings and fixes version 3. A new reviewed schema must describe policy/native-pool hashes, 36 steps, observed targets, fixed V0/T0 hashes, reference-size budgets, sample-plan hash, source/environment bindings and output locations. Do not alter v1/v2/v3 schemas or hashes.

Minimum future additions, **not implemented by this task**, with names checked unused:

- `src/v5_missing_data/build_training_cohorts.py`: one approved reconstruction/materialization pass, original replay, exact IDs/targets, observed/imputed masks/ages, all policy pools, common cohorts, client/day audit and compact immutable contract. Prefer binned arrays plus origin indices and lazy batches over dense overlapping X.
- `src/v5_missing_data/run_policy_experiments.py`: new capability adapters, round subset planning, paired FedAvg/centralized training, validation-only selection, atomic checkpoints, exact accounting and tested deterministic resume. Reuse pure model/aggregation/metric helpers; do not bypass old runner checks or monkeypatch their fixed types.
- `src/v5_missing_data/evaluate_policy_models.py`: a separately locked test-only evaluator and common/recovered panel receipts. Test access requires separate approval after fitting.
- New focused tests/configuration under a separately approved namespace, including `configs/experiments/v5_missing_data/gru_gap_policy_v1.json`; all outputs below new `output/experiments/v5_missing_data/model_comparison/v1/` children. Avoid filenames matching the protected root `src/experiment_*.py` glob.

The completed audits save counts, not full per-origin membership, full missingness chronology or retained binned grids. Therefore an actual future training-data build necessarily needs one explicitly authorized source reconstruction. Do not rerun the coverage audit just to recreate its summaries; combine required preparation with ID/target locking and the unresolved gap-cause diagnostics, and retain reusable compact artifacts. No such reconstruction occurred in preparing this document.

Tests needed before that implementation can be accepted: exact original replay; no-fill/filled input equality on shared windows; prefix-age/reset behavior; target immutability; intersection and mapping hashes; missing evaluation nodes; fixed reference-size sampling/no replacement; identical paired batch plans/budgets; aggregation using processed counts; training-only preprocessing/data access; validation tie rules; interruption/resume; finite/reload checks on synthetic/validation data; and historical artifact preservation.

## 8. Computational plan and execution gates

Dense float32 training X alone would be approximately 4.34/5.63/6.49/7.26 GiB for G0/G1/G2/G3. Including every policy's native validation/test X raises the four-policy total to approximately 24.64 GiB, excluding targets, IDs, temporaries and checkpoints. Lazy bin/origin storage is preferred. One `[512,36,14]` float32 input batch is 0.984 MiB; this is not total GPU memory. Recurrent activations and workspace increase with sequence length despite unchanged parameter counts.

Recorded forward-fill audit resources are 271.869 seconds excluding initial preflight, and 2,567,647,232 bytes peak lifetime RSS (about 2.39 GiB). These are preparation-diagnostic measurements, not training benchmarks. Historical v3 one-hour completed wall times vary from approximately 961 seconds to 19,694 seconds across seeds; contention/utilization is unresolved. Do not extrapolate one as a guaranteed six-hour runtime. The proposed exposure×sequence-length proxy is about 3.08 times the old per-seed workload, but it is not a wall-time estimate or an energy measure. A new approved smoke pilot must measure GPU peak allocated/reserved memory, process RSS, examples/sec, steps/sec and validation time before committing to 24 fits.

Gates: (1) review this estimand and sampled-budget departure; (2) authorize additive preparation implementation/tests only; (3) approve one materialization run and verify all IDs/counts/targets/client budgets; (4) approve a validation-only smoke and inspect memory/resume/accounting; (5) review estimated total cost before the 12 primary fits; (6) fit/report sensitivities as predeclared; (7) lock all validation-selected exports and separately authorize final test reporting. No automated transition between gates.

## 9. Commands and their status

All commands run manually from the project root using `.venv`. Check every destination immediately before its future run. Do not overwrite old cohorts, checkpoints or partial results. Current audits remain under their original paths.

**Available now, read-only verification only; already passed during this task:**

```bash
cd /media/csedept/cse2014/Project/m100_training_template
.venv/bin/python -B -m src.v5_missing_data.forward_fill_audit \
  --config configs/experiments/v4_history/coverage_audit_v1.json \
  --reference-data-dir output/experiments/v1/frozen \
  --raw-data-dir data/extracted/b067a6ea8a820f12dbab8ee65f0d3fb862e3f6ea0a8646614d09247f7c71b357 \
  --baseline-diagnostic-dir output/experiments/v5_missing_data/diagnostic/v1 \
  --output-dir output/experiments/v5_missing_data/forward_fill_audit/v1 \
  --verify-only
```

**The following are exact proposed CLI contracts for future additive code. The modules/config do not exist. DO NOT RUN THEM NOW.** Their actual interfaces must be implemented, tested and reviewed before launch commands are confirmed. These are not supported flags for any historical runner.

Proposed one-pass preparation, only after explicit authorization:

```bash
.venv/bin/python -B -m src.v5_missing_data.build_training_cohorts \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --audit-dir output/experiments/v5_missing_data/forward_fill_audit/v1 \
  --reference-data-dir output/experiments/v1/frozen \
  --raw-data-dir data/extracted/b067a6ea8a820f12dbab8ee65f0d3fb862e3f6ea0a8646614d09247f7c71b357 \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts
```

Proposed validation-only smoke, after verified preparation:

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage smoke --method centralized_sgd --policy ff20 --seed 42 \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_v1
```

Proposed independently authorized primary and sensitivity fit commands, never test reporting:

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage fit --matrix primary \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/primary

.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage fit --matrix sensitivity \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/sensitivity
```

Do not supply an executable final-test command before its adapter, validation-selected hashes and immutable selection lock exist. No resume command is promised for these nonexistent runners; deterministic resume must first be demonstrated. Retain partial outputs, inspect errors and use a separately checked new version unless tested resume explicitly supports the interruption point.

## 10. Decision criteria and unresolved issues

Support for further forecasting research would require: exact data/target alignment; identical planned budgets and initial states; finite validation/export/reload behavior; a transparently reported paired G2−G0 primary contrast on T0; MAE/RMSE trade-offs and per-node heterogeneity; and honest coverage/age/resource costs. A small, inconsistent or adverse difference is not evidence of improvement. Coverage recovery alone supports data-availability work, not better prediction, energy savings, cooling/PUE benefits, SLA outcomes or carbon-emissions prediction.

Still unresolved: actual new ID inventories/hashes and origin/date coverage; exact blocking-channel/run-length versus split-leading-gap attribution; performance under imputation; GPU cost; and approval of the proposed estimand, training-pool weighting and per-session sample budget. A new prospective holdout and deployment/privacy guarantees are not established. No node is an independent data center and no seed is an independent dataset.

Stop before fitting if materialized counts differ from the audited/nesting-derived counts, targets are imputed or mismatched, mapping/cuts/scalers drift, any worker sees validation/test records, budgets cannot be paired, evaluation cohorts differ, artifact hashes fail, or resource limits are exceeded. Do not fix failures by weakening checks or overwriting history.

## Appendix: direct evidence references

Paths below are repository-relative. Implementation descriptions above refer to the named functions; configuration values were read from actual files, not inferred from tab names. No external literature citation or unrun experimental outcome is invented.

- `output/experiments/v5_missing_data/forward_fill_audit/v1/{manifest.json,completed.json,reference_replay.json,summary.json,coverage_by_node.csv,preservation.json,resources.json,source_quality.json,preparation_config.json,forward_fill_audit.py,eligibility.py}`.
- `output/experiments/v5_missing_data/diagnostic/v1/{summary.json,masked_coverage_by_node.csv}`.
- `output/experiments/v1/frozen/{contract.json,preprocessing.json,node_names.npy}`; frozen-array content hashes were verified through the existing preflight, without inference.
- `src/v5_missing_data/{forward_fill_audit.py,eligibility.py,diagnose.py}`; `src/{history_coverage_v4.py,history_data_v4.py,multinode.py,preprocess.py,experiment_data.py,experiment_models.py,experiment_eval.py,forecast_plot.py,model.py,federated_data.py,federated_train.py,federated_rounds.py,centralized_sgd.py,federated_evaluate.py,centralized_evaluate.py}`; root `experiments.py` and `federated.py`.
- `configs/experiments/v2_fedavg/{gru_masked_seed42.json,README.md}` and `configs/experiments/v3_centralized_sgd/{gru_masked_seed42_schedule_matched.json,README.md}`; other seeds' completed counters are in the paths below.
- `output/experiments/v3_centralized_sgd/main/gru_masked_seed{42,43,44}_schedule_matched/completed.json`; `output/experiments/v3_centralized_sgd/reports/validation_comparison_v1/REPORT.md` for historical protocol/metric interpretation, not new-cohort forecast baselines.

Key immutable bindings:

| Evidence | SHA-256 |
| --- | --- |
| Original frozen contract | `24b2224ea9a7e1f255faf6357ffe7a3536fcbcfd90ab8b73179798349736b722` |
| Forward-fill preparation config | `778fa2aedb4d43711ff60f2b0b7d523d0f189bd8b47496ddfc5158c03367b8d9` |
| Forward-fill source snapshot | `6177203259a29844951e2cf428c15dc132508839a8acdd2a23368b13c1a833b3` |
| Forward-fill summary | `54741d23eb42314a62047be0a16e1dba215799adcd3db8b49544713f23c3af6e` |
| Forward-fill per-node CSV | `f45a4115eeb057e7894f8dbf7ea30acdd24b6ca329e5d4e07cacd05208fe0ac0` |
| Prior no-fill diagnostic summary | `3e2ace973fec31d4d5e41b440620e44a6a38654aaebbdaa3db69f534efe20c40` |
| Prior no-fill per-node CSV | `82e71554c4a8483edbe729227e1921e0591d03201773db38079d2847a57c86e1` |

Documentation preservation check: before/after inventory of all 3,833 pre-existing non-venv/non-Git files, including existing caches; zero content/size/mtime/mode mismatches required before this addition is accepted. Access times are not compared. Only this new Markdown document is added. The copied project has no usable Git metadata at this location; historical receipts and file inventories provide the preservation evidence instead of a Git clean-status claim.
