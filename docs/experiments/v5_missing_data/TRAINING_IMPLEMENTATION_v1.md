# V5 sampled-pass adapters: implementation and execution gates

27 October 2026. Additive implementation only. No model fit, forward pass, optimizer step, prediction scoring, GPU smoke, raw reconstruction or test evaluation was executed in this task. CPU model initialization and synthetic tensor/checkpoint operations are not fitted-model results.

## Files added

| New path | Purpose |
| --- | --- |
| `src/v5_missing_data/training_data.py` | Hash-verified retained-bin access, policy-specific training pools, common evaluation views, restricted copied client capabilities |
| `src/v5_missing_data/sampling.py` | Separate RNG streams, exact reference-size subsets, independent shuffles, budget replay, validation scheduling and earliest-tie selection |
| `src/v5_missing_data/training_io.py` | Exclusive output creation, source snapshots, atomic round commits, saved sampled rows, receipt chains, tensor-exact selected exports |
| `src/v5_missing_data/run_policy_experiments.py` | Explicit plan/smoke/fit/verify CLI for paired FedAvg and centralized SGD; no automatic execution gate transitions |
| `src/v5_missing_data/evaluate_policy_models.py` | Common-validation adapter, future immutable 24-model selection lock, separately invoked test evaluator and read-only verification |
| `src/v5_missing_data/test_policy_training.py` | Synthetic data, planning, initialization, aggregation, checkpoint and lock tests with forbidden-operation guards |
| `configs/experiments/v5_missing_data/gru_gap_policy_v1.json` | Locked protocol, 34 source hashes, six evidence hashes, cohort receipt and exact settings |
| `docs/experiments/v5_missing_data/TRAINING_IMPLEMENTATION_v1.md` | This implementation record, limitations, estimates, manual commands and recovery guidance |

All eight destinations were checked unused immediately before installation. No pre-existing file was edited, renamed, replaced or deleted. No new experiment-output directory was created.

## Verified inputs and scientific scope

Direct protocol: `docs/experiments/v5_missing_data/NO_FILL_FORWARD_FILL_PROTOCOL_v1.md`, especially Sections 3.1 and 9. Protocol SHA-256: `dc5a65609156a5c218efc242e1e7f116bff5903449c51d0a095a29c15868a09c`.

Prepared evidence: `output/experiments/v5_missing_data/model_comparison/v1/cohorts/{REPORT.md,contract.json,scalers.json,node_mapping.json,client_budgets.json,completed.json}`. Completion-receipt SHA-256: `f5750d8a6905684daf6116838c2e793bfd4edd5731edd96fd1a444bedf561d99`. Original frozen contract: `output/experiments/v1/frozen/contract.json`, SHA-256 `24b2224ea9a7e1f255faf6357ffe7a3536fcbcfd90ab8b73179798349736b722`.

| Policy | Native training pool | Processed per selected client |
| --- | ---: | --- |
| no_fill | 2,310,256 | Its frozen no-fill count m_k |
| ff10 | 2,996,160 | Same m_k, uniform distinct subset |
| ff20 | 3,459,118 | Same m_k, uniform distinct subset |
| ff30 | 3,867,660 | Same m_k, uniform distinct subset |

There are 979 original training clients. Common validation has 31,448 windows / 929 evaluated nodes; common test has 6,590 / 570. Missing evaluation nodes remain in the embedding mapping and are not assigned zero error. These are different cohorts from the historical 199,670/75,566-window validation/test benchmark; its forecasting errors cannot be reused here.

Canonical identity is `(original node string, forecast-origin UTC datetime64[ns])`, ordered by original node index, then origin. Common validation ordered identity SHA-256 is `b01288d877f40a2cfa25d7acf5de5180af7ac80e3c2f69e60fb17e408e1592c1`; common test is `7d44fa703b877579791162594922088bea42674c215de8503e65768ca8a2648a`. All policies reference the same saved arrays; no replacement common arrays are written.

The estimand is access to enlarged native training pools under a fixed sample budget. Training memberships intentionally differ. This does not isolate numerical imputation quality, and fully common training would supply identical inputs under all policies. No predictive improvement is assumed. The test period was previously observed; seeds are repeats on the same dataset, not independent datasets. This is simulated federation, with no secure aggregation, differential privacy or OS isolation.

## Data and preprocessing contract

Each history has 36 completed ten-minute bins; the target remains the saved mean of six originally observed future power bins. The original 979-string mapping, fourteen features, calendar grid and split cuts are retained. Targets must lie entirely within the assigned split; histories may contain prior-split observations. The adapter rejects non-grid origins, wrong node identities and crossing targets, checks all six original future power bins are finite, and verifies saved target normalization. Exact target arithmetic and cross-policy equality were already established by cohort preparation and are protected by its immutable artifact hashes; no targets are recomputed by training.

Historical filling uses retained original bins and original-source ages. Imputed values never renew age. Sources must precede their destination, stay in the same node/channel, and fall on the same side of both split cuts as the filled bin. Only ages 1/2/3 permitted by the selected policy are filled. Leading gaps remain invalid. Weather channels 7/8 are never filled and are zeroed after scaling. Evaluation always uses common no-fill histories, regardless of training policy.

Conversion order matches the existing pipeline: append the categorical node index, cast raw values to float32, subtract/divide original float64 training scalers, cast to float32, then mask standardized weather. Scalers remain those fitted on the original one-hour training-window population; they are not fitted to v5 or evaluation examples. `src/model.py::SequenceRegressor`, `src/federated_train.py::GlobalModelState/aggregate_fedavg` and `src/experiment_eval.py::metric_pair` are reused unchanged.

Workers receive only immutable copied `LocalData` training inputs, targets, identities and selected native row positions. They receive no dataset paths or validation/test capability. The coordinator owns validation separately. Context verification hashes all frozen artifacts, including test files, without interpreting test examples or scoring them. This API is a leakage guard, not a security boundary against arbitrary Python filesystem access.

## Sampling, models and selection

Seeds are 42, 43 and 44; algorithms are FedAvg and continuing centralized SGD. Both use fresh CPU-seeded GRU initialization, hidden size 64, two layers, 979-by-8 embedding and the fixed residual persistence coefficients. There is no historical checkpoint warm start. Training is SGD lr 0.01, momentum 0, weight decay 0, MSE on standardized targets; batches are 512, partial batches retained. Validation batches are 1,024.

Each round/block samples 98 clients uniformly without replacement. Stream domains 0/1/2 preserve initialization/client selection/local shuffle conventions. New domain 3 draws a uniform m_k subset from each native pool. The subset is sorted before the independent domain-2 permutation. The same code, including the subset draw, is used for no-fill; method/policy labels do not enter the RNG. Candidate pools that are identical produce identical row plans. The exact RNG algorithm and NumPy version are bound by configuration/environment receipts.

FedAvg clients start from one immutable round snapshot, reset local SGD and aggregate by processed m_k. Native pool counts are recorded but never used as aggregation weights. Centralized SGD processes the same ordered client/batch plans on one continuing model. Its stateless optimizer is recreated per client session; with zero momentum/decay there is no optimizer history to preserve. Neither matching nominal learning rates nor matching exposures makes the optimization trajectories equivalent. Standard FedAvg applies to the full embedding table, including dilution of client-specific row corrections; no special embedding algorithm is introduced.

| Seed | Examples per main method/policy | SGD steps per main method/policy |
| --- | ---: | ---: |
| 42 | 46,268,600 | 100,038 |
| 43 | 46,238,350 | 100,014 |
| 44 | 46,275,837 | 100,074 |

Preflight independently replays the 200-round client draws and integer accounting and compares these totals with the prepared budget manifest. m_k ranges 1,598–3,396. Actual row identities, their order hashes, subset/shuffle seeds, native counts, m_k, batch sizes, steps and cumulative examples are saved per round, alongside validation metrics and timings.

Main runs complete 200 rounds/blocks, with full common validation at 0,10,...,200. The earliest strict minimum pooled validation RMSE in W selects the exported model; MAE does not select it. The initial neural residual-persistence candidate can differ slightly from the separately reconstructed float64 persistence baseline. The engineering smoke is the first three rounds of the same 98-client schedule and validates at 0 and 3. It is not a short hyperparameter-selection experiment.

Selected exports include per-node validation errors. Future test outputs use the unchanged W metric formulas, pooled and equal-node summaries, persistence, all three seed scores, mean/sample SD, and paired ff20-minus-no_fill primary contrasts; ff10/ff30 contrasts are sensitivities. Scores, not prediction vectors, are averaged. Recovered-window secondary panels are not implemented here; their conditional population needs a separate reviewed reporting stage. No command can silently switch the primary evaluator to native test pools.

## Verification actually executed

- Before/after preservation: **3,922 / 3,922 pre-existing files checked; zero mismatches** (SHA-256, byte size, mtime and permission mode; excludes only .venv/.git and access times).
- In-project synthetic command: `.venv/bin/python -B -m unittest discover -s src/v5_missing_data -p 'test_*.py' -v`: **69 passed, zero failures/errors/skips (27 new tests + 42 existing v5 tests)**.
- Real-data CLI `--stage plan`: **preflight_passed**; independently replayed all three recorded budget totals; 24 declared fits; no output creation or model forward/optimizer/scoring.
- Existing `build_training_cohorts --verify-only`: **verified_complete**, 16 cohort arrays verified, original reference replay train 4,506,133 / validation 199,670 / test 75,566; raw decoder false; training/inference/prediction scoring false. Its original-input preflight checks the frozen contract, fourteen arrays, twenty recorded raw Parquet hashes and historical preservation records.
- Configuration binds **34 source files and six evidence files**. All installed source hashes match it.
- Temporary verification records: `/tmp/m100-v5-training-we741ntb/before.json`, `before_install.json`, `after_checks.json`, `project_synthetic_tests.log`, `project_preflight.log`, `existing_cohort_verification.log`. These are external temporary working records, not replacement historical receipts. Initial full-inventory SHA-256: `32af018b3b91273d7b654fe20f48eac5db499540c20b399e2d8aa0dfaa369eb4`.
- New configuration SHA-256: `ff63776db8b4be6ace0dab76572f61d426c32f3635fbb25a2280910b56120301`.

Only synthetic fixtures were transformed, in temporary directories. The implementation tests guard `SequenceRegressor.forward`, `SGD.step` and the evaluator's `metric_pair`; all three call counts were zero. Tests cover common-policy identity/order, node isolation, observed-target rejection, exact split endpoints and prior-split history, audited fill/cast/mask/scaler equality, native m_k sampling, distinct rows, RNG domains, paired CPU initialization, processed-count aggregation, finite parameter rejection, validation schedule/ties, atomic ledger replay, interrupted pending directories, hash tampering, source snapshot preservation, export tensor reload and immutable 24-model test locks before test-array access.

This does **not** verify trained GPU resume equivalence, finite actual model predictions, learning convergence, GPU memory headroom or real evaluation arithmetic end to end. No trained-state comparison was executed. Resume tests exercise untrained tensor checkpoints and synthetic selection metadata, not optimizer updates. The future validation smoke must pass before any main fit can be authorized; an uninterrupted/restarted GPU comparison remains necessary before claiming numerical resume equivalence.

## Manual commands: safe now

Run from the project root with the existing `.venv`. These commands do not start a model fit or inference. Preflight requires the proposed `primary` destination to remain unused and creates no output directory.

```bash
cd /media/csedept/cse2014/Project/m100_training_template

.venv/bin/python -B -m unittest discover \
  -s src/v5_missing_data -p 'test_*.py' -v

.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage plan \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/primary
```

Expected preflight: `preflight_passed`, 24 declared fits, the budgets above and common identity hashes; `model_forward_optimizer_or_scoring` is false. Stop on any integrity/count/configuration error. Do not edit expected hashes to resolve it.

## Proposed GPU smoke: requires explicit new authorization

Do not run this section until the user authorizes GPU fitting. Execute in the foreground so errors and interruption messages remain visible. The paths below were unused at implementation time; recheck at launch. The shell guard refuses occupied destinations. Each initial command deliberately stops after round 1, after which the matching resume command completes the three-round engineering smoke.

```bash
cd /media/csedept/cse2014/Project/m100_training_template

if [ -e output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_v1 ] || [ -L output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_v1 ]; then
  printf '%s\n' 'STOP: centralized smoke destination already exists; preserve it.'
else
  .venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
    --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
    --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
    --stage smoke --method centralized_sgd --policy ff20 --seed 42 \
    --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_v1 \
    --stop-after-round 1
fi
```

Only after the first command reports `stopped` with completed round 1, or an interrupted run has a verifiable complete round:

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage smoke --method centralized_sgd --policy ff20 --seed 42 \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_v1 \
  --resume

.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage verify \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/centralized_ff20_seed42_v1
```

The FedAvg code path needs its own smoke; it is not covered by running the central method:

```bash
if [ -e output/experiments/v5_missing_data/model_comparison/v1/pilots/fedavg_ff20_seed42_v1 ] || [ -L output/experiments/v5_missing_data/model_comparison/v1/pilots/fedavg_ff20_seed42_v1 ]; then
  printf '%s\n' 'STOP: FedAvg smoke destination already exists; preserve it.'
else
  .venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
    --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
    --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
    --stage smoke --method fedavg --policy ff20 --seed 42 \
    --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/fedavg_ff20_seed42_v1 \
    --stop-after-round 1
fi
```

After its verified first-round stop:

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage smoke --method fedavg --policy ff20 --seed 42 \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/fedavg_ff20_seed42_v1 \
  --resume

.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage verify \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/pilots/fedavg_ff20_seed42_v1
```

Success means complete round 3; validation only at 0/3; equal paired examples/steps/sample-plan hashes; finite parameters and predictions; immutable common validation/target/persistence hashes; exact selected validation reload; and `verified_complete`. The two methods may select different validation checkpoints and must not be tuned from the smoke. Expected artifacts in each new directory: `manifest.json`, `configuration.json`, `source/`, `rounds/0000` through `0003` (state, sampled rows and receipts), `selected/{model.pt,receipt.json}`, and `completed.json`. The completion record measures invocation wall time, lifetime process RSS and CUDA peak allocated/reserved bytes. Per-round receipts measure local sessions and validation time. RSS includes imports/loading; invocation wall time excludes earlier preflight; resume invocations are not summed automatically.

## Proposed main fits: separate authorization after smoke and cost review

The following commands are implemented but were not executed or approved for launch by this task. Each root must be unused. They execute sequential fits and stop on failure; there is no automatic transition from primary to sensitivities or test evaluation.

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage fit --matrix primary \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/primary
```

This is 12 fits: two algorithms × no_fill/ff20 × three seeds. After review and separate sensitivity authorization:

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage fit --matrix sensitivity \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/sensitivity
```

This is the remaining 12 fits: ff10/ff30. A single interrupted main fit can resume explicitly, for example only if this is its actual existing manifest:

```bash
.venv/bin/python -B -m src.v5_missing_data.run_policy_experiments \
  --plan configs/experiments/v5_missing_data/gru_gap_policy_v1.json \
  --data-dir output/experiments/v5_missing_data/model_comparison/v1/cohorts \
  --stage fit --method fedavg --policy no_fill --seed 42 \
  --output-dir output/experiments/v5_missing_data/model_comparison/v1/primary/fedavg/no_fill_seed42 \
  --resume
```

Verify individual outputs with `--stage verify` and their exact directory. Do not relaunch a matrix into a partly occupied root. Preserve completed fits; remaining fits need explicit single-fit commands and unused paths. A future test selection lock must first verify all 24 complete 200-round ledgers and derive earliest validation-selected exports. As required by protocol Section 9, no executable final-test inference command is supplied before those models and the immutable selection lock exist. Test reporting then needs separate explicit approval.

## Resource estimates, not v5 training measurements

Dependencies are the existing `.venv` packages: PyTorch, NumPy and the existing preparation/evaluation import dependencies; all installed versions are recorded by `training_io.software()`. No package installation or upgrade was performed. Future GPU commands require `cuda:0` and fail instead of silently using CPU. Sources/configuration, software versions, cohort hash and device are checked on resume. Exact cross-hardware bit identity is not promised.

Retained cohort storage is about 1.54 GiB (85 existing files). Dense 36-step model datasets are not generated. One float32 `[512,36,14]` input batch is 0.984 MiB; a maximum 3,396-row client input copy is about 6.53 MiB, before normalization temporaries, immutable copies, targets, recurrent activations and CUDA workspace. The model has 49,561 trainable values plus a two-value fixed buffer; one float32 state is 198,252 bytes. These are size calculations, not peak-memory measurements.

Saving all sampled row positions costs about 0.345 GiB uncompressed per fit (46.3 million int64 positions); compressed NPZ sizes are unmeasured. Two model states per 201 checkpoints add about 76 MiB per fit, excluding serialization/JSON overhead. Across 24 fits the uncompressed row/state proxy is about 10.1 GiB, plus snapshots, receipts, logs, selected exports and filesystem overhead. Compression may reduce row storage; interrupted pending directories consume additional space and are deliberately preserved.

Existing preparation measured 236.823 s excluding preflight/verification, with 4,243,406,848 bytes lifetime peak RSS; this is not a training estimate. Historical v3 one-hour main runs record 960.634–19,693.509 s, approximately 90.22 million examples and 186,000 optimizer steps. The proposed examples×history-length proxy is about 3.08 times that historical workload. Blind scaling gives approximately 0.82–16.9 hours per fit / 20–405 hours for 24 sequential fits, **an uncalibrated workload envelope, not a scheduling estimate**: historical contention is unresolved, GRU scaling is nonlinear, new lazy CPU reconstruction/checkpoint overhead is unmeasured, and validation sizes differ. Do not commit to the matrix based on this envelope. Measure both authorized smokes, validation time, GPU peaks, RSS, examples/s and steps/s, then revise cost estimates.

Historical one-hour CUDA peak allocated/reserved were roughly 95/118 MiB and sampled process GPU usage roughly 682 MiB; they do not establish six-hour headroom. No measured v5 GPU or RSS requirement is asserted. Allow host headroom beyond the mapped-bin working set and client temporaries, and stop if measured resource limits are exceeded.

## Recovery, limitations and authorization

Each round is committed by directory rename after synchronized checkpoint, sampled rows and hash-chain receipt writes. Interrupted `.pending-*` directories remain untouched and are not treated as commits. Resume verifies exact config/source/environment, all committed checkpoint/trace hashes, deterministic row plans/counts, fixed state and earliest validation selection. It starts from the last complete state and replays the next uncommitted round; committed training is not repeated. Stateless SGD and this GRU's zero dropout require no hidden optimizer/RNG history after initialization, but numerical GPU resume equivalence remains unverified. A completed run's resume path verifies it without re-evaluating it.

If interruption occurs before the first valid manifest/round commit, or integrity verification fails, preserve the directory and share the error. Do not delete pending state, manually update hashes, alter source/configuration, or bypass checks. Recovery may require a separately reviewed unused version. Partial test inference cannot resume: preserve the partial evaluation and obtain review for a new unused destination; never rerun earlier FedAvg/v3 evaluations.

Remaining empirical questions are predictive behavior, realized exposure/row diversity, finite trained predictions, validated GPU resume equivalence, numerical evaluator behavior with real predictions, actual resource cost and selected-checkpoint effective budgets. Common evaluation coverage is retrospective and sparse. Native training distributions differ intentionally; m_k weighting is the approved reference-exposure estimand, not native-population weighting. The original pooled training scalers are reused by design. No 24-hour study, energy saving, carbon prediction or privacy-preserving deployment claim follows from this implementation.

Next required authorization: **explicit approval for validation-only GPU smoke fitting**, followed by inspected results and cost review. The 12 primary fits, 12 sensitivity fits and final locked test reporting each remain gated. Stop on any provenance, target, split, identity, budget, finite-value, reload or resource failure; do not weaken safeguards to proceed.
