# Research methodology and data design

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Distinct experiment tracks

| Track | History / outputs | Cohort/method |
| --- | --- | --- |
| v1 | 6 steps / scalar power | 4,506,133 train, 199,670 validation, 75,566 test; LSTM/GRU/XGBoost; neural Adam |
| v2/v3 | 6 steps / scalar power | Same frozen windows; FedAvg vs schedule-replay centralized SGD |
| v5 | 36 steps / scalar power | Native training pools; common validation 31,448 / 929 nodes; test 6,590 / 570 nodes |
| Phase 2 | Primary 36; proposed 6/36/144 / power and temperature | [PARTIALLY IMPLEMENTED] design only; March cohorts/scalers/models [BLOCKED] |

Evidence: [frozen contract](../output/experiments/v1/frozen/contract.json), [v5 contract](../output/experiments/v5_missing_data/model_comparison/v1/cohorts/contract.json), [Phase 2](experiments/phase2_multioutput/v1/PROTOCOL.md). Historical one-hour versus six-hour score differences do not isolate history effects: cohorts and budgets differ.

## Exact feature order

`src/experiment_data.py::FEATURES` defines the all-node fourteen-slot order. The optional-GPU single-node `preprocess.FEATURE_NAMES` template is different and must not replace it.

| Index | Feature | Unit / transformation |
| --- | --- | --- |
| 0 | power_kw | Binned `ipmi_pub/total_power`, W / 1,000 |
| 1 | cpu_user | Ganglia % |
| 2 | cpu_system | Ganglia % |
| 3 | cpu_idle | Ganglia % |
| 4 | cpu_wio | Ganglia % |
| 5 | memory_used_fraction | `1 - binned(mem_free)/binned(mem_total)` |
| 6 | inlet_temperature_mean_c | Binned `ipmi_pub/ambient`, degC |
| 7 | outdoor_temperature_c | Weather `temp`, degC |
| 8 | outdoor_humidity_percent | Weather `humidity`, % |
| 9 | hour_sin | UTC bin-right-edge daily sine |
| 10 | hour_cos | Daily cosine |
| 11 | week_sin | Weekly sine, Unix-time offset +3 days |
| 12 | week_cos | Weekly cosine |
| 13 | node_index | Categorical frozen mapping position |

[Temperature semantics](../output/phase1_4/temperature_semantics.md) records author documentation describing ambient as node-inlet temperature. Its mean is temporal, not a verified spatial probe average. Exact placement, calibration and drift remain [UNKNOWN]. Indoor channel 6 is distinct from outdoor 7; neither is cooling energy.

## Preparation and targets

`multinode.read_node_groups` scans selected Parquet partitions, orders by node/time and keeps the last duplicate. `preprocess._range_mask` rejects nonfinite/out-of-range percentages, negative power/free memory and nonpositive total memory. Historical temperature checks add no speculative cutoffs. `weighted_bins` uses causal time-weighted holds capped at 300 seconds; ten-minute bins need at least 50% coverage. `_weather_asof` takes the latest observation strictly before bin right edge with age <=30 minutes. Validity does not prove continuous observation.

For origin t, history is `[t-H*10min,t)` and target is the arithmetic mean of six original valid future power-bin estimates over `[t,t+60 min)`, not instantaneous power at t+60min. Historical targets are standardized kW; physical reporting inverts scaling then multiplies by 1,000. Identity is `(original node string, forecast-origin UTC datetime64[ns])`, ordered by original node index then origin. Policy row offsets are not interchangeable identities.

`multinode.node_windows` uses cumulative validity masks and independent node arrays; no gaps/nodes are concatenated. `prepare_multinode` derives history/horizon from configured history hours, resample minutes and horizon minutes. v1 `weather_view` requires `[N,6,14]`; v5 `training_data.HISTORY/HORIZON` fixes 36/6. A JSON history change alone is insufficient.

## Chronological rules

| Historical split | Inclusive lower UTC | Exclusive upper UTC |
| --- | --- | --- |
| Train | 2022-01-01 00:00 | 2022-02-11 07:10 |
| Validation | 2022-02-11 07:10 | 2022-02-20 03:30 |
| Test | 2022-02-20 03:30 | 2022-03-01 00:00 |

Cuts are integer-truncated calendar-bin fractions .70/.85, not accepted-window percentages. Targets lie wholly within a split. Historical validation/test histories may use prior-split observations before the origin. v5 fill state resets at cuts; prior-split history may include values filled causally within that earlier split. Targets never use filled values.

Phase 2 deliberately requires **whole history, target and raw contributors within the assigned split**, hourly origins and non-overlapping targets per node. It keeps original train/validation endpoints, excludes old February test from fitting, and intends March 1–April 1 for retrospective test. The future bridge must prove these differences, not relabel historical bins compatible.

## Scaling, weather and missingness

Historical v1/v2/v3 and v5 reuse pooled original training-window means/stds; categorical channel 13 stays mean 0 / std 1. Near-zero input stds become one. v5 scalers were fitted on the original **one-hour** training population, not its six-hour native pools. `SplitData.batch` preserves raw-float 32 → original-float64 normalization → model-float 32 ordering. No holdout fit occurs.

Weather masking zeros standardized channels 7/8 on copies, retaining indoor temperature. v5 ignores outdoor weather for history eligibility, never fills it and masks it identically. Earlier v1 froze weather-complete windows before masking and did not recover missing-weather windows. Those cohort designs differ.

| v5 policy | Native training windows | Maximum input fill age |
| --- | ---: | --- |
| no_fill | 2,310,256 | 0 |
| ff10 | 2,996,160 | 1 bin |
| ff20 | 3,459,118 | 2 bins |
| ff30 | 3,867,660 | 3 bins |

Fill uses the last original finite bin in the same node/channel/split; filled values never renew age and leading gaps stay invalid. Separate observed-power vectors protect future labels. Common no-fill validation/test are identical across policies. The estimand is access to enlarged training pools under fixed compute, not pure imputation accuracy. Matching training too would make complete-window policy inputs identical.

Phase 2 instead declares no policy bin forward-fill and full inputs with matched weather enabled/masked. Its fresh eligible-training scaler is a design rule, not fitted evidence; silently importing old scalers would change that protocol.

## Completed GRU and FL design

`SequenceRegressor` replaces node slot 13 with a learned 979×8 embedding:13 numeric +8 embedding =21 recurrent values. v5 uses two 64-unit GRU layers, last-step output and Linear(64, 1). Fixed `baseline_coefficients` add standardized last-bin persistence. Zero head weights/bias initialize a zero correction; float 32 neural persistence may differ slightly from separate float64 physical persistence.

Fresh seeded models use SGD lr 0.01, zero momentum/decay, standardized-target MSE, batch 512 and retained remainders. Each of 200 rounds/blocks samples 98/979 clients uniformly without replacement. Node k processes `m_k`, its no-fill count (1,598–3,396), drawing distinct native rows then independently shuffling. RNG domains 0/1/2/3 separate initialization, clients, shuffle and subset; no-fill uses the same sampling code.

FedAvg clients clone one immutable snapshot and reset local SGD. Weights use processed `m_k`, not native pool counts. `aggregate_fedavg` validates names/shapes/dtypes/finite values/fingerprints/fixed buffers and accumulates trainable tensors in CPU float64 before casting back. Whole embedding tables are averaged, diluting a client's row correction. Centralized SGD processes the same ordered plans on one continuing model without averaging; matching exposure does not equalize optimization.

## Phase 2 multi-output definition — not trained

[PARTIALLY IMPLEMENTED] The frozen design uses inputs `[N,H,14]` (H=36 primary, 6/36/144 history study) and targets `[N,2]`: next-hour mean power W and documented node-inlet temperature degC. Both require six original observed future bins. Planned GRU/LSTM replace the ID with an eight-dimensional embedding, use two 64-unit layers and Linear(64,2), without importing the power-only residual buffer.

Selection is earliest minimum `J = 0.5 * ((RMSE_power / train_std_power)^2 + (RMSE_temperature / train_std_temperature)^2)` on complete validation. Physical MAE/RMSE for each output accompany J; mixing raw W and degC without scaling is not its objective. A fresh training-only scaler per declared suite uses its accepted six-hour reference population; all paired methods/histories/masks in that suite reuse it. Constant targets stop review. No real scaler was fitted.

Training eligibility is at least 24 accepted training windows on at least seven distinct UTC dates, never holdout availability. The original 979-name mapping is candidate identity, not guaranteed March eligibility. Fewer than 98 eligible training clients blocks the declared participation setting. The design contains 36 prospective fits plus four seedless persistence references; its real data/training/locked-test adapters are not implemented. See [frozen evaluation](experiments/phase2_multioutput/v1/EVALUATION_SPEC.md).

## Evaluation, alternatives and future work

Complete validation at 0,10,...,200 selects earliest strict minimum pooled RMSEW. Three-round smoke validates at 0/3. Test is separate after all selections are locked. Persistence is last completed ten-minute power, not preceding-hour mean. MAE=mean(abs(error)); RMSE=sqrt(mean(error²)). Pooled MAE count-weights node MAEs; pooled RMSE square-roots count-weighted node MSE. Equal-node mean RMSE differs. Across-seed means average scores, with sample SD(ddof=1), not ensemble predictions or confidence intervals.

[PLANNED] Controlled H=6/36/144 comparison needs per-split identity intersections, invariant targets/mapping, consistent normalization/budgets and separate native-coverage panels. Historical full-input v5 audits report zero 24-hour test windows under tested policies; March cannot be assumed to repair that. [PLANNED] Environmental ablation distinguishes indoor 6 from outdoor 7/8. [PLANNED] Energy decisions need actions, operational constraints, measured energy boundaries and controlled policy comparison; forecast RMSE is not a savings measurement.
