# Model Architecture and Experiment Flowcharts

Documentation version 2. Diagrams describe inspected existing code; they are not newly executed experiments. The previous report and all experiment artifacts remain unchanged.

## Data and experiment flow

```mermaid
flowchart TD
    trace["M100 node telemetry and outdoor weather<br/>January-February 2022"]
    bins["Causal ten-minute bins<br/>Fixed feature definitions and observed power"]
    trace --> bins
    bins --> short["Historical cohort<br/>6 history steps = 1 hour"]
    bins --> long["v5 missing-data cohort<br/>36 history steps = 6 hours"]
    short --> historical["Frozen v1 arrays<br/>Shared chronological splits and original scalers"]
    long --> policies["No fill / FF10 / FF20 / FF30<br/>Input filling only; split reset; no target filling"]
    policies --> longtrain["Policy-native training pools<br/>979 compute-node client IDs"]
    policies --> longeval["One common no-fill evaluation cohort<br/>Validation: 31448 windows / 929 nodes<br/>Test: 6590 windows / 570 nodes"]
    historical --> oldtrain["Historical LSTM / GRU / XGBoost<br/>Separate matched FedAvg / central-SGD study"]
    longtrain --> newtrain["v5: two algorithms x four policies x three seeds<br/>24 fits; matched processed-count budgets"]
    newtrain --> selected["Validation-selected exports"]
    longeval -. "validation only" .-> selected
    selected --> lockedtest["Immutable selection lock<br/>Previously completed test reporting"]
    longeval -. "test only after lock" .-> lockedtest
    oldtrain --> saved["Existing saved predictions, metrics and receipts"]
    lockedtest --> saved
    saved --> reports["Read-only arithmetic and documentation<br/>Separate compatible cohort panels"]
```

This describes the completed project workflow, not a command to rerun it. A target is always the next six observed ten-minute power bins. The historical and v5 panels have different accepted cohorts and protocols; their scores are not a controlled history-length comparison. Clients are compute nodes, not independent data centers. The v5 test origins occur on one already-observed evening.

Evidence:

- [src/multinode.py](../../src/multinode.py) — `weighted_bins/node_windows/prepare_multinode`

- [src/experiment_data.py](../../src/experiment_data.py) — `freeze`

- [src/v5_missing_data/build_training_cohorts.py](../../src/v5_missing_data/build_training_cohorts.py)

- [src/v5_missing_data/training_data.py](../../src/v5_missing_data/training_data.py) — `CohortContext/SplitData`

## Shared residual-GRU model architecture

```mermaid
flowchart TD
    input["Standardized input: B x T x 14<br/>T = 6 historical; T = 36 in v5<br/>Masked variants set channels 7 and 8 to zero"]
    input --> numeric["Channels 0-12<br/>B x T x 13 numerical values"]
    input --> node["Channel 13: frozen categorical node index<br/>Same node identity throughout each history"]
    node --> embedding["Shared learned embedding: 979 x 8<br/>Lookup at every time step: B x T x 8"]
    numeric --> concat["Concatenate along the feature axis<br/>B x T x 21"]
    embedding --> concat
    concat --> gru["GRU: two layers, hidden size 64<br/>Sequence output: B x T x 64"]
    gru --> last["Final sequence output<br/>B x 64"]
    last --> head["Linear 64 to 1<br/>Zero-initialized correction head"]
    input --> power["Last input time step, channel 0<br/>Standardized node power: B x 1"]
    power --> baseline["Fixed persistence mapping<br/>a times last standardized power plus b"]
    head --> add["Add learned correction and persistence reference<br/>Standardized next-hour mean target: B x 1"]
    baseline --> add
    add --> inverse["Inverse target scaling<br/>1000 times sigma_y times output plus 1000 times mu_y"]
    inverse --> prediction["Predicted next-hour mean node IT power<br/>Watts; one scalar per window"]
```

B denotes batch size and T history steps. Node identity occupies one stored channel; its eight embedding values replace that channel inside the model. a = sigma_x,power / sigma_y and b = (mu_x,power - mu_y) / sigma_y. baseline_coefficients is a fixed buffer. The zero-initialized head makes the initial candidate mathematically persistence, subject to recorded float32 rounding. This is the GRU used in the matched FedAvg/central-SGD studies; the historical LSTM shares the wrapper but uses an LSTM recurrent module. No model was instantiated to draw this diagram.

Evidence:

- [src/model.py](../../src/model.py) — `SequenceRegressor.__init__/forward`

- [src/experiment_models.py](../../src/experiment_models.py) — `neural_model`

- [src/federated_rounds.py](../../src/federated_rounds.py) — `initialize_model`

- [src/v5_missing_data/run_policy_experiments.py](../../src/v5_missing_data/run_policy_experiments.py) — `initialize`

- [configs/experiments/v1/gru_masked_seed42.json](../../configs/experiments/v1/gru_masked_seed42.json)

- [configs/experiments/v5_missing_data/gru_gap_policy_v1.json](../../configs/experiments/v5_missing_data/gru_gap_policy_v1.json)

## Historical LSTM, GRU and XGBoost paths

```mermaid
flowchart TD
    x["Historical input: B x 6 x 14<br/>Weather enabled or masked; fixed frozen IDs/scalers"]
    x --> sequence["13 numerical channels plus shared 8-value node embedding<br/>B x 6 x 21"]
    sequence --> lstm["LSTM: two layers, hidden size 64"]
    sequence --> gru["GRU: two layers, hidden size 64"]
    lstm --> readout["Final time step then Linear 64 to 1<br/>Learn standardized residual correction"]
    gru --> readout
    x --> lags["Flatten six lags of 13 numerical features<br/>78 numerical columns; oldest to newest"]
    x --> category["One frozen categorical node index<br/>Column 78; native categorical handling"]
    lags --> trees["XGBoost residual regressor<br/>79 columns; no neural node embedding"]
    category --> trees
    x --> persistence["Fixed persistence reference<br/>Last input power in target-standardized units"]
    readout --> sum["Residual correction plus persistence"]
    trees --> sum
    persistence --> sum
    sum --> output["Inverse target scaling to W<br/>Next-hour mean node IT-power prediction"]
```

This is the historical one-hour v1 comparison only. Its neural fits use Adam; XGBoost uses its own boosting settings. They are not the matched SGD/FedAvg control. XGBoost columns 0-77 are time-major numerical lags, with step 0 corresponding to 60 minutes before origin and step 5 to 10 minutes; node index 78 is categorical, not a continuous numerical magnitude or learned embedding. Weather masking zeroes temperature/humidity lag columns. Validation can retain the initial persistence candidate instead of a learned correction.

Evidence:

- [src/experiment_models.py](../../src/experiment_models.py) — `lag_mapping/lag_features/xgb_matrix/fit_neural/fit_xgboost`

- [src/model.py](../../src/model.py) — `SequenceRegressor`

- [configs/experiments/v1/lstm_masked_seed42.json](../../configs/experiments/v1/lstm_masked_seed42.json)

- [configs/experiments/v1/gru_masked_seed42.json](../../configs/experiments/v1/gru_masked_seed42.json)

- [configs/experiments/v1/xgboost_masked_seed42.json](../../configs/experiments/v1/xgboost_masked_seed42.json)

## Matched FedAvg and centralized-SGD training

```mermaid
flowchart TD
    init["Fresh paired seeded model initialization<br/>Fixed persistence coefficients; zero correction head"]
    init --> initialval["Evaluate initial persistence candidate<br/>Complete common validation cohort"]
    initialval --> schedule["Deterministic round/block plan<br/>98 of 979 node clients; no replacement<br/>Matched ordered batches within a paired run"]
    schedule --> fedstate["FedAvg: immutable global state at round start"]
    schedule --> centralstate["Centralized SGD: one continuing model state"]
    fedstate --> clients["Each selected client starts from the same state<br/>Fresh local SGD; only its training examples"]
    clients --> updates["Validate all client updates and processed counts<br/>Abort invalid round; never silently drop a client"]
    updates --> aggregate["Sample-count-weighted FedAvg<br/>Entire trainable embedding table included<br/>Keep fixed buffers unchanged"]
    aggregate --> fednext["Next global model"]
    centralstate --> sequential["Process the same ordered client/batch sessions sequentially<br/>Fresh stateless SGD per session<br/>Carry model parameters forward; no averaging"]
    sequential --> centralnext["Next central block model"]
    fednext --> due{"Validation due?<br/>Every 10 rounds/blocks, including 200"}
    centralnext --> due
    due -->|Yes| validation["Complete global validation cohort<br/>Earliest strict minimum pooled RMSE"]
    due -->|No| commit
    validation --> commit["Atomic state/receipt commit and counters"]
    commit --> cap{"Round/block 200 complete?"}
    cap -->|No| schedule
    cap -->|Yes| selected["Export validation-selected checkpoint<br/>Test reporting is a separate locked stage"]
```

The diagram shows algorithm structure within each separately frozen protocol. Historical v2/v3 clients process one full local training pass. V5 clients process m_k sampled native-pool examples, where m_k is their no-fill reference count, and aggregate by that processed count. SGD is lr 0.01, momentum 0 and weight decay 0; batch size is 512 with the partial batch retained. Client training is simulated on one compute process; parallel boxes indicate logical clients, not actual parallel execution or protected physical sites. Validation is scheduled only at the stated candidates; unscheduled rounds still commit. Equal nominal settings do not produce identical optimization trajectories.

Evidence:

- [src/federated_rounds.py](../../src/federated_rounds.py) — `run_rounds/train_client`

- [src/centralized_sgd.py](../../src/centralized_sgd.py) — `plan_block/train_block/run_control`

- [src/federated_train.py](../../src/federated_train.py) — `GlobalModelState/aggregate_fedavg`

- [src/v5_missing_data/run_policy_experiments.py](../../src/v5_missing_data/run_policy_experiments.py) — `run_one/train_local`

- [src/v5_missing_data/sampling.py](../../src/v5_missing_data/sampling.py) — `round_plan/due/select`

## V5 missing-data policies and fixed sampling budgets

```mermaid
flowchart TD
    bins["Original frozen per-node feature bins<br/>Separate immutable observed-power target bins"]
    bins --> target{"Six original future power bins observed<br/>and target interval entirely inside split?"}
    target -->|No| reject["Exclude candidate window<br/>Never fill targets"]
    target -->|Yes| policies["36-bin input-history eligibility<br/>Same features, mapping, UTC grid and scalers"]
    policies --> nofill["No fill<br/>Require complete unmasked input history"]
    policies --> ff10["FF10<br/>Carry original value at most 1 missing bin"]
    policies --> ff20["FF20<br/>Carry original value at most 2 missing bins"]
    policies --> ff30["FF30<br/>Carry original value at most 3 missing bins"]
    nofill --> budget["Freeze m_k = each client's no-fill training count"]
    nofill --> pool["Four distinct native training pools<br/>Preserve observed targets and node/origin IDs"]
    ff10 --> pool
    ff20 --> pool
    ff30 --> pool
    budget --> sample["Per selected client session<br/>Sample m_k distinct native rows then shuffle"]
    pool --> sample
    sample --> transform["Original frozen standardization<br/>Zero outdoor-weather channels 7 and 8"]
    transform --> fit["Paired FedAvg / centralized-SGD fits<br/>Same processed counts, seeds and batch sizes"]
    nofill --> common["Common no-fill validation and test windows<br/>Identical input histories and observed targets<br/>Same IDs and ordering across all policies"]
    common --> val["Validation: 31448 windows / 929 nodes<br/>Checkpoint selection only"]
    common --> test["Test: 6590 windows / 570 nodes<br/>Already-observed period; locked final reporting"]
```

Forward filling uses only the same node/channel’s last original observation, resets at split cuts, never fills leading gaps, and does not renew age from an imputed value. Weather is never filled and is masked after standardization. Inputs may use causal prior-split history under that earlier split’s fill state; targets cannot cross their split boundary. Native training windows are 2310256 / 2996160 / 3459118 / 3867660 for no fill / FF10 / FF20 / FF30. All validation/test batches use the no-fill view even for models trained under filled policies. This tests training-pool access at fixed sample budgets, not accuracy on recovered filled evaluation windows.

Evidence:

- [src/v5_missing_data/training_data.py](../../src/v5_missing_data/training_data.py) — `SplitData.batch/TrainingRegistry`

- [src/v5_missing_data/build_training_cohorts.py](../../src/v5_missing_data/build_training_cohorts.py) — `selections_for_node`

- [src/v5_missing_data/sampling.py](../../src/v5_missing_data/sampling.py) — `round_plan`

- [output/experiments/v5_missing_data/model_comparison/v1/cohorts/contract.json](../../output/experiments/v5_missing_data/model_comparison/v1/cohorts/contract.json)

- [output/experiments/v5_missing_data/model_comparison/v1/cohorts/client_budgets.json](../../output/experiments/v5_missing_data/model_comparison/v1/cohorts/client_budgets.json)

## Saved-prediction metrics and seed aggregation

```mermaid
flowchart TD
    saved["One fit's saved predictions, targets and persistence<br/>W; aligned by frozen node/origin identity"]
    saved --> errors["Residual per window<br/>prediction minus observed target"]
    errors --> pooled["Pooled MAE and RMSE<br/>Equal weight per accepted window"]
    errors --> nodes["Group residuals by evaluated node<br/>Per-node MAE and RMSE; retain sample counts"]
    nodes --> macro["Macro MAE = mean node MAE<br/>Macro RMSE = mean node RMSE<br/>Equal weight per evaluated node"]
    nodes --> consistency["Paired node-level differences and distributions<br/>Retain every evaluated node, including regressions"]
    pooled --> scores["Three individual seed scores per configuration<br/>Seeds 42, 43 and 44; same cohort"]
    macro --> scores
    scores --> summary["Arithmetic mean and sample SD, ddof = 1<br/>Do not average prediction vectors"]
    summary --> report["Report each split and cohort separately<br/>Persistence comparison on matching examples"]
```

Pooled RMSE is the square root of count-weighted mean squared node RMSE, not the mean node RMSE. Missing nodes never receive zero error. Historical and v5 cohorts remain separate. Seeds are stochastic repeats on one dataset; sample SD is not a confidence interval and nodes/windows are not assumed independent datasets. The report uses already-saved predictions and performs no new inference.

Evidence:

- [src/experiment_eval.py](../../src/experiment_eval.py) — `metric_pair/evaluate`

- [src/v5_missing_data/evaluate_policy_models.py](../../src/v5_missing_data/evaluate_policy_models.py) — `predict_and_score/evaluate_locked`

- [analysis/thesis_results/verification.json](../../analysis/thesis_results/verification.json)

- [output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2/results.json](../../output/experiments/v5_missing_data/model_comparison/v1/test_evaluation/v2/results.json)

## Validation selection, test lock and report-only analysis

```mermaid
flowchart TD
    train["Training-only local/central batches"] --> candidates["Model candidates from the predeclared schedule"]
    validation["Frozen common validation cohort<br/>Separate coordinator capability"] --> valrmse["Pooled validation RMSE in W"]
    candidates --> valrmse
    valrmse --> best["Earliest strict minimum<br/>Keep initial persistence candidate eligible"]
    best --> selected["Selected checkpoint plus validation receipt<br/>Seed, policy, config and source provenance"]
    selected --> verify["Verify recorded hashes and completion<br/>All expected identities; reject duplicates/missing runs"]
    verify --> lock["Immutable selection lock before test inference<br/>Model/configuration/source and cohort identities"]
    lock --> original["Previously authorized, completed test evaluation<br/>Only locked exports; no reselection"]
    test["Frozen common test IDs and observed targets<br/>No path to training or checkpoint selection"] --> original
    original --> saved["Saved predictions, node metrics, manifest and receipt"]
    saved --> analysis["Current documentation stage<br/>Read existing artifacts and draw diagrams only"]
```

This flowchart documents an already-completed separation of validation selection and test reporting. It does not describe the historical test period as untouched: prior test exposure remains a limitation. For v5 there are exactly 24 locked identities; v2 manifest discovery handles historical directory naming without changing run identities. There is deliberately no arrow from test metrics back to training, checkpoint selection, seed selection or hyperparameter tuning. Rendering this document does not execute any stage shown above.

Evidence:

- [src/experiment_models.py](../../src/experiment_models.py) — `select_if_better`

- [src/v5_missing_data/sampling.py](../../src/v5_missing_data/sampling.py) — `select`

- [src/v5_missing_data/evaluate_policy_models_v2.py](../../src/v5_missing_data/evaluate_policy_models_v2.py) — `discover_runs/preflight/make_lock/verify_lock`

- [src/v5_missing_data/evaluate_policy_models.py](../../src/v5_missing_data/evaluate_policy_models.py) — `load_lock/evaluate_locked`
