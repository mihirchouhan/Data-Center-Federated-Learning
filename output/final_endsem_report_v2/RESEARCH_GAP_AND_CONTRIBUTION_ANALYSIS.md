# Research Gap and Contribution Analysis

## Defensible demonstrated contribution

The contribution is a versioned experimental workflow for known-node forecasting from heterogeneous real telemetry, accompanied by carefully qualified empirical comparisons. Established algorithms—GRU, LSTM, XGBoost and FedAvg—are components, not novelty claims. The system simulates clients for 979 compute nodes in one facility. It does not establish a multi-organization federation.

|Contribution|Actual evidence|Claim boundary|
|---|---|---|
|Causal sensor-to-window contract|Raw reader/bin/window source, observed-target rules and manifests|Historical semantic replay remains UNKNOWN|
|Missing-input comparison|24 v5 fits, 4 policies, 3 seeds, 2 methods|Common test is 6,590 windows on one date|
|Matched method comparison|v5 centralized errors lower in all 12 pairs|Not a general superiority theorem|
|Two-output experiment|B1 power/ambient, three locked models, 194,212 March windows|Mixed persistence outcome; physical ambient position uncertain|
|History-length study|Nine completed FF30 fits at 1/6/24 hours|Longest-history gating; no-fill counterpart not established|
|Temporal and residual studies|T2/T3 records; T4 six final models and 18 metric rows|Nested periods; T4 changes several factors|
|Evidence reconciliation|Explicit score identities, raw hashes and contradictory prose identified|Preservation does not prove every numerical computation|

## Paper-derived gap

The mandatory paper uses adaptive federated MLP decision learning followed by reduced physical optimization. The project uses time-series prediction of observed telemetry. Their principal gap is the missing bridge from a forecast to an action with validated physical and service consequences. This is a research opportunity, not an implemented feature. The seven-table companion compares 25 dimensions and explicitly excludes security mechanisms.

## Prioritized next work

1. Finish the separately authorized, locked B1 centralized March comparison using the exact B1 windows and saved FedAvg predictions. Its fits are completed, while test metrics are not verified present.
2. Reconcile Candidate1/T4 design versus execution: common-cohort counts, longest-history eligibility, history containment, weather masking, scaler population and exact optimization exposure. Preserve old artifacts and publish any correction additively.
3. Use a later period or preregistered retrospective design for temporal robustness. March is already observed; further March-informed choices cannot generate untouched confirmatory evidence.
4. Run one-factor matched comparisons for weather, model family, missingness mechanism, FedProx and participation only after a frozen protocol and explicit authorization. Historical architecture/weather experiments exist, but do not substitute for current same-cohort studies.
5. Audit candidate multi-output measurements for units, entity scope, alignment and coverage. B1 supports power and ambient forecasts, with no demonstrated joint-output accuracy advantage. Facility energy/cooling/PUE require facility-level studies.
6. Specify a limited action space and validated response model before policy optimization. Add energy and service accounting and compare actions under matching workloads. A hypothetical simulator reduction is not an operational saving.

## Unsupported claims

No evidence here establishes energy savings, cooling optimization, PUE reduction, carbon reduction, SLA improvements, general FL superiority, unseen-site generalization, a universal best history, or reproduction of the mandatory paper's full architecture. T4's 24.81 W pooled MAE is meaningful only with its pooled persistence baseline of 33.0661 W, not the 33.3504 W macro baseline. Original historical preflight is FAILED; semantic reproduction is UNKNOWN.

## Intended contribution

A future contribution may connect coverage-aware forecasts to constrained and independently evaluated decisions. Its acceptance criteria should include trustworthy response dynamics, feasible actions, uncertainty/robustness treatment and measured energy/service outcomes. Until those criteria are met, describe the work as forecasting research and a decision-support direction.

**Latest-status qualification:** T5 weather-ablation records are completed; T6 model-family completion appeared during this session and is reported separately in main-report §11.8. Future-work language about these topics refers to further controlled replication, not absence of all results. Two pre-existing T6 progress logs changed concurrently; full-tree zero-mismatch preservation is not claimed. Core inspected source/result hashes remain unchanged.

## Revision v2: additional demonstrated and unresolved work

T7 supplies a completed saved-result study of synthetic random/burst/group input loss. The supported contribution is the sensitivity record under its precise fallback policy, not proof that real failures follow that generator. The new coverage/preflight audits improve population visibility but contradict their own eligibility prose. Their2-layer residual/MSE runner is a distinct design with partial external execution evidence; it must not be merged with the completed 1-layer direct/MSE history panel or T4's L1 winner.

Immediate decisions are to reconcile the eligibility threshold and intended loss before a new history run, and separately authorize the already-trained B1 centralized same-window March evaluation if desired. A validated action-response simulator and measured energy/SLA evaluation remain absent. The paper's optimization contribution remains a direction for future work, not a completed reproduction.

**Provenance qualification in v2:** Nine new history-audit/preflight manifest bindings fail. Their referenced historical input bytes match older records; their own execution provenance remains unresolved. No failed hash or scientific gate was repaired.

**Current-execution snapshot:** The new residual-history benchmark began progressing externally during this update; a one-hour seed42 partial-fit log was read without process/model control. It is ongoing, not a completed nine-fit result. Its source changed during review. See `EVIDENCE/UPDATE_V2/current_execution_observation.json`; no provisional metric is promoted to a final finding.
