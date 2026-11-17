# Newly incorporated and revalidated work

## New since the original report's inventory snapshot

- **T7 missing-telemetry robustness:**99 saved condition/seed/period rows.25 canonical conditions at 3 seeds plus 4 conditions per extended/March period. Three frozen T4 models reused; zero new training fits. Hash and saved-summary checks pass; narrative/test-count limitations are explicit.
- **T1.1 common-history coverage audit v2:** saved native/common counts for1/6/24-hour histories and four policies, a node exclusion matrix and channel-gap audit. New manifest/source bindings fail; count arithmetic alone is not execution provenance.
- **T1.2 history preflight v1:** population/eligibility/config/resource documents. Counts reconcile but the assertion that979clients satisfy>=1,024 windows is false; all eligibility flags are False. Its own manifest also fails bindings.
- **T1.2 controlled-history implementation:** source plus 3 test methods exist. Two-layer residual/MSE GRU, not the old one-layer direct/MSE 9-fit study or T4's L1 winner. Default output initially absent; an external partial run later appeared. No complete matrix is verified.

The inventory comparison identifies61 additional non-report paths since the earlier baseline, including9new source/test/tool files. Some T6 outputs were already incorporated late in v1; new since its baseline does not mean newly discovered or trained in this update. Exact paths are in `EVIDENCE/UPDATE_V2/new_files_since_previous_snapshot.csv`.

## Revalidated and retained

Legacy shared LSTM; historical18-fit architecture/weather panel; earlier one-hour FedAvg/centralized control; historical24-fit v5 matrix; B1 three-seed two-output March/persistence; B1 central training without verified March scores; older FF30 history panel; T2/T3 temporal studies; T4 residual benchmark; T5 weather study; T6 model-family benchmark. These retain separate targets, populations and budgets.

## Not newly completed

No completed FedProx or participation-rate matrix, validated action-response simulator, or energy/SLA policy experiment was found. The no-fill primary9-fit history matrix remains unverified complete; the new OptionA proposal instead evaluates FF30-trained models on a smaller observed-input subset. B1central March adapter exists, but its evaluation outputs were not verified present. No live ongoing experiment is inferred solely from old progress logs.

**Current-execution snapshot:** The new residual-history benchmark began progressing externally during this update; a one-hour seed42 partial-fit log was read without process/model control. It is ongoing, not a completed nine-fit result. Its source changed during review. See `EVIDENCE/UPDATE_V2/current_execution_observation.json`; no provisional metric is promoted to a final finding.
