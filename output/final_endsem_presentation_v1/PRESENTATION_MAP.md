# Presentation map

|Slide|Title|Seconds|Sources|
|---|---|---:|---|
|1|Federated Forecasting|30|REPORT|
|2|Forecasting is a prerequisite, not a savings result|40|REPORT, PAPER|
|3|Three questions organize the research|40|REPORT, REGISTRY|
|4|Base paper: learn operating decisions, then optimize|55|PAPER|
|5|Same federated idea; different labels and evidence|55|PAPER, COMPARISON|
|6|The implemented pipeline ends at verified forecasts|45|REPORT, ARCHITECTURE, REGISTRY|
|7|Fourteen slots retain operational and thermal context|50|PREPROCESS, FEATURES, REPORT|
|8|Predict the next-hour mean, not a point at +60 min|55|B1_PREP, B1_EVAL, REPORT|
|9|979 clients means 979 mapped compute nodes|45|B1_MODEL, B1_RUNNER, REPORT|
|10|Two GRU formulations answer different questions|55|B1_MODEL, B1_RUNNER, T4_SOURCE, T4_CONFIG|
|11|FedAvg averages complete local model states|55|B1_RUNNER, FEDAVG, REPORT|
|12|Frozen identities make comparisons interpretable|50|B1_RUNNER, B1_EVAL, REGISTRY|
|13|Completed evidence is a portfolio, not one leaderboard|45|REGISTRY, V5, B1_EVAL, T4, T5, T6, T7|
|14|Earlier results already challenge an FL advantage|55|LEGACY, V5, V5_SEEDS|
|15|B1 two-output forecasting has mixed outcomes|60|B1_EVAL, B1_PERSIST, B1_LOCK|
|16|T4: residual L1 forecasting improves persistence|60|T4, T4_SEEDS, T4_AUDIT, T4_MANIFEST|
|17|T4 gains persist, but later-period errors increase|55|T4_SEEDS, T4_AUDIT|
|18|Longer history shows a small conditional benefit|55|HISTORY, HISTORY_SEEDS, NEW_STATUS, FAILED_BINDINGS|
|19|Weather and model families: gains depend on metric|55|T5, T6, LATE_RESULTS|
|20|Missingness pattern matters as much as its rate|55|T7, T7_SUMMARY, T7_SOURCE|
|21|Broad node gains coexist with a power-sensor failure|50|T7_SUMMARY, T7_NODES, T7_SOURCE|
|22|Energy-aware decisions remain a separate research layer|45|PAPER, REPORT, B1_CENTRAL_STATUS|
|23|Limits and integrity gates remain visible|50|REPORT_VALIDATION, FAILED_BINDINGS, AUDIT, NEW_STATUS|
|24|Conclusions: useful forecasts, bounded claims|35|REPORT, V5, B1_EVAL, T4, T7|

Total:1195seconds, including chart-reading pauses. Technicalbackup andvivaquestions are in the separate script; do not read every backup paragraph during the timed talk.
