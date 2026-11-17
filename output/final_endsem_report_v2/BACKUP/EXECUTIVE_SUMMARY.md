# Executive Summary — M.Tech End-Semester Presentation

**Mihir Chouhan · 25CSE2014 · NIT Goa, Computer Science and Engineering**  
**Supervisor:** Dr. Pravati Swain  
**Presentation:** 19 December 2026 · **Evidence cutoff:** 17 November 2026

**Research title:** Federated Forecasting and Energy-Aware Decision Support for Distributed Data Centers under Heterogeneous and Missing Telemetry.

The completed work forecasts future node telemetry from M100 ExaData. The 979 clients are simulated compute-node partitions from one physical facility. Forecasting and several controlled comparisons are implemented; operational energy-aware control is not yet evaluated.

|Separate track|Verified stored outcome|Main qualification|
|---|---|---|
|Earlier shared LSTM, 1-hour input|MAE 27.934 W, RMSE 54.884 W; persistence 30.927/64.417 W|75,566 windows, 979 nodes; not FedAvg|
|Historical v5, 24 fits|Centralized SGD lower pooled MAE/RMSE in all 12 policy/seed pairs|6,590 common windows, 570 nodes, one test date|
|B1 6-hour two-output FedAvg|Power MAE 53.357 W, RMSE 89.798 W; ambient .448/.725 °C|194,212 March windows, 976 nodes; selected round200 for all seeds|
|B1 persistence|Power 42.392/92.600 W; ambient .303/.610 °C|B1 improves power RMSE only among these four pooled comparisons|
|Newer FF30 history study|Mean pooled MAE 33.302/32.651/32.574 W for 1/6/24 hours|Nine fits; same longest-history-eligible cohort; design/count discrepancies documented|
|T4 residual FedAvg canonical test|Pooled MAE 24.8133 W, RMSE 57.5502 W; persistence 33.0661/64.0451 W|243,138 windows, 978 nodes; retrospective and track-specific|
|T4 centralized canonical control|Pooled MAE 23.9865 W, RMSE 56.9061 W|Full-union central epochs, not exact B1-style exposure replay|

Across-seed means are descriptive summaries of seeds42/43/44 on the same data. The main report supplies sample standard deviations and macro-node metrics. T4's macro MAE is 25.2242 W versus approximately 33.3504 W macro persistence. Never compare its pooled 24.81 W to macro 33.35 W.

The mandatory paper, Liu et al. arXiv:2511.00623v3, is a different task: adaptive federated MLP learning of operating decisions followed by physical optimization. Its R², dispatch-cost deviations and optimization runtime cannot be ranked against this project's watt/°C errors. The official v3 PDF was read; the named uploaded copy was unavailable locally. The companion analysis provides pages, equations, tables and comparison limits.

Strongest supported findings: method rankings are protocol-dependent; historical v5 favors centralized optimization; B1's two-output formulation has mixed persistence performance; longer history has modest conditional effects in the completed FF30 panel; and T4's residual L1 configuration provides the strongest current power-persistence comparison. None proves that federation itself caused the T4 gain.

The next completion opportunity is the already-trained B1 centralized control's separately authorized same-window March evaluation. Reconcile newer protocol/execution differences before stronger claims. FedProx, participation sensitivity, validated action-response simulation and energy/SLA policy evaluation remain future work. Historical strict preflight remains **FAILED**, semantic reproduction **UNKNOWN**. March is previously inspected retrospective evaluation data.

This report-generation task ran no training, inference, GPU work, raw preparation or scaler fitting. No existing artifact was written by this task; two concurrent T6 progress-log changes are disclosed, and the package is separate. See `REPORT_VALIDATION.md` for fresh byte checks, arithmetic checks, omissions and verification limits.

**Latest-status qualification:** T5 weather-ablation records are completed; T6 model-family completion appeared during this session and is reported separately in main-report §11.8. Future-work language about these topics refers to further controlled replication, not absence of all results. Two pre-existing T6 progress logs changed concurrently; full-tree zero-mismatch preservation is not claimed. Core inspected source/result hashes remain unchanged.
