# Comparative Analysis with the Base Paper

This chapter compares Liu et al., arXiv:2511.00623v3, with the verified forecasting implementation. Page references identify the official 14-page PDF; the named uploaded copy was unavailable locally, so its byte identity is unverified. Full source identity and mathematical reading are in `BASE_PAPER_DETAILED_ANALYSIS.md`. Project claim identifiers resolve to the evidence register. **Not Implemented**, **Not Evaluated** and **Not Directly Comparable** are substantive boundaries, not missing scores to be estimated.

### Table 1: Overall research comparison

|Dimension|Base paper|Our research|Interpretation/status|
|---|---|---|---|
|1. Problem|Coupled power, heat and workload decisions; §II, pp. 3–4|Next-hour telemetry means from recent history|Different scientific tasks|
|2. Objectives|Near-reference operating decisions with a reduced optimization solve|Forecast accuracy, missingness/history/control comparisons|Forecasting objective demonstrated; operational objective future|
|3. Scope|Learning-to-optimization framework; §III, pp. 4–7|Observed-telemetry forecasting workflow|Complete paper architecture Not Implemented|
|18. Energy-related objective|Explicit economic objective Eq. (1a)|Energy-aware decision support is a research direction|Savings Not Evaluated|
|24. Limitations|Five modeled agents; reproduction details require source-code review|One physical facility, observed periods, changing track contracts|Neither validates universal deployment benefit|

**Interpretation.** The paper learns decision variables that enter a physical coordination problem. Our supervised labels are future sensor-bin means. The shared word “federated” therefore does not make the pipelines reproductions of one another. The project's defensible result is the bounded empirical behavior of explicitly versioned forecasters. The paper supplies a motivation for a later decision layer, not evidence that forecasting error already measures energy-management performance. [D01; M01–M04; T01]

### Table 2: Dataset and experimental setup comparison

|Dimension|Base paper|Our research|Interpretation/status|
|---|---|---|---|
|4. Data source|Market/load/operational simulation data; §IV-A, p. 9, ref. [29]|M100 ExaData, raw Parquet, sensor audit and saved experiments|Different label generation|
|5. Entities|Five modeled data centers; Table II, p. 9|979 mapped compute-node clients in one facility|Not 979 physical data centers|
|6. Heterogeneity|Distinct local datasets and unequal update acceptance; Fig. 5|Unequal histories, counts, observed node errors|Formal non-IID taxonomy not fully evaluated|
|7. Missing data|Comparable sensor-gap treatment not verified in PDF|Coverage-controlled bins; no-fill/FF10/FF20/FF30 inputs|Our target values remain observed|
|8. Inputs|Prices, workload arrivals and queue state; §III-A, p. 4|14 operational, thermal, weather and calendar slots|Some executed tracks mask weather|
|10. Temporal formulation|Multi-period day-ahead decisions, Eq. (1)|6/36/144 ten-minute histories, next-six-bin mean|Different horizon and origin contracts|

**Interpretation.** This project's data are real telemetry, but the federation is simulated and includes known nodes. More clients does not establish stronger multi-site generalization. The historical v5 common test contains 6,590 windows on one date; B1 contains 194,212 hourly March windows; T4's canonical test contains 243,138 overlapping-origin windows. These populations must remain distinct. The paper's exact sample split and comparable telemetry-gap rates are Not verified from available artifacts in this review. [V01; B01; C03; T04]

### Table 3: Model and federated learning comparison

|Dimension|Base paper|Our research|Interpretation/status|
|---|---|---|---|
|11. Neural model|MLP ensemble, hidden [256,1024,256], ReLU/dropout 0.15; Eq. (4), §IV-A|GRU/LSTM plus node embedding; B1/T4 2×64 GRU|No paper MLP-ensemble reproduction|
|12. FL method|Conditional adaptive sharing/acceptance; Algorithm 1, p. 6|Standard sample-weighted FedAvg|Different algorithm|
|13. Aggregation|Selected shared blocks, weighted aggregate and local momentum mixing; Eqs. (5),(10),(13)|Full state including embedding; accepted-example weighting|No custom row-wise embedding rule|
|14. Local learning|1,500 warm-up steps, batch 512; 25×800 local steps|B1 SGD .01, batch 128, one local epoch, 200 rounds|Not equal computational budgets|
|15. Adaptation|Recent median/slope/volatility acceptance; Eqs. (11), temporary opt-out|Uniform fixed 98/979 sampling in principal runs|Adaptive participation Not Implemented|

**Interpretation.** Full-table averaging dilutes a node-specific embedding update when unchanged copies from other selected clients are averaged with it; that is documented standard FedAvg behavior here. It is unrelated to the paper's selective block sharing and acceptance. T4 further differs from B1: Adam, batch 64, residual L1 training and float32 aggregation are implemented. An average over all these versions would obscure the experiment. FedProx remains a proposed controlled comparison, not a completed adaptive substitute. [M01–M04; T09; F01]

### Table 4: Prediction and evaluation comparison

|Dimension|Base paper|Our research|Interpretation/status|
|---|---|---|---|
|9. Targets|Operational decisions/states, including servers, queue, cooling and temperature; §III-A|Observed mean node power; B1 also recorded IPMI ambient|Different semantics and physical scale|
|16. Evaluation|Independent/default/adaptive learning plus optimization reference; §IV|Chronological holdouts, frozen identities, persistence, selected centralized controls|Different reference questions|
|17. Prediction metrics|NRMSE averaged across decision dimensions, R²; Eq. (15), p. 9|Pooled and macro-node MAE/RMSE, W or °C|Not Directly Comparable|
|Selection|Conditional validation acceptance; Algorithm 1|Track-specific earliest minimum validation error|No test-based reselection authorized|
|Uncertainty|Ensemble spread, Eq. (12)|Three-seed sample SD where available|Seed SD is not calibrated predictive uncertainty|

**Interpretation.** R² measures variance explained relative to a mean predictor, while watt MAE measures absolute physical error and RMSE weights large errors more heavily. A 0.988 R² cannot be ranked against 24.8133 W MAE. Nor is a three-seed standard deviation an uncertainty interval for an individual future node. The correct empirical questions use the same labels, identities and aggregation: the historical v5 centralized control wins both pooled errors in all 12 pairs, whereas B1 beats persistence only on power RMSE among its four pooled output/error comparisons. [V02; B02; T02]

### Table 5: Energy-management and optimization comparison

|Dimension|Base paper|Our research|Interpretation/status|
|---|---|---|---|
|19. Formulation|MINLP P1, envelope/MIQP P2, reduced P3; Eqs. (1)–(3),(14)|Forecasting losses only|Operational optimization Not Implemented|
|20. Decisions|Servers, cooling, generators, storage, grid trades|No executed control action|Action-response model Not Implemented|
|21. SLA/service|Queue/delay constraints and penalty; Eqs. (1j)–(1k)|Operational proxies exist, final SLA definition absent|SLA benefit Not Evaluated|
|22. Energy cost|Modeled dispatch costs relative to reference; Table III|No same-action energy/cost experiment|Not Directly Comparable|
|23. Efficiency|Optimization workflow time, Fig. 8|Some training times; T4 VRAM fields are fixed placeholders|No comparable end-to-end speedup|
|Excluded dimensions|Privacy, key management, cryptographic protocols, cybersecurity/security mechanisms|Not compared|Out of Scope for this comparison|

**Interpretation.** Facility cooling and PUE channels exist upstream, but availability does not create a causal controller or a node-level cooling label. A forecast can support a decision only after its action variables, physical response and constraints are specified and validated. The paper's cost model supplies a useful conceptual separation, not a transferable cost measurement. This project cannot claim energy/SLA savings or use hardcoded memory fields as measured efficiency evidence. [D02; T09; F02]

### Table 6: Quantitative results comparison

|Record|Base paper's measured quantity|Our separately measured quantity|Comparison indicator|
|---|---|---|---|
|Learning accuracy|Table II, p. 9: adaptive R² .951/.988/.988/.984/.988 across DC1–5|T4 canonical FedAvg pooled MAE 24.8133 W; RMSE 57.5502 W|Not Directly Comparable: decision R² versus sensor errors|
|Learning comparator|Table II default FL R² .591/.807/.797/.642/.935|v5 centralized pooled MAE/RMSE lower than FedAvg in 12/12 pairs|Different populations and algorithms|
|Cost accuracy|Table III, p. 10: independent .12%, default .07%, adaptive .08% aggregate deviation|No operational cost evaluation|Not Evaluated|
|Runtime|Fig. 8, p. 10: ~.61 s versus ~5.30 s; ratio ~8.69×|No equivalent optimization workflow benchmark|Not Directly Comparable|
|Persistence|Mean-reference R² and independent-learning comparison, not our persistence task|B1 power MAE 53.3568 versus 42.3922 W; RMSE 89.7978 versus 92.6004 W|Within-B1 comparison only|

**Interpretation.** The paper's adaptive method does not minimize every displayed aggregate deviation: default FL's 0.07% is below adaptive FL's 0.08%. Its argument also considers component errors and decision quality. Similarly, our B1 finding is mixed and T4's favorable persistence result does not establish FL superiority over its centralized control. The runtime ratio is recomputed from approximate published times, not independently benchmarked. T4's 24.81 W pooled MAE pairs with 33.0661 W pooled persistence; 33.3504 W is a macro-node baseline. [B02; T02; T03]

### Table 7: Research gaps and opportunities derived from the comparison

|25. Opportunity|Existing foundation|Missing evidence/component|Defensible next step|
|---|---|---|---|
|Telemetry-to-decision connection|Observed forecasting and paper's physical formulation|Validated response to actions|Specify action/response/constraints before simulation claims|
|Temporal robustness|Completed FF30 history, T2/T3 and T4 panels|Consistent contracts and prospective periods|Reconcile executed cohort rules; register later evaluation|
|Client heterogeneity|Per-node counts/errors, known-node embedding|Controlled non-IID/FedProx/participation study|Fix windows and budget, vary one factor|
|Missing telemetry|24-fit input-fill matrix|Broader independent dates and failure modes|Replicate with explicit target-observation rules|
|Multi-output forecasting|B1 power/ambient execution|Uniform benefit, sensor placement/calibration|Audit targets and compare single-/multi-output baselines|
|Operational objectives|Upstream facility signals and modeled paper costs|Validated energy/SLA accounting and controller|Match units, aggregation scale and intervention evidence|

**Interpretation.** The present contribution is an implemented and traceable forecasting study with informative negative as well as positive findings. GRU, FedAvg and real telemetry are established components, not sufficient novelty claims. Intended contributions concern a controlled forecast-to-decision link and broader robustness, which remain future work. The immediate completion opportunity is the already-trained B1 centralized control's separately authorized same-window March evaluation; it would answer a missing matched forecasting comparison without pretending to reproduce the base paper's operational architecture. [B04; F01; F02]

**Latest-status qualification:** T5 weather-ablation records are completed; T6 model-family completion appeared during this session and is reported separately in main-report §11.8. Future-work language about these topics refers to further controlled replication, not absence of all results. Two pre-existing T6 progress logs changed concurrently; full-tree zero-mismatch preservation is not claimed. Core inspected source/result hashes remain unchanged.
