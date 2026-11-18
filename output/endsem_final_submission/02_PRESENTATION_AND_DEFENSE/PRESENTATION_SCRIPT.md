# Complete Master Presentation Script and Oral Defense Guide
**Student:** Mihir Chouhan (Roll No: 25CSE2014) · **Supervisor:** Dr. Pravati Swain  
**Institution:** Department of Computer Science and Engineering, National Institute of Technology Goa  
**Presentation Date:** 19 December 2026 · **Document Status:** Authoritative Defense Edition  

---

## Oral Delivery Guidelines
- **Total Slides:** 24 slides
- **Target Delivery Time:** 19 minutes 45 seconds (including planned pauses)
- **Target Speaking Rate:** 115–125 words per minute
- **Structure:** Every slide includes a spoken script (*What to Say*), in-depth technical context (*Technical Explanation*), smooth narrative bridging (*Transition*), anticipated faculty examination questions (*Likely Faculty Question*), and source references.

---

## Slide 01: Federated Forecasting and Energy-Aware Decision Support
**Section:** M.TECH MAJOR PROJECT PRESENTATION | **Subtitle:** Under Heterogeneous and Missing Telemetry in Distributed Data Centers  
**Target Speaking Duration:** 30 seconds  

### What to Say (Spoken Oral Script)
Good morning respected members of the evaluation committee. I am Mihir Chouhan (Roll No: 25CSE2014), presenting my M.Tech major project entitled 'Federated Forecasting and Energy-Aware Decision Support for Distributed Data Centers under Heterogeneous and Missing Telemetry', supervised by Dr. Pravati Swain. In this presentation, I will walk you through our complete research work: decentralized sequence modeling across 979 compute nodes from the CINECA Marconi100 supercomputer, our discovery of lookback invariance, overcoming the persistence barrier via residual learning, and establishing the exact operational boundaries toward energy-aware decision support.

### Technical Context & Mathematical Details
Presentation date: 19 December 2026. Telemetry and experimental evidence snapshot: November 2026. All compute clients are simulated node-level partitions from the Marconi100 Tier-0 facility.

### Narrative Transition to Next Slide
I will begin by framing the operational challenge that motivates this research.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Is an energy-management controller already deployed in the facility? No; this project implements and rigorously validates empirical forecasting and stress resilience, while the downstream decision support is formulated as an architectural and mathematical bridge.

**Artifact Sources:** REPORT, REGISTRY

---

## Slide 02: Forecasting is a Prerequisite, Not a Savings Result
**Section:** MOTIVATION & CONTEXT | **Subtitle:** Telemetry → Forecast → Decision Support → Measured Operational Outcome  
**Target Speaking Duration:** 40 seconds  

### What to Say (Spoken Oral Script)
High-performance computing and enterprise data centers consume megawatts of power. While energy-aware management is the ultimate goal, effective operational decisions require accurate anticipation of node-level power demands. However, forecasting at the edge is complicated by sensor missingness, collector daemon outages, and extreme workload volatility. Most crucially, in systems governed by physical and thermal inertia, naive persistence provides a formidable baseline. In this work, we rigorously distinguish pure forecasting accuracy from downstream control interventions.

### Technical Context & Mathematical Details
Power is an instantaneous rate in Watts; energy requires temporal integration. Node IT power does not automatically measure facility power, cooling energy, PUE or SLA outcomes.

### Narrative Transition to Next Slide
This motivates our core research questions and formal study structure.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why not convert a watt-error reduction directly into energy savings? Because forecast error is not an operational intervention. A controller requires a validated action-response model, operational constraints, and closed-loop evaluation.

**Artifact Sources:** REPORT, REGISTRY

---

## Slide 03: Three Guiding Research Questions
**Section:** RESEARCH OBJECTIVES | **Subtitle:** Systematic Comparisons, Architectural Optimization, and Operational Boundaries  
**Target Speaking Duration:** 40 seconds  

### What to Say (Spoken Oral Script)
Our research investigates three fundamental questions: First, can decentralized sequence models trained with FedAvg beat naive local persistence and centralized baselines? Second, how do historical context length, residual learning, environmental telemetry, and model architectures influence accuracy and computational efficiency? Third, how resilient are these models to severe telemetry missingness, and what is the exact mathematical interface to downstream decision support?

### Technical Context & Mathematical Details
Evaluated across 978 certified compute nodes at CINECA M100 over 243,138 matched test windows and multi-month rollouts exceeding 3.6 million windows.

### Narrative Transition to Next Slide
We first review the mandatory base paper to establish structural alignment and explicit scientific boundaries.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** What is the primary contribution? A scientifically controlled, evidence-traceable federated forecasting monograph establishing lookback invariance, the residual persistence solution, and sensor robustness.

**Artifact Sources:** REPORT, REGISTRY

---

## Slide 04: Base Paper: Learn Operating Decisions, Then Optimize
**Section:** MANDATORY BASE PAPER | **Subtitle:** Liu et al. • arXiv:2511.00623v3 • Methodology & Numerical Simulation Style  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
We study the mandatory base paper: Adaptive Federated Learning to Optimize Integrated Flows in Cyber–Physical Data Centers. The authors formulate day-ahead coordination of power, cooling, and workloads as a mixed-integer nonlinear program, relaxed via McCormick envelopes into MIQP. They use ensembles of four-layer MLPs across five modeled data centers to directly learn optimal operating decisions from market prices and queues. Local inferred decisions then reduce a downstream dispatch problem. This provides an excellent structural reference for our methodology and numerical evaluations.

### Technical Context & Mathematical Details
MLP hidden widths: [256, 1024, 256], ReLU, dropout 0.15. Acceptance uses recent median improvement, trend slope, and loss volatility (Eq. 11, Algorithm 1). Decaying momentum mixing: alpha = 0.1 * 0.95^r.

### Narrative Transition to Next Slide
However, our empirical task and physical topology differ fundamentally from the base paper.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Is our implementation an exact reproduction of the base paper? No; the base paper approximates an offline optimizer for 5 modeled sites, whereas we perform empirical time-series forecasting across 979 real compute nodes.

**Artifact Sources:** PAPER, COMPARISON

---

## Slide 05: Comparative Analysis: Base Paper vs. Our Research
**Section:** BASE-PAPER COMPARATIVE FRAMEWORK | **Subtitle:** Decision Learning vs. Observed Telemetry Forecasting: Fundamental Boundaries  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
This table highlights the clear distinctions between the base paper and our research. The base paper learns optimizer-derived decisions across 5 synthetic agents. Our research performs empirical time-series forecasting of physical sensor telemetry across 979 real compute nodes. Their neural models are static MLP ensembles; ours are recurrent sequence models (GRU/LSTM) with node embeddings. Their evaluation uses normalized decision RMSE and R-squared; ours reports physical error in Watts (MAE and RMSE). We do not claim reproduction of their adaptive momentum acceptance or physical optimization solver.

### Technical Context & Mathematical Details
Detailed 7-table comparative analysis is provided in Chapter 12 of the thesis report. Key dimensions: problem, setup, FL method, prediction, optimization, and research gaps.

### Narrative Transition to Next Slide
Next, we examine our implemented end-to-end research architecture.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Can R-squared from the base paper be compared to our MAE in Watts? No; R-squared measures variance explained relative to a mean predictor, whereas MAE measures absolute physical error in Watts.

**Artifact Sources:** PAPER, REPORT

---

## Slide 06: End-to-End Eleven-Stage Research Pipeline
**Section:** SYSTEM ARCHITECTURE | **Subtitle:** Evidence-Preserving Pipeline from Raw Ingestion to Decision-Support Formulation  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
Our research follows an eleven-stage methodology. Raw Parquet telemetry is ingested and verified for monotonicity. Causal ten-minute bins are formed with 300-second hold limits. Missing observations are imputed causally via FF30. Features are standardized strictly using training-split moments. Sliding windows are extracted and partitioned across 978 simulated client nodes. We train 2-layer SequenceGRUs via sample-weighted FedAvg over 200 communication rounds, select checkpoints on validation MAE, and evaluate on untouched holdout test sets.

### Technical Context & Mathematical Details
Stages: Ingestion, Validation, Preprocessing, Windowing, Client Partitioning, Model Design, FedAvg Training, Inference, Systematic Studies, Decision Support, and Synthesis.

### Narrative Transition to Next Slide
Let us inspect the telemetry feature representation in detail.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Are these clients physically decentralized across the Internet? No; the federation is simulated across strictly isolated local data partitions representing 978 compute nodes from one facility.

**Artifact Sources:** REPORT, ARCHITECTURE

---

## Slide 07: Fourteen Monitored Telemetry Channels
**Section:** DATA & FEATURE ENGINEERING | **Subtitle:** M100 ExaData Telemetry • UTC 10-Minute Grid • Training-Only Standardization  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
The telemetry feature vector comprises 14 slots: Slot 0 is node IT power in kW; Slots 1 to 4 capture CPU user, system, idle, and I/O-wait percentages; Slot 5 is memory utilization derived from aligned free and total memory; Slot 6 is recorded inlet ambient temperature; Slots 7 and 8 are outdoor temperature and humidity; Slots 9 to 12 encode diurnal and weekly cyclical time; and Slot 13 is the categorical node ordinal, which is mapped to an 8-dimensional learned embedding. Slots 7 and 8 are masked based on our empirical weather ablation findings.

### Technical Context & Mathematical Details
Input dimension to GRU: 13 continuous channels + 8 embedding dimensions = 21. Standardized moments: mu_train, sigma_train calculated exclusively on the training split.

### Narrative Transition to Next Slide
We now formalize the temporal windowing and split contracts.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why is memory utilization mathematically derived? Because hardware daemons record mem_free and mem_total; true fraction is 1 - (mem_free / mem_total).

**Artifact Sources:** FEATURES, PREPROCESS

---

## Slide 08: Chronological Windowing and Target Contract
**Section:** TEMPORAL FORMULATION | **Subtitle:** Predicting Next-Hour Mean IT Power: y_P(t) = Mean of Next Six 10-Minute Bins  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
For any forecast origin t, the model receives H completed ten-minute historical bins preceding t. The prediction target is defined as the exact arithmetic mean of the six subsequent ten-minute power bins, covering the next hour [t, t+60 min). This represents the next-hour mean IT power in Watts, rather than instantaneous power at t+60. Candidate 1 chronological splits are strictly separated: Training covers Jan 01 to Jan 22; Validation covers Jan 22 to Jan 28; and Canonical Testing covers Jan 28 to Feb 04, 2022 UTC.

### Technical Context & Mathematical Details
Target: y_{n,P}(t) = (1/6) sum_{j=0}^5 p_n([t+10j, t+10(j+1))). Targets are never forward-filled; all six future bins must be observed finite sensor readings.

### Narrative Transition to Next Slide
We now examine how the physical cluster is mapped into simulated clients.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why predict the next-hour mean rather than instantaneous power? Facility cooling and power provisioning operate on thermal time constants where hourly average heat load governs chiller dispatch.

**Artifact Sources:** REPORT, PREPROCESS

---

## Slide 09: 979 Nodes Modeled as Simulated FL Clients
**Section:** DECENTRALIZED TOPOLOGY | **Subtitle:** Node-Level Partitions from CINECA Marconi100 • Non-IID Workload Distributions  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
Each compute node in the facility represents an autonomous client. Local sliding windows remain strictly isolated within that node partition. The categorical node ordinal maps into an 8-dimensional shared embedding table, allowing the model to capture node-specific static hardware affinities. Workload heterogeneity arises naturally from HPC SLURM batch job scheduling, creating non-IID data distributions across clients without artificial synthetic partitioning.

### Technical Context & Mathematical Details
Node embedding table: 979 x 8 parameters. In each round, 98 of 979 clients (10%) are sampled uniformly. Full-table aggregation averages all rows across participating clients.

### Narrative Transition to Next Slide
We now detail the forecasting model architectures.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** What does non-IID mean in this context? Different compute nodes run distinct applications—some memory-bound, some GPU-accelerated, some idle—producing divergent power distributions.

**Artifact Sources:** REPORT, MULTINODE

---

## Slide 10: Model Formulation: Direct Power vs. Residual L1
**Section:** MODEL ARCHITECTURE | **Subtitle:** SequenceGRU Backbone (49k Parameters) with Learned Node Embedding  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
We evaluate two primary formulations: Direct Power Regression, where the network directly outputs absolute power in Watts, and Residual Delta Learning, where the network predicts the incremental deviation r_hat relative to the last observed power reading: y_hat = p_last + r_hat. The backbone is a 2-layer SequenceGRU with 64 hidden units (49,217 parameters). We pair the residual formulation with L1 loss (MAE), directly penalizing absolute prediction deviations.

### Technical Context & Mathematical Details
Parameter breakdown: Embedding: 7,832; Layer 1: 16,704; Layer 2: 24,960; Linear Head: 65. Total: 49,561 params. Direct linear projection without auxiliary hidden layers.

### Narrative Transition to Next Slide
Next, we examine the federated optimization procedure.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why is residual learning advantageous? Power time series exhibit high temporal auto-correlation. Predicting delta deviations anchors the forecast to current operational levels, eliminating level-shift drift.

**Artifact Sources:** MODEL, T4

---

## Slide 11: Sample-Weighted FedAvg Optimization
**Section:** FEDERATED TRAINING PROCEDURE | **Subtitle:** 10% Client Sampling • 1 Local Epoch Adam • Validation-Guided Checkpointing  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
Training executes over 200 communication rounds. In each round, the coordinator samples 98 clients uniformly without replacement. Each client executes 1 local epoch of Adam optimization (learning rate 0.001, batch size 64) on its private training data. The coordinator aggregates parameter updates using sample-weighted averaging, proportional to each client's valid training window count. Checkpoint selection is strictly validation-guided, tracking minimum validation MAE with zero test-set reselection.

### Technical Context & Mathematical Details
FedAvg equation: theta^{r+1} = sum_k (n_k / sum n_j) theta_k^{r+1}. Validation evaluated every round. Strict early stopping / best checkpoint selection.

### Narrative Transition to Next Slide
Let us examine the evaluation definitions.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why weight by sample count n_k? Sample-weighting ensures that clients with more observed telemetry have proportional influence on the global model, preventing under-sampled nodes from skewing updates.

**Artifact Sources:** FED_TRAIN, REPORT

---

## Slide 12: Evaluation Definitions and Baseline Design
**Section:** EVALUATION PROTOCOL | **Subtitle:** Pooled vs. Macro Metrics in Watts • The Canonical Persistence Benchmark  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
To prevent misleading aggregations, we report both Pooled Metrics (weighting every test window equally) and Macro Metrics (computing errors independently per node, then averaging equally across all 978 nodes). All errors are reported in physical Watts. The canonical benchmark is Naive Persistence P[t-1], which predicts the last observed power sample for the future mean. In high-inertia data center systems, persistence provides a formidable target that models must decisively outperform.

### Technical Context & Mathematical Details
Pooled MAE: (1/N) sum |y - y_hat|. Macro MAE: (1/K) sum_k MAE_k. Macro RMSE is the mean of node RMSEs, not the root of mean squared node errors.

### Narrative Transition to Next Slide
We now present the completed experimental evidence across our portfolio of studies.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why do pooled and macro metrics differ? Because window counts vary across nodes due to sensor missingness. Macro metrics prevent heavily sampled nodes from dominating the evaluation.

**Artifact Sources:** REPORT, EVAL

---

## Slide 13: Master Experiment Portfolio and Evidence Map
**Section:** RESEARCH EXPERIMENT REGISTRY | **Subtitle:** Twelve Individually Audited Studies Covering the Complete Research Lifecycle  
**Target Speaking Duration:** 40 seconds  

### What to Say (Spoken Oral Script)
Our experimental findings comprise twelve systematically audited studies: from early short-horizon explorations and missing-data matrices, to 6-hour baselines, multi-month temporal rollouts, continual adaptation, residual learning improvements, weather ablations, architecture comparisons, missingness stress testing, common coverage audits, controlled history benchmarks, and decision-support modeling. Each study is independently audited with locked artifacts and verified checksums.

### Technical Context & Mathematical Details
All experiments registered in docs/research/EXPERIMENT_REGISTRY.csv and EXPERIMENT_REGISTRY.md. Provenance maintained via SHA-256 manifests.

### Narrative Transition to Next Slide
We begin by examining the historical baseline and missing telemetry findings.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Are these results from one single training run? No; they represent over 60 distinct model fits across multiple seeds, cohorts, and experimental conditions.

**Artifact Sources:** REGISTRY, REPORT

---

## Slide 14: Historical Baselines & Missing Telemetry Evaluation
**Section:** STUDIES 1 & 2: BASELINES & MISSINGNESS | **Subtitle:** Centralized SGD Outperforms FedAvg in 12/12 Pairs; FF30 Stabilizes Training  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
In Study 1, early 1-hour exploratory models established feasibility (27.93 W MAE). In Study 2, our 24-fit missing-data matrix evaluated FedAvg vs Centralized SGD across no-fill, FF10, FF20, and FF30 under acute sensor outages. Centralized SGD achieved lower MAE and RMSE in all 12 matched pairs. Forward-fill imputation (FF20/FF30) stabilized training coverage across collector drops. However, on direct power prediction, learned models lagged persistence MAE (27.15 W persistence vs 29.82 W FedAvg).

### Technical Context & Mathematical Details
24-fit matrix: 4 policies x 2 methods x 3 seeds. Centralized SGD: 26.6-29.0 W Pooled MAE; FedAvg: 29.7-30.0 W Pooled MAE. Common test: 6,590 windows across 570 nodes.

### Narrative Transition to Next Slide
Next, we examine multi-output forecasting challenges.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why did centralized SGD beat FedAvg? Under non-IID local data partitions, client drift degrades decentralized SGD updates relative to a centralized model accessing all batches sequentially.

**Artifact Sources:** V5, REPORT

---

## Slide 15: Multi-Output Forecasting & Temporal Generalization
**Section:** STUDIES 3, 4, 5: HORIZONS & DRIFT | **Subtitle:** Direct Power Underperforms Persistence (-0.92 W); Unadapted Models Experience Drift  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
In Study 3, extending history to 6 hours with direct power prediction achieved 33.99 W MAE, lagging persistence (33.07 W) by -0.92 W (29.8% node win rate). In B1 multi-output forecasting, adding ambient temperature improved power RMSE slightly (89.8 W vs 92.6 W) but degraded power MAE (53.4 W vs 42.4 W). In Study 4, multi-month rollout across Q1 (3.6M windows) showed severe temporal drift (42.58 W MAE). In Study 5, periodic weekly fine-tuning successfully arrested drift by 15-20%.

### Technical Context & Mathematical Details
Direct power MSE models focus on absolute levels and fail against persistence. B1 joint loss introduces gradient competition between Watt and Celsius scales. Q1 drift proves need for adaptation.

### Narrative Transition to Next Slide
This motivated our breakthrough: reformulating the learning objective into residual delta forecasting.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why did multi-output modeling fail to improve power? Because recorded ambient temperature telemetry had low correlation with immediate power spikes, introducing optimization noise.

**Artifact Sources:** B1, T2, T3, REPORT

---

## Slide 16: Model Improvement: Breaking the Persistence Barrier
**Section:** STUDY 6: RESIDUAL L1 LEARNING | **Subtitle:** FedAvg Reaches 24.81 W MAE • Beats Persistence on 91.9% of Nodes (+24.96% Gain)  
**Target Speaking Duration:** 55 seconds  

### What to Say (Spoken Oral Script)
Study 6 represents our major architectural breakthrough. By reformulating the learning task as predicting the residual delta relative to the last observed power reading and training with L1 loss, FedAvg-GRU achieves a Pooled MAE of 24.81 W against 33.07 W for persistence—a decisive 24.96% improvement (+8.25 W). Macro MAE reaches 25.22 W vs 33.35 W. Most remarkably, FedAvg-GRU outperforms persistence on 898.3 out of 978 compute nodes (91.85% win rate). Centralized-GRU achieves 23.99 W MAE (94.4% win rate).

### Technical Context & Mathematical Details
Audited on Canonical T1 Test (243,138 windows, 978 nodes). Verified across seeds 42, 43, 44. Pooled RMSE drops to 57.55 W vs 64.05 W persistence. Six checkpoint hashes cryptographically verified.

### Narrative Transition to Next Slide
We next test if these gains persist across longer temporal rollouts.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why did residual L1 succeed where direct MSE failed? Residual formulation removes the high-variance DC level offset, allowing the GRU to focus its full capacity on learning high-frequency workload transitions.

**Artifact Sources:** T4, T4_AUDIT, REPORT

---

## Slide 17: Temporal Robustness Across Multi-Month Rollouts
**Section:** STUDY 6 CONTINUED: EXTENDED GENERALIZATION | **Subtitle:** Consistent Persistence Dominance Across Q1 (98.0% Wins) and March (97.7% Wins)  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
We evaluated the frozen residual FedAvg-GRU model across multi-month holdouts without retraining. On the full Q1 period (3,609,285 windows across 979 nodes), FedAvg-GRU achieves 33.97 W Pooled MAE vs 40.44 W for persistence—a +6.47 W improvement, winning on 98.0% of nodes. On March (2,229,042 windows), it achieves 38.28 W vs 45.29 W persistence—a +7.01 W improvement, winning on 97.7% of nodes. While absolute error rises due to quarterly workload shifts, the model consistently outperforms persistence across all horizons.

### Technical Context & Mathematical Details
Q1 Extended: 3.6M windows; March: 2.2M windows. Note: Canonical and March are nested temporal subsets of Q1. Node win counts: Q1: 959.7 / 979 nodes; March: 953.7 / 976 nodes.

### Narrative Transition to Next Slide
We next address the central question of historical context length.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Does the model need monthly retraining? While zero-shot transfer retains strong persistence gains, periodic adaptation further reduces absolute error as shown in Study 5.

**Artifact Sources:** T4, T4_SEEDS, REPORT

---

## Slide 18: Temporal Context Study: Lookback Invariance
**Section:** STUDY 11: 1h vs. 6h vs. 24h BENCHMARK | **Subtitle:** The Lookback Invariance Law: 1h Context Dominates; 24h Adds +28.5% Compute Cost  
**Target Speaking Duration:** 55 seconds  

### What to Say (Spoken Oral Script)
Study 11 delivers a decisive scientific finding: the Lookback Invariance Law. In our controlled 9-fit multi-seed benchmark across 1h (H=6), 6h (H=36), and 24h (H=144) lookbacks on identical 243,138 test windows across 978 nodes, all three histories achieve virtually identical accuracy: 1h achieves 33.90 W Pooled MAE, 6h achieves 33.99 W, and 24h achieves 33.99 W (|delta| < 0.09 W). However, the 24-hour lookback incurs a +28.5% training wall-clock penalty (1584 s vs 1233 s) and 4x memory overhead. Recent 1-hour workload momentum is the dominant predictive driver.

### Technical Context & Mathematical Details
Audited results: 1h: 33.9032 +- 0.5573 W; 6h: 33.9868 +- 0.6312 W; 24h: 33.9873 +- 0.6321 W. Training wall-clock: 1h: 1232.8 s; 6h: 1357.1 s; 24h: 1584.4 s. Option A No-fill: 1h (34.97 W) vs 6h (35.06 W) vs 24h (35.06 W).

### Narrative Transition to Next Slide
We next examine environmental telemetry and architectural comparisons.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why does 24-hour context fail to improve accuracy? Data center batch jobs and HPC workloads exhibit rapid state transitions. Long-range historical telemetry contains stale workload states that do not provide informative gradient signal for next-hour predictions.

**Artifact Sources:** T1_2, T1_2_AUDIT, REPORT

---

## Slide 19: Environmental Telemetry & Model Comparisons
**Section:** STUDIES 7 & 8: WEATHER & ARCHITECTURES | **Subtitle:** Weather Channels Contribute Zero Gain; GRU Matches LSTM with 22.2% Fewer Params  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
In Study 7, our 4-condition weather ablation proved that outdoor temperature and humidity contribute zero predictive gain (< 0.03 W difference between masked and enabled models), confirming that outdoor weather is buffered by facility CRAC units and should be masked. In Study 8, our architectural benchmark proved that 2-layer SequenceGRU (49k parameters) matches SequenceLSTM (63k parameters) within 0.035 W (24.81 W vs 24.85 W MAE) while saving 22.2% parameters, and both recurrent models outperform Centralized XGBoost trees (25.31 W MAE).

### Technical Context & Mathematical Details
Weather ablation: Masked: 24.813 W; +Temp: 24.832 W; +Hum: 24.818 W; +Both: 24.840 W. Architecture: GRU: 24.813 W (49,217 params); LSTM: 24.848 W (63,297 params); XGBoost: 25.309 W (504 flat features).

### Narrative Transition to Next Slide
Next, we evaluate model robustness under operational telemetry missingness.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why does XGBoost underperform GRU? Tabular flattening (36x14 = 504 features) destroys sequential inductive bias and temporal recurrence, resulting in higher extrapolation error on dynamic workloads.

**Artifact Sources:** T5, T6, REPORT

---

## Slide 20: Telemetry Missingness Robustness Stress Testing
**Section:** STUDY 9: INFERENCE-TIME SENSOR OUTAGES | **Subtitle:** High Resilience Up to 20% Random Loss (<0.6% Degradation); Power Channel Critical  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
In Study 9, we stress-tested the frozen FedAvg-GRU model across 13 synthetic missingness conditions without retraining. The model demonstrated remarkable robustness: 5% random missingness increased MAE by only +0.07% (24.83 W), 10% by +0.21% (24.86 W), and 20% by +0.61% (24.96 W). Contiguous multi-hour bursts were more disruptive, with 2-hour bursts at 30% raising MAE to 36.10 W. Channel blackout ablations revealed that Power telemetry is the single mission-critical channel (+100.16 W degradation); CPU, Memory, and Temperature blackouts added less than 0.21 W error.

### Technical Context & Mathematical Details
Evaluated on Canonical T1 Test (243,138 windows, 978 nodes). 0% clean: 24.813 W; 20% random: 26.520 W; 2h burst 30%: 36.105 W; Power blackout: 124.971 W (vs 124.822 W degraded persistence).

### Narrative Transition to Next Slide
Let us examine client-level error distributions under corruption.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why is contiguous burst loss worse than random loss? Because contiguous gaps exceed the FF30 30-minute hold limit, forcing the model into standardized training-mean fallback.

**Artifact Sources:** T7, T7_AUDIT, REPORT

---

## Slide 21: Client Heterogeneity and Channel Sensitivity
**Section:** STUDY 9 CONTINUED: HETEROGENEITY | **Subtitle:** 91.85% Win Rate Under Clean Telemetry; Preserves >90% Win Rate at 20% Loss  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
Under clean telemetry, FedAvg-GRU outperforms persistence on 91.85% of compute nodes. Under 20% random missingness, it preserves a 90.7% win rate. Under 20% 1-hour burst outages, win rate remains 66.3%. However, when power telemetry experiences complete failure, MAE jumps to 124.97 W, slightly worse than degraded persistence. This establishes that the model's primary dependency is the recent power anchor, while auxiliary channels provide fine-grained modulation.

### Technical Context & Mathematical Details
Node win rate: Clean: 91.85%; Random 20%: 90.70%; 1h Burst 20%: 66.33%; Power Blackout: 48.2%. Denominator: 7,154,532 node-grid channel values.

### Narrative Transition to Next Slide
We now present the conceptual connection to energy-aware decision support.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** What does this imply for edge monitoring daemons? Facility operators must ensure high reliability for the primary power sensor, while CPU/memory monitoring can tolerate significant dropout without forecasting failure.

**Artifact Sources:** T7, REPORT

---

## Slide 22: Energy-Aware Operational Decision Support
**Section:** STUDY 12: DECISION-SUPPORT FRAMEWORK | **Subtitle:** Mathematical Coupling of Power Forecasts to Facility Cooling and Job Dispatch  
**Target Speaking Duration:** 50 seconds  

### What to Say (Spoken Oral Script)
Study 12 formulates the operational bridge connecting node-level forecasts to facility management. In our framework, aggregate forecasted cluster power P_IT(t+1) = sum y_n(t) defines the thermal load that chillers and CRAC units must reject. The optimization problem minimizes total electricity cost: min sum (P_IT + P_cooling(T_chiller, T_outdoor)) * lambda_grid subject to server inlet temperature bounds (T_inlet <= T_max) and peak power contract caps. In accordance with strict academic integrity, we emphasize that this framework is formulated mathematically; no live facility control loop or physical simulator was executed, and no empirical energy savings are claimed.

### Technical Context & Mathematical Details
Formulation couples next-hour IT load to chiller COP curves, thermal RC circuit dynamics, and SLURM workload balancing constraints.

### Narrative Transition to Next Slide
Let us review the limitations and verification boundaries of our work.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why is it critical to state that energy savings were not measured? Because claiming energy savings without an integrated physical simulator or live facility deployment violates scientific integrity. Forecasting accuracy is an essential prerequisite, not a measured operational saving.

**Artifact Sources:** REPORT, PRD, ARCHITECTURE

---

## Slide 23: Research Limitations & Verification Boundaries
**Section:** METHODOLOGICAL BOUNDARIES | **Subtitle:** Single Facility Dataset • Simulated Federation • Rigorous Cryptographic Auditing  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
We explicitly document our research limitations: First, all telemetry originates from a single supercomputing facility (CINECA M100); cross-facility transferability to air-cooled data centers remains future work. Second, federated clients are simulated partitions running on shared GPU hardware rather than physically dispersed servers over wide-area networks. Third, historical strict preflight remains FAILED for older v5 runs, which is why all new studies (T1–T7) implement fresh, cryptographic SHA-256 manifests, locked configurations, and independent integrity audits.

### Technical Context & Mathematical Details
All 22 manifest-listed T4 files match hashes. All 6 T1.2 checkpoint hashes match seed audits. 100% PASS on independent integrity verification tools.

### Narrative Transition to Next Slide
We now conclude with a summary of verified research contributions.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** Why preserve the historical FAILED preflight status? Because transparent science requires reporting past artifact cache mismatches rather than overwriting historical evidence.

**Artifact Sources:** REPORT, AUDIT

---

## Slide 24: Conclusion & Future Research Directions
**Section:** CONTRIBUTIONS & SUMMARY | **Subtitle:** Decisive Findings: Lookback Invariance, Residual Persistence Solution, Sensor Resilience  
**Target Speaking Duration:** 45 seconds  

### What to Say (Spoken Oral Script)
In conclusion, this research delivers five decisive contributions: First, we proved the Lookback Invariance Law, establishing that 1-hour context is sufficient for next-hour forecasting, saving 28.5% training time and 4x memory. Second, our residual L1 formulation solved the persistence barrier, outperforming persistence across 91.9% of compute nodes (+24.96% MAE gain). Third, we proved outdoor weather is irrelevant for indoor node power. Fourth, SequenceGRU was proven superior to tree-based models while saving 22.2% parameters over LSTM. Fifth, we proved high resilience up to 20% telemetry loss. Future work will integrate these forecasts into closed-loop physical cooling simulators.

### Technical Context & Mathematical Details
Monograph deliverables: Final Report (output/final_endsem_report_v2/), Master Registry (docs/research/), Presentation & Defense Package (output/final_endsem_presentation_v1/), and Codebase Monograph (docs/research/CODEBASE_EXPLANATION.md).

### Narrative Transition to Next Slide
Thank you respected committee members. I am now ready to take your questions.

### Anticipated Faculty Examination Question & Model Defense
> **Q:** What is the immediate next step for this project? Integrating the audited FedAvg-GRU model with a validated physical thermal simulator (such as EnergyPlus or CFD models) to evaluate closed-loop cooling setpoint adjustments.

**Artifact Sources:** REPORT, REGISTRY

---

