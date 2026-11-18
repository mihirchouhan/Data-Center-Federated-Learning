# Central Experiment Registry and Research Terminology Glossary

**Project Title:** Federated Forecasting and Energy-Aware Decision Support for Distributed Data Centers under Heterogeneous and Missing Telemetry  
**Institution:** National Institute of Technology Goa, Department of Computer Science and Engineering  
**Date of Presentation:** 19 December 2026  
**Registry Version:** 2.0 (Standardized Research Edition)  

---

## 1. Overview and Scientific Naming Policy

This document establishes the authoritative, definitive registry of all research investigations conducted within the project repository. In accordance with academic research standards, all reader-facing deliverables—including the project thesis report, presentation slides, speaker scripts, viva documentation, and codebase technical guides—must employ descriptive scientific titles rather than short internal development labels (such as B1, T1, T2, T4, T5, T6, T7, T1.1, T1.2, or v5).

Underlying directory paths, Git history, script names, configuration files, and saved model checkpoints retain their original identifiers to strictly guarantee cryptographic provenance, auditability, and backward compatibility. The table below maps each internal identifier to its standardized formal scientific title, research question, cohort, status, and verified outcome.

---

## 2. Master Experiment Registry Table

| Formal Scientific Title | Internal ID | Research Question | Model & Strategy | Cohort & Windows | Status | Key Audited Result |
|---|---|---|---|---|---|---|
| **Short-Horizon Federated Power Forecasting Study** | `historical_v5_b1` | Can a shared sequence model trained across distributed compute nodes forecast 1-hour ahead mean node IT power using a 1-hour historical input context? | Shared Sequence LSTM / GRU (2 layers) (FedAvg (sample-weighted)) | 979 compute nodes at CINECA facility | 1 hour (6 ten-minute steps, H=6) | Completed, but with historical limitations | Pooled MAE ~33.28 W, Persistence MAE ~33.35 W |
| **Federated Learning and Missing-Telemetry Evaluation** | `v5_missing_data_matrix` | How do different forward-fill imputation policies (no_fill, ff10, ff20, ff30) affect training stability and generalization of FedAvg vs Centralized SGD under operational sensor outages? | 2-layer SequenceGRU (49k parameters) (FedAvg (sample-weighted, 200 rounds) vs Centralized SGD) | 979 compute nodes | 6 hours (36 ten-minute steps, H=36) | Completed, but with documented sensor outage limitations | Centralized SGD: 26.6-29.0 W Pooled MAE; FedAvg: 29.7-30.0 W Pooled MAE |
| **Six-Hour Context-Based Federated Forecasting Study** | `t1_frozen_6h` | Does extending historical context to 6 hours (36 ten-minute steps) with direct IT power prediction allow FedAvg-GRU to outperform naive local persistence? | 2-layer SequenceGRU (49k parameters, hidden dim 64) (FedAvg (sample-weighted, 200 rounds, 1 local epoch)) | 978 compute nodes (Node 704 excluded due to zero valid windows) | 6 hours (36 ten-minute steps, H=36) | Completed and verified | FedAvg: Pooled MAE 33.99 W, Macro MAE 34.26 W, Pooled RMSE 63.77 W; Persistence: Pooled MAE 33.07 W, Macro MAE 33.35 W |
| **Extended Temporal Generalization Study** | `t2_temporal_generalization` | How well does a frozen 6-hour FedAvg-GRU model generalize out-of-sample over multi-month operating horizons without parameter updates? | Frozen 2-layer SequenceGRU from Six-Hour Context Study (Zero-shot inference from frozen global checkpoints) | 979 compute nodes (Q1: 3,609,285 windows; March: 2,229,042 windows across 976 nodes) | 6 hours (36 ten-minute steps, H=36) | Completed and verified | Q1: Pooled MAE 42.58 W vs Persistence 40.44 W; March: Pooled MAE 46.90 W vs Persistence 45.29 W |
| **Temporal Adaptation and Continual Fine-Tuning Study** | `t3_temporal_adaptation` | Can periodic local client fine-tuning mitigate temporal concept drift in multi-month federated deployment? | 2-layer SequenceGRU (Continual FedAvg fine-tuning (periodic client updates)) | 979 compute nodes | 6 hours (36 ten-minute steps, H=36) | Completed and verified | Adapted Pooled MAE reduced drift by ~15-20% relative to frozen static baseline |
| **Federated Forecasting Model-Improvement Study** | `t4_model_improvement` | Can a residual learning formulation (predicting residual delta relative to last observed power) paired with L1 loss consistently surpass naive persistence in federated forecasting? | 2-layer SequenceGRU with Residual L1 formulation: y_pred = y_last + r_pred (FedAvg (sample-weighted, 200 rounds, 1 local epoch)) | 978 compute nodes on Canonical T1 Test (243,138 windows); 979 nodes on Q1 (3,609,285 windows) | 6 hours (36 ten-minute steps, H=36) | Completed, audited, and frozen | FedAvg: Pooled MAE 24.81 W, Macro MAE 25.22 W (beats persistence on 91.9% of nodes, +8.25 W improvement); Centralized: Pooled MAE 23.99 W, Macro MAE 24.37 W (94.4% win rate) |
| **Environmental Telemetry Contribution Study** | `t5_weather_ablation` | Does outdoor environmental telemetry (ambient temperature and relative humidity) contribute predictive information for indoor compute-node IT power forecasting? | 2-layer SequenceGRU with Residual L1 formulation (FedAvg (sample-weighted, 200 rounds)) | 978 compute nodes (243,138 matched test windows) | 6 hours (36 ten-minute steps, H=36) | Completed, audited, and frozen | Cond A (Masked): 24.813 W; Cond B (+Temp): 24.832 W; Cond C (+Hum): 24.818 W; Cond D (+Both): 24.840 W (Differences < 0.03 W, within noise std) |
| **Sequence and Tree-Based Forecasting Comparison** | `t6_architecture_benchmark` | How does the selected SequenceGRU compare against SequenceLSTM and Centralized XGBoost in forecasting accuracy, parameter efficiency, and communication overhead? | FedAvg-GRU (49k params) vs FedAvg-LSTM (63k params) vs Centralized XGBoost (504 flat features) (FedAvg (GRU, LSTM) vs Centralized pooling (XGBoost)) | 978 compute nodes (243,138 matched test windows) | 6 hours (36 ten-minute steps, H=36) | Completed, audited, and frozen | GRU: Pooled MAE 24.81 W (49k params); LSTM: Pooled MAE 24.85 W (63k params); XGBoost: Pooled MAE 25.31 W (depth 6, 200 trees) |
| **Missing-Telemetry Robustness and Sensitivity Study** | `t7_missing_telemetry_robustness` | How robust is the frozen FedAvg-GRU model to inference-time telemetry missingness, and which telemetry channels are most critical for forecasting accuracy? | Frozen FedAvg-GRU-T4 reference models (Seeds 42, 43, 44) (Zero-shot inference-time stress testing (no retraining)) | 978 compute nodes | 6 hours (36 ten-minute steps, H=36) | Completed, audited, and frozen | 0%: 24.81 W; 10% random: 24.86 W (+0.21%); 20% random: 24.96 W (+0.61%); 30% contiguous: 25.33 W (+2.09%); Power channel loss: +11.8 W degradation |
| **Common History Coverage and Telemetry Outage Audit** | `t1_1_common_history_coverage` | Why did earlier investigations report zero valid test windows for 24-hour history, and can a scientifically rigorous common cohort be certified across 1h, 6h, and 24h lookbacks? | N/A (Data engineering and coverage audit) (N/A) | 978 certified compute nodes (Node 704 excluded due to zero valid windows) | 1h (H=6), 6h (H=36), 24h (H=144) | Completed, audited, and frozen | Certified Candidate 1 FF30 universe: exactly 243,138 matched test windows common across 1h, 6h, and 24h histories on 978 nodes; identified late-Feb Ganglia outage as root cause of prior zero-coverage. |
| **Temporal Context and Forecasting-Horizon Study** | `t1_2_controlled_history_benchmark` | Does expanding historical input context from 1 hour to 6 hours or 24 hours improve federated next-hour node IT power forecasting, and what are the computational trade-offs? | 2-layer SequenceGRU (49k params) (FedAvg (sample-weighted, 200 rounds, 9 full fits across 3 seeds)) | 978 compute nodes (Primary Test: 243,138 windows; Secondary Option A: 50,392 windows across 635 nodes) | 1h (H=6), 6h (H=36), 24h (H=144) | Completed, audited, and frozen | 1h: 33.90 W Pooled MAE (1233s); 6h: 33.99 W Pooled MAE (1357s); 24h: 33.99 W Pooled MAE (1584s). Persistence: 33.07 W. Lookback invariance verified (|delta| < 0.09 W). 24h incurs +28.5% training runtime penalty. |
| **Energy-Aware Operational Decision Study** | `decision_support_framework` | How can decentralized node-level power forecasts be integrated into cluster-level job scheduling and facility cooling decision-support systems? | FedAvg-GRU forecasting module coupled to constrained optimization solver (Federated forecast ingestion with centralized or distributed dispatch) | Cluster and facility scale (979 nodes aggregated into rack/hall cooling zones) | 6 hours (H=36) | Planned / Conceptual Framework Designed (No deployed simulator) | Theoretical formulation completed; simulation integration remains future work |

---

## 3. Detailed Experiment Profiles

### Study 1: Short-Horizon Federated Power Forecasting Study
- **Internal Identifier:** `historical_v5_b1`
- **Scientific Research Question:** Can a shared sequence model trained across distributed compute nodes forecast 1-hour ahead mean node IT power using a 1-hour historical input context?
- **Research Objective:** Establish the feasibility of decentralized sequence modeling across compute nodes and evaluate baseline forecasting performance against local persistence baselines.
- **Dataset and Period:** M100 HPC Telemetry (Ganglia/IPMI, Jan-Mar 2022)
- **Experimental Cohort:** 979 compute nodes at CINECA facility
- **Temporal Context & Horizon:** History: 1 hour (6 ten-minute steps, H=6) | Forecast Horizon: 1 hour ahead (mean IT power)
- **Model Architecture:** Shared Sequence LSTM / GRU (2 layers)
- **Optimization & Federated Protocol:** FedAvg (sample-weighted)
- **Missing Telemetry Policy:** no_fill / exploratory hold
- **Random Seeds:** 42, 43, 44
- **Evaluation Protocol:** Holdout chronological test period; macro and pooled error evaluation against persistence baseline
- **Verified Metrics & Findings:** Pooled MAE ~33.28 W, Persistence MAE ~33.35 W
- **Scientific Status:** **Completed, but with historical limitations**
- **Source Configurations:** `configs/historical/v5_base.yaml, src/models/v5_model.py`
- **Evidence & Result Artifacts:** `output/experiments/b1_no_ff/, output/experiments/v1/`
- **Limitations & Methodological Bounds:** Evaluated on uncalibrated March outage period; mixed persistence comparison; uncalibrated temperature telemetry.

### Study 2: Federated Learning and Missing-Telemetry Evaluation
- **Internal Identifier:** `v5_missing_data_matrix`
- **Scientific Research Question:** How do different forward-fill imputation policies (no_fill, ff10, ff20, ff30) affect training stability and generalization of FedAvg vs Centralized SGD under operational sensor outages?
- **Research Objective:** Measure the sensitivity of federated aggregation and centralized optimization to transient collector missingness across an empirical 24-fit experimental matrix.
- **Dataset and Period:** M100 HPC Telemetry (Feb 20-Mar 01, 2022 sensor outage period)
- **Experimental Cohort:** 979 compute nodes
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** 2-layer SequenceGRU (49k parameters)
- **Optimization & Federated Protocol:** FedAvg (sample-weighted, 200 rounds) vs Centralized SGD
- **Missing Telemetry Policy:** no_fill, ff10 (10m), ff20 (20m), ff30 (30m)
- **Random Seeds:** 42, 43, 44 (24 fits total)
- **Evaluation Protocol:** Paired seed evaluation over late-February holdout test window
- **Verified Metrics & Findings:** Centralized SGD: 26.6-29.0 W Pooled MAE; FedAvg: 29.7-30.0 W Pooled MAE
- **Scientific Status:** **Completed, but with documented sensor outage limitations**
- **Source Configurations:** `output/experiments/v5_missing_data/model_comparison/v1/`
- **Evidence & Result Artifacts:** `output/experiments/v5_missing_data/model_comparison/v1/results_analysis/`
- **Limitations & Methodological Bounds:** Evaluated during an acute Ganglia daemon collector drop, exaggerating missingness for long history lengths; single late-Feb evaluation slice.

### Study 3: Six-Hour Context-Based Federated Forecasting Study
- **Internal Identifier:** `t1_frozen_6h`
- **Scientific Research Question:** Does extending historical context to 6 hours (36 ten-minute steps) with direct IT power prediction allow FedAvg-GRU to outperform naive local persistence?
- **Research Objective:** Provide a rigorous benchmark of standard direct-power SequenceGRU federated forecasting against persistence across a 978-node population.
- **Dataset and Period:** M100 HPC Telemetry (Candidate 1: Jan 01-Feb 04, 2022)
- **Experimental Cohort:** 978 compute nodes (Node 704 excluded due to zero valid windows)
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** 2-layer SequenceGRU (49k parameters, hidden dim 64)
- **Optimization & Federated Protocol:** FedAvg (sample-weighted, 200 rounds, 1 local epoch)
- **Missing Telemetry Policy:** Forward-fill max 30 mins (FF30)
- **Random Seeds:** 42, 43, 44
- **Evaluation Protocol:** Canonical T1 test set (Jan 28-Feb 04, 2022 UTC, 243,138 matched windows)
- **Verified Metrics & Findings:** FedAvg: Pooled MAE 33.99 W, Macro MAE 34.26 W, Pooled RMSE 63.77 W; Persistence: Pooled MAE 33.07 W, Macro MAE 33.35 W
- **Scientific Status:** **Completed and verified**
- **Source Configurations:** `configs/t1_direct_power_6h.yaml, src/train_fedavg.py`
- **Evidence & Result Artifacts:** `output/experiments/temporal_generalization_6h_v1/, output/final_endsem_presentation_v1/PRESENTATION_METRICS.csv`
- **Limitations & Methodological Bounds:** Direct power formulation slightly underperformed persistence on MAE (-0.92 W) because absolute level shifts dominated the MSE loss.

### Study 4: Extended Temporal Generalization Study
- **Internal Identifier:** `t2_temporal_generalization`
- **Scientific Research Question:** How well does a frozen 6-hour FedAvg-GRU model generalize out-of-sample over multi-month operating horizons without parameter updates?
- **Research Objective:** Quantify temporal concept drift across varying quarterly HPC workload regimes over 3.6 million evaluation windows.
- **Dataset and Period:** M100 HPC Telemetry (Q1 2022: Jan 28-Apr 01 and March 2022)
- **Experimental Cohort:** 979 compute nodes (Q1: 3,609,285 windows; March: 2,229,042 windows across 976 nodes)
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** Frozen 2-layer SequenceGRU from Six-Hour Context Study
- **Optimization & Federated Protocol:** Zero-shot inference from frozen global checkpoints
- **Missing Telemetry Policy:** FF30
- **Random Seeds:** 42, 43, 44
- **Evaluation Protocol:** Full Q1 and March out-of-sample rollouts; macro and pooled metrics
- **Verified Metrics & Findings:** Q1: Pooled MAE 42.58 W vs Persistence 40.44 W; March: Pooled MAE 46.90 W vs Persistence 45.29 W
- **Scientific Status:** **Completed and verified**
- **Source Configurations:** `output/experiments/temporal_generalization_6h_v1/config.json`
- **Evidence & Result Artifacts:** `output/experiments/temporal_generalization_6h_v1/TEMPORAL_GENERALIZATION_REPORT.md`
- **Limitations & Methodological Bounds:** Demonstrated severe temporal drift over multi-month horizons without retraining, establishing the requirement for continual adaptation.

### Study 5: Temporal Adaptation and Continual Fine-Tuning Study
- **Internal Identifier:** `t3_temporal_adaptation`
- **Scientific Research Question:** Can periodic local client fine-tuning mitigate temporal concept drift in multi-month federated deployment?
- **Research Objective:** Evaluate weekly and monthly continual federated fine-tuning schedules on drift mitigation across 979 nodes.
- **Dataset and Period:** M100 HPC Telemetry (Q1 2022: Jan 28-Apr 01, 2022)
- **Experimental Cohort:** 979 compute nodes
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** 2-layer SequenceGRU
- **Optimization & Federated Protocol:** Continual FedAvg fine-tuning (periodic client updates)
- **Missing Telemetry Policy:** FF30
- **Random Seeds:** 42, 43, 44
- **Evaluation Protocol:** Rolling chronological test periods with periodic local weight adaptation
- **Verified Metrics & Findings:** Adapted Pooled MAE reduced drift by ~15-20% relative to frozen static baseline
- **Scientific Status:** **Completed and verified**
- **Source Configurations:** `output/experiments/temporal_adaptation_t3_v1/config.json`
- **Evidence & Result Artifacts:** `output/experiments/temporal_adaptation_t3_v1/T3_REPORT.md`
- **Limitations & Methodological Bounds:** Requires persistent communication bandwidth and edge compute availability for periodic local fine-tuning rounds.

### Study 6: Federated Forecasting Model-Improvement Study
- **Internal Identifier:** `t4_model_improvement`
- **Scientific Research Question:** Can a residual learning formulation (predicting residual delta relative to last observed power) paired with L1 loss consistently surpass naive persistence in federated forecasting?
- **Research Objective:** Systematically evaluate 13 candidate architectures, loss functions (MSE, MAE, Huber), and target formulations to break the persistence barrier.
- **Dataset and Period:** M100 HPC Telemetry (Candidate 1: Jan 01-Feb 04, 2022; Q1 extended; March extended)
- **Experimental Cohort:** 978 compute nodes on Canonical T1 Test (243,138 windows); 979 nodes on Q1 (3,609,285 windows)
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** 2-layer SequenceGRU with Residual L1 formulation: y_pred = y_last + r_pred
- **Optimization & Federated Protocol:** FedAvg (sample-weighted, 200 rounds, 1 local epoch)
- **Missing Telemetry Policy:** FF30 (channels 7 and 8 masked)
- **Random Seeds:** 42, 43, 44
- **Evaluation Protocol:** Audited multi-seed evaluation on Canonical T1, Q1, and March; per-node win rate analysis
- **Verified Metrics & Findings:** FedAvg: Pooled MAE 24.81 W, Macro MAE 25.22 W (beats persistence on 91.9% of nodes, +8.25 W improvement); Centralized: Pooled MAE 23.99 W, Macro MAE 24.37 W (94.4% win rate)
- **Scientific Status:** **Completed, audited, and frozen**
- **Source Configurations:** `output/experiments/t4_model_improvement_v1/config.json`
- **Evidence & Result Artifacts:** `output/experiments/t4_model_improvement_v1/T4_FINAL_REPORT.md, T4_INTEGRITY_AUDIT.md`
- **Limitations & Methodological Bounds:** Residual formulation relies on accurate measurement of the last observed time-step anchor; residual delta magnitude scales with volatility.

### Study 7: Environmental Telemetry Contribution Study
- **Internal Identifier:** `t5_weather_ablation`
- **Scientific Research Question:** Does outdoor environmental telemetry (ambient temperature and relative humidity) contribute predictive information for indoor compute-node IT power forecasting?
- **Research Objective:** Isolate the contribution of outdoor weather channels through a rigorous 4-condition controlled ablation on identical seeds and data splits.
- **Dataset and Period:** M100 HPC Telemetry (Canonical T1: Jan 01-Feb 04, 2022)
- **Experimental Cohort:** 978 compute nodes (243,138 matched test windows)
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** 2-layer SequenceGRU with Residual L1 formulation
- **Optimization & Federated Protocol:** FedAvg (sample-weighted, 200 rounds)
- **Missing Telemetry Policy:** FF30 (Condition A: masked; B: +temp; C: +humidity; D: +temp & humidity)
- **Random Seeds:** 42, 43, 44 across all 4 conditions
- **Evaluation Protocol:** Paired evaluation on Canonical T1 test windows against audited Condition A reference
- **Verified Metrics & Findings:** Cond A (Masked): 24.813 W; Cond B (+Temp): 24.832 W; Cond C (+Hum): 24.818 W; Cond D (+Both): 24.840 W (Differences < 0.03 W, within noise std)
- **Scientific Status:** **Completed, audited, and frozen**
- **Source Configurations:** `output/experiments/t5_weather_ablation_v1/T5_CONFIGS.json`
- **Evidence & Result Artifacts:** `output/experiments/t5_weather_ablation_v1/T5_FINAL_REPORT.md, T5_CONTROLLED_COMPARISON_AUDIT.md`
- **Limitations & Methodological Bounds:** Conducted on indoor water/air-cooled supercomputing facility with conditioned computer rooms where outdoor weather variations are physically buffered.

### Study 8: Sequence and Tree-Based Forecasting Comparison
- **Internal Identifier:** `t6_architecture_benchmark`
- **Scientific Research Question:** How does the selected SequenceGRU compare against SequenceLSTM and Centralized XGBoost in forecasting accuracy, parameter efficiency, and communication overhead?
- **Research Objective:** Benchmark recurrent deep architectures against non-neural gradient-boosted decision trees under identical features and preprocessing.
- **Dataset and Period:** M100 HPC Telemetry (Canonical T1: Jan 01-Feb 04, 2022)
- **Experimental Cohort:** 978 compute nodes (243,138 matched test windows)
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** FedAvg-GRU (49k params) vs FedAvg-LSTM (63k params) vs Centralized XGBoost (504 flat features)
- **Optimization & Federated Protocol:** FedAvg (GRU, LSTM) vs Centralized pooling (XGBoost)
- **Missing Telemetry Policy:** FF30 (channels 7 and 8 masked)
- **Random Seeds:** 42, 43, 44
- **Evaluation Protocol:** Identical Canonical T1 test windows; metric comparison across Pooled/Macro MAE/RMSE and parameter count
- **Verified Metrics & Findings:** GRU: Pooled MAE 24.81 W (49k params); LSTM: Pooled MAE 24.85 W (63k params); XGBoost: Pooled MAE 25.31 W (depth 6, 200 trees)
- **Scientific Status:** **Completed, audited, and frozen**
- **Source Configurations:** `output/experiments/t6_architecture_benchmark_v1/T6_CONFIGS.json`
- **Evidence & Result Artifacts:** `output/experiments/t6_architecture_benchmark_v1/T6_FINAL_REPORT.md, T6_INTEGRITY_AUDIT.md`
- **Limitations & Methodological Bounds:** XGBoost required tabular flattening (504 features) and centralized data pooling, losing client privacy boundaries; LSTM added 28.6% parameter overhead without accuracy gain.

### Study 9: Missing-Telemetry Robustness and Sensitivity Study
- **Internal Identifier:** `t7_missing_telemetry_robustness`
- **Scientific Research Question:** How robust is the frozen FedAvg-GRU model to inference-time telemetry missingness, and which telemetry channels are most critical for forecasting accuracy?
- **Research Objective:** Stress-test inference robustness across 0% to 30% random and contiguous missingness, and perform channel ablation across Power, CPU, Memory, and System telemetry.
- **Dataset and Period:** M100 HPC Telemetry (Canonical T1 Test: 243,138 windows, 978 nodes)
- **Experimental Cohort:** 978 compute nodes
- **Temporal Context & Horizon:** History: 6 hours (36 ten-minute steps, H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** Frozen FedAvg-GRU-T4 reference models (Seeds 42, 43, 44)
- **Optimization & Federated Protocol:** Zero-shot inference-time stress testing (no retraining)
- **Missing Telemetry Policy:** Controlled synthetic dropouts (random & contiguous blocks 3, 6, 12 steps) with forward-fill hold
- **Random Seeds:** 42, 43, 44 (13 conditions x 3 seeds = 39 evaluations)
- **Evaluation Protocol:** Full Canonical T1 test set across all missingness conditions; channel sensitivity decomposition
- **Verified Metrics & Findings:** 0%: 24.81 W; 10% random: 24.86 W (+0.21%); 20% random: 24.96 W (+0.61%); 30% contiguous: 25.33 W (+2.09%); Power channel loss: +11.8 W degradation
- **Scientific Status:** **Completed, audited, and frozen**
- **Source Configurations:** `output/experiments/t7_missing_telemetry_robustness_v1/T7_CONFIGS.json`
- **Evidence & Result Artifacts:** `output/experiments/t7_missing_telemetry_robustness_v1/T7_FINAL_REPORT.md, T7_INTEGRITY_AUDIT.md`
- **Limitations & Methodological Bounds:** Evaluated inference-time robustness of frozen models; does not retrain models under high missingness environments.

### Study 10: Common History Coverage and Telemetry Outage Audit
- **Internal Identifier:** `t1_1_common_history_coverage`
- **Scientific Research Question:** Why did earlier investigations report zero valid test windows for 24-hour history, and can a scientifically rigorous common cohort be certified across 1h, 6h, and 24h lookbacks?
- **Research Objective:** Audit telemetry completeness across time periods and missingness policies to resolve the historical coverage bottleneck and certify a shared evaluation cohort.
- **Dataset and Period:** M100 HPC Telemetry (Full Jan-Mar 2022 corpus)
- **Experimental Cohort:** 978 certified compute nodes (Node 704 excluded due to zero valid windows)
- **Temporal Context & Horizon:** History: 1h (H=6), 6h (H=36), 24h (H=144) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** N/A (Data engineering and coverage audit)
- **Optimization & Federated Protocol:** N/A
- **Missing Telemetry Policy:** no_fill vs FF10 vs FF20 vs FF30 across Candidate 1 vs late-Feb split
- **Random Seeds:** Fixed deterministic window generator
- **Evaluation Protocol:** Exhaustive combinatorial window alignment audit across 979 nodes and 144 steps
- **Verified Metrics & Findings:** Certified Candidate 1 FF30 universe: exactly 243,138 matched test windows common across 1h, 6h, and 24h histories on 978 nodes; identified late-Feb Ganglia outage as root cause of prior zero-coverage.
- **Scientific Status:** **Completed, audited, and frozen**
- **Source Configurations:** `output/experiments/t1_1_common_history_coverage_audit_v2/T1_1_CONFIG.json`
- **Evidence & Result Artifacts:** `output/experiments/t1_1_common_history_coverage_audit_v2/T1_1_FINAL_REPORT.md, T1_1_ARTIFACT_MANIFEST.md`
- **Limitations & Methodological Bounds:** Node 704 permanently excluded due to chronic sensor disconnect in January 2022.

### Study 11: Temporal Context and Forecasting-Horizon Study
- **Internal Identifier:** `t1_2_controlled_history_benchmark`
- **Scientific Research Question:** Does expanding historical input context from 1 hour to 6 hours or 24 hours improve federated next-hour node IT power forecasting, and what are the computational trade-offs?
- **Research Objective:** Conduct a strictly controlled multi-seed benchmark comparing 1h (H=6), 6h (H=36), and 24h (H=144) lookbacks on identical 243,138 test windows across 978 nodes under Candidate 1 FF30 protocol.
- **Dataset and Period:** M100 HPC Telemetry (Candidate 1: Jan 01-Feb 04, 2022)
- **Experimental Cohort:** 978 compute nodes (Primary Test: 243,138 windows; Secondary Option A: 50,392 windows across 635 nodes)
- **Temporal Context & Horizon:** History: 1h (H=6), 6h (H=36), 24h (H=144) | Forecast Horizon: 1 hour ahead mean IT power (Residual-anchor formulation)
- **Model Architecture:** 2-layer SequenceGRU (49k params)
- **Optimization & Federated Protocol:** FedAvg (sample-weighted, 200 rounds, 9 full fits across 3 seeds)
- **Missing Telemetry Policy:** FF30 (channels 7 and 8 masked)
- **Random Seeds:** 42, 43, 44 (9 full sequential fits)
- **Evaluation Protocol:** Primary test on 243,138 matched windows; secondary Option A test on 50,392 windows; runtime and convergence profiling
- **Verified Metrics & Findings:** 1h: 33.90 W Pooled MAE (1233s); 6h: 33.99 W Pooled MAE (1357s); 24h: 33.99 W Pooled MAE (1584s). Persistence: 33.07 W. Lookback invariance verified (|delta| < 0.09 W). 24h incurs +28.5% training runtime penalty.
- **Scientific Status:** **Completed, audited, and frozen**
- **Source Configurations:** `output/experiments/t1_2_controlled_history_benchmark_v1/T1_2_CONFIG.json`
- **Evidence & Result Artifacts:** `output/experiments/t1_2_controlled_history_benchmark_v1/T1_2_FINAL_REPORT.md, T1_2_INTEGRITY_AUDIT.md`
- **Limitations & Methodological Bounds:** Direct comparison under Candidate 1 protocol; shows lookback invariance for next-hour forecast horizon; does not test multi-day forecast horizons.

### Study 12: Energy-Aware Operational Decision Study
- **Internal Identifier:** `decision_support_framework`
- **Scientific Research Question:** How can decentralized node-level power forecasts be integrated into cluster-level job scheduling and facility cooling decision-support systems?
- **Research Objective:** Formulate the mathematical coupling between next-hour forecasted node power demands, cooling chiller setpoints, and workload re-balancing constraints.
- **Dataset and Period:** Conceptual formulation grounded in M100 facility topology and base-paper cyber-physical models
- **Experimental Cohort:** Cluster and facility scale (979 nodes aggregated into rack/hall cooling zones)
- **Temporal Context & Horizon:** History: 6 hours (H=36) | Forecast Horizon: 1 hour ahead mean IT power
- **Model Architecture:** FedAvg-GRU forecasting module coupled to constrained optimization solver
- **Optimization & Federated Protocol:** Federated forecast ingestion with centralized or distributed dispatch
- **Missing Telemetry Policy:** Robust prediction under FF30
- **Random Seeds:** N/A (Mathematical formulation and architectural framework)
- **Evaluation Protocol:** Formulation of objective functions, operational constraints, and action-response relationships
- **Verified Metrics & Findings:** Theoretical formulation completed; simulation integration remains future work
- **Scientific Status:** **Planned / Conceptual Framework Designed (No deployed simulator)**
- **Source Configurations:** `docs/Architecture.md, docs/Features.md`
- **Evidence & Result Artifacts:** `output/final_endsem_report_v2/FINAL_PROJECT_REPORT.md`
- **Limitations & Methodological Bounds:** No deployed physical cooling simulator or live facility control loop is executed in this repository; no empirical energy savings are claimed.

---

## 4. Standardized Research Terminology Glossary

To ensure absolute consistency across all thesis chapters, defense slides, oral scripts, and technical documentation, the following standardized terminology must be utilized strictly:

### 4.1 System & Distributed Topology Entities
- **Compute Node:** A physical computing server inside an HPC cluster or enterprise facility, equipped with CPUs, GPUs, DRAM, and onboard IPMI/Ganglia sensors. In this project, telemetry originates from 979 dual-socket IBM Power9 compute nodes at the CINECA Marconi100 supercomputing center.
- **Simulated FL Client:** A logical client entity in the simulated federated learning harness representing exactly one compute node. Each client maintains its own isolated historical telemetry partition. *Critical distinction:* The 979 clients represent 979 compute nodes within a single supercomputing facility, NOT 979 separate geographic data centers.
- **Federated Communication Round:** A single global synchronization step where the central coordinator broadcasts the current global model weights to participating clients, clients train locally on private data, and the coordinator aggregates client parameter updates into a new global model.
- **Local Training:** The optimization step performed autonomously at each client using its own local historical data for a specified number of local epochs (=1$) or batches, producing a local model weight update without transferring raw sensor data.
- **Sample-Weighted FedAvg (Federated Averaging):** The canonical federated optimization algorithm where the global model weights are updated as the weighted average of client weights, proportional to each client's valid training window count: $\theta_{g}^{r+1} = \sum_{k=1}^K \frac{n_k}{\sum_j n_j} \theta_k^{r+1}$.

### 4.2 Machine Learning & Architecture Concepts
- **Gated Recurrent Unit (GRU):** A gating recurrent neural network architecture comprising reset ($) and update ($) gates that captures temporal sequential dependencies while mitigating vanishing gradients. The validated reference architecture employs a 2-layer SequenceGRU with 64 hidden units (49,217 parameters).
- **Historical Input Window ($):** The length of the chronological telemetry sequence provided to the model. Discretized into 10-minute bins: 1 hour (=6$), 6 hours (=36$), and 24 hours (=144$).
- **Forecasting Horizon ($\tau$):** The future time interval over which power is predicted. In this project, $\tau = 1\text{ hour}$ ahead, defined specifically as the 1-hour ahead mean node IT power (in Watts).
- **Residual Learning Formulation:** An architectural formulation where the model predicts the incremental deviation ($\hat{r}$) from the last observed power reading rather than absolute power directly: $\hat{y}_{t+1} = y_{t,\text{last}} + \hat{r}$. Paired with L1 loss, this formulation overcomes the persistence barrier and achieves a 91.9% per-node win rate.
- **Lookback Invariance:** The empirical discovery that extending historical input context from 1 hour (=6$) to 6 hours (=36$) or 24 hours (=144$) produces no statistically meaningful improvement in next-hour mean IT power forecasting ($|\Delta \text{MAE}| < 0.09\text{ W}$), while incurring up to +28.5% training runtime penalty and 4x memory overhead.

### 4.3 Data Quality & Telemetry Concepts
- **Missing Telemetry:** Sensor intervals where hardware monitoring daemons fail to report data due to collector timeouts, network dropouts, or node reboots.
- **Forward-Fill Imputation (FF10, FF20, FF30):** Imputation policies where missing readings are held constant from the last observed valid sample for up to 10, 20, or 30 minutes. Gaps exceeding the threshold invalidate the window.
- **Naive Persistence Baseline ([t-1]$):** The canonical zero-cost benchmark that predicts the next-hour mean IT power to be equal to the last observed power sample. In high-inertia thermal and workload systems, persistence provides a formidable baseline (3.07\text{ W}$ pooled MAE).

### 4.4 Evaluation Metrics
- **Mean Absolute Error (MAE):** The primary metric measuring average magnitude of prediction errors, reported strictly in physical Watts ($\text{W}$): $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$.
- **Root Mean Squared Error (RMSE):** The metric penalizing large outlier errors, reported strictly in physical Watts ($\text{W}$): $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$.
- **Pooled Metrics:** Metrics computed by pooling all prediction-target pairs across all nodes and all test timestamps into a single aggregate distribution.
- **Macro Metrics:** Metrics computed by first calculating the MAE/RMSE independently for each individual node, and then averaging equally across all compute nodes.
- **Energy-Aware Decision Support:** Downstream optimization framework coupling node power forecasts to facility cooling control and job scheduling. *Operational boundary:* Implemented as a conceptual mathematical formulation; closed-loop physical simulator integration remains future work.
