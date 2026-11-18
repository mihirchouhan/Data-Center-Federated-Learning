# Comprehensive Viva Defense Examination Questions and Evidence-Based Answers

**Project Title:** Federated Forecasting and Energy-Aware Decision Support for Distributed Data Centers under Heterogeneous and Missing Telemetry  
**Student:** Mihir Chouhan (Roll No: 25CSE2014) · **Supervisor:** Dr. Pravati Swain  
**Department of Computer Science and Engineering, National Institute of Technology Goa**  
**Presentation Date:** 19 December 2026 · **Document Status:** Authoritative Viva Guide  

---

## Part I: System Architecture, Data, and Client Topology

### 1. Why use Federated Learning instead of centralized training in a single supercomputing facility?
**Answer:** While all 979 compute nodes in this study reside within the CINECA Marconi100 facility, centralized collection of high-frequency telemetry (10-second intervals across 14 channels) creates severe ingestion bottlenecks, bandwidth saturation on management networks, and single-point-of-failure storage overhead. Federated Learning (FL) shifts optimization directly to edge compute nodes: each node processes its own telemetry locally, computing gradient updates without streaming raw data. Furthermore, in multi-tenant or hybrid cloud deployments, FL preserves workload confidentiality across distinct administrative partitions.

### 2. Why select Gated Recurrent Units (GRU) rather than Transformers or LSTMs?
**Answer:** Gated Recurrent Units strike the optimal Pareto frontier between temporal sequence modeling accuracy, memory footprint, and communication payload. As proven in our *Sequence and Tree-Based Forecasting Comparison (Study 8)*, SequenceGRU (49,217 parameters) matches SequenceLSTM (63,297 parameters) within $0.035	ext{ W}$ ($24.81	ext{ W}$ vs $24.85	ext{ W}$ MAE) while saving **22.2% trainable parameters** and eliminating the cell-state vector from communication rounds. Furthermore, compared to self-attention Transformers, GRU incurs $O(H)$ linear time complexity rather than $O(H^2)$ quadratic complexity, which is crucial when scaling across 978 edge clients.

### 3. What does a "simulated FL client" mean, and why are 979 clients not 979 independent data centers?
**Answer:** In our experimental architecture, a simulated FL client represents exactly one physical compute node (an IBM Power9 server) within the Marconi100 Tier-0 facility. Each client maintains its own isolated historical telemetry partition $\mathcal{D}_k$ and executes local Adam optimization. The coordinator samples 98 clients per round and performs sample-weighted aggregation. It is vital to emphasize that these 979 clients belong to **one physical facility**, not 979 separate geographic data centers. Claims regarding multi-site wide-area network latency or cross-data-center transfers are explicitly avoided.

### 4. What exact physical quantity does the network forecast?
**Answer:** The primary prediction target is the **next-hour mean node IT power demand** ($ar{P}_{t+1}$, in Watts):
$$ar{P}_{t+1} = rac{1}{6} \sum_{j=0}^5 p_n([t + 10j, t + 10(j+1))).$$
It is the temporal arithmetic mean across the subsequent six 10-minute bins. It is NOT instantaneous power at $t+60	ext{ minutes}$, and NOT cumulative energy in Watt-hours. Facility cooling infrastructure (chillers, pumps, CRACs) operates on thermal time constants where the hourly average heat dissipation governs chiller COP and thermal dynamics.

### 5. How are sensor missingness and collector outages handled?
**Answer:** We implement the `FF30` causal forward-fill policy: missing sensor readings in the 10-minute grid are held forward from the last observed finite sample for up to 30 minutes (3 time steps). Gaps exceeding 30 minutes invalidate the window. Target labels are **strictly observed** and never imputed. Standardized training moments are computed exclusively on the training split, completely eliminating data leakage.

### 6. What is the role of the learned categorical node embedding?
**Answer:** Each compute node is assigned an 8-dimensional learned embedding vector via `nn.Embedding(979, 8)`. Concatenated with the 13 continuous channels, this provides a 21-dimensional input to the recurrent layers. The embedding enables the model to capture static node-specific hardware biases and thermal micro-climates within server racks while sharing the temporal recurrent backbone across the cluster.

---

## Part II: Core Empirical Discoveries and Benchmarks

### 7. What is the Lookback Invariance Law discovered in Study 11?
**Answer:** In our *Temporal Context and Forecasting-Horizon Study (Study 11)*, we conducted a multi-seed benchmark comparing 1-hour ($H=6$), 6-hour ($H=36$), and 24-hour ($H=144$) historical contexts on identical 243,138 test windows across 978 nodes. All three histories achieved virtually identical Pooled MAE:
- 1-hour: $33.9032 \pm 0.5573	ext{ W}$ ($1232.8	ext{ s}$ training time)
- 6-hour: $33.9868 \pm 0.6312	ext{ W}$ ($1357.1	ext{ s}$ training time)
- 24-hour: $33.9873 \pm 0.6321	ext{ W}$ ($1584.4	ext{ s}$ training time)
The error delta is under $0.084	ext{ W}$ (well within the $\pm 0.63	ext{ W}$ seed standard deviation). However, the 24-hour context imposes a **$+28.5\%$ training wall-clock penalty** and quadruples recurrent tensor memory. This demonstrates that recent 1-hour workload momentum heavily dominates long-range telemetry for next-hour predictions.

### 8. How did the research overcome the persistence barrier in Study 6?
**Answer:** In thermal and power systems, naive persistence ($P[t-1]$, predicting the last observed power reading) achieves an aggressive baseline ($33.07	ext{ W}$ pooled MAE). Direct power regression under MSE loss focuses on absolute power magnitudes and fails to consistently beat persistence.  
In our *Federated Forecasting Model-Improvement Study (Study 6)*, we reformulated the learning objective as **residual delta forecasting** ($\hat{y}_{t+1} = p_{t,	ext{last}} + \hat{r}$) paired with **L1 loss** (MAE). This formulation anchors the prediction to current power levels and eliminates the high-variance DC level offset. As a result, FedAvg-GRU achieved a Pooled MAE of **$24.81 \pm 0.04	ext{ W}$** ($25.22 \pm 0.03	ext{ W}$ Macro MAE), outperforming persistence on **898.3 / 978 compute nodes (91.85% win rate)** with an improvement of **$+8.25	ext{ W}$ (+24.96%)**.

### 9. Does outdoor weather telemetry contribute to node IT power forecasting?
**Answer:** No. In our *Environmental Telemetry Contribution Study (Study 7)*, we conducted a 4-condition controlled ablation isolating outdoor temperature and relative humidity across identical splits and seeds:
- Condition A (Weather Masked): $24.8133 \pm 0.0358	ext{ W}$
- Condition B (+ Temperature): $24.8322 \pm 0.0520	ext{ W}$
- Condition C (+ Humidity): $24.8184 \pm 0.0360	ext{ W}$
- Condition D (+ Both): $24.8398 \pm 0.0504	ext{ W}$
The variance across conditions ($< 0.03	ext{ W}$) is within random seed noise. Indoor compute servers operate inside climate-controlled halls buffered by CRAC infrastructure. Masking outdoor weather channels saves edge network bandwidth without accuracy loss.

### 10. How resilient is the forecasting model to telemetry missingness?
**Answer:** In our *Missing-Telemetry Robustness Study (Study 9)*, the frozen residual FedAvg-GRU model was stress-tested across 13 synthetic missingness conditions:
- 0% Missing (Reference): $24.81	ext{ W}$
- 5% Random Missing: $25.19	ext{ W}$ ($+0.38	ext{ W}$ degradation)
- 10% Random Missing: $25.59	ext{ W}$ ($+0.78	ext{ W}$ degradation)
- 20% Random Missing: $26.52	ext{ W}$ ($+1.71	ext{ W}$ degradation)
- 30% Random Missing: $28.07	ext{ W}$ ($+3.26	ext{ W}$ degradation)
- Power Channel Complete Outage: $124.97	ext{ W}$ ($+100.16	ext{ W}$ degradation)
The model retains over $90\%$ node win rates up to $20\%$ random missingness. Channel ablations prove that Power telemetry is the single mission-critical channel; CPU, Memory, and Temperature outages add under $0.21	ext{ W}$ degradation.

---

## Part III: Base-Paper Comparison and Operational Boundaries

### 11. What is the fundamental difference between your work and the mandatory base paper (Liu et al.)?
**Answer:** The base paper (*Adaptive Federated Learning to Optimize Integrated Flows in Cyber–Physical Data Centers*, arXiv:2511.00623v3) implements an adaptive FL-to-optimization framework across 5 synthetic data center agents, using static MLP ensembles to approximate the decisions of an offline mixed-integer quadratic program (MIQP).  
Our work performs pure empirical time-series forecasting across 979 real compute nodes at CINECA M100. We evaluate recurrent sequence models (GRU/LSTM) on real telemetry under explicit missingness policies (`FF30`). We do not claim reproduction of their adaptive momentum acceptance, McCormick envelopes, or physical optimization dispatch solver.

### 12. Why are energy savings not claimed in this thesis?
**Answer:** In strict accordance with academic research integrity, a forecast error reduction cannot be equated with operational energy savings. Converting forecasts into energy reduction requires:
1. Actuator action variables (chiller chilled-water supply temperatures, server frequency governors).
2. A validated physical action-response simulator (such as EnergyPlus or CFD modeling).
3. Operational constraints (server inlet temperature $\le 25^\circ	ext{C}$, SLURM queue SLA deadlines).
While our *Energy-Aware Operational Decision Study (Study 12)* establishes the complete mathematical optimization formulation, physical simulator execution remains future work. Claiming energy savings without closed-loop deployment is scientifically unsupportable.

### 13. What is the historical strict preflight status, and why is it preserved?
**Answer:** The repository rules dictate that historical strict preflight remains FAILED due to past artifact cache mismatches in older v5 exploratory runs. Rather than rewriting Git history or falsifying baseline hashes, we preserve this status transparently. All modern studies (T1 through T7) implement fresh, cryptographic SHA-256 manifests, locked seeds, and independent integrity audits that achieve a **100% PASS rate**.

---

## Part IV: Defense Summary of Primary Contributions

### 14. What are the five primary contributions of this M.Tech research?
1. **The Lookback Invariance Law:** Proved that 1-hour historical context is sufficient for next-hour mean IT power forecasting, disproving the assumption that multi-day lookbacks improve accuracy and saving 28.5% training time.
2. **Solving the Persistence Barrier:** Developed the residual delta formulation ($\hat{y} = p_{	ext{last}} + \hat{r}$) with L1 loss, achieving $24.81	ext{ W}$ Pooled MAE and outperforming persistence across 91.9% of compute nodes (+24.96% gain).
3. **Decoupling Outdoor Environmental Telemetry:** Proved through controlled ablation that outdoor weather signals provide zero predictive gain ($<0.03	ext{ W}$) for indoor server power forecasting.
4. **Architectural Pareto Efficiency:** Validated that a lightweight 49k-parameter SequenceGRU matches larger LSTMs (63k) and outperforms tree-based models (XGBoost) while maintaining strict decentralized privacy boundaries.
5. **Telemetry Missingness Resilience:** Characterized performance degradation under random and contiguous dropouts, demonstrating strong robustness up to 20% loss and identifying power telemetry as the single critical sensor.
