# Master End-Semester Research Deliverables and Submission Package

**Project Title:** Federated Forecasting and Energy-Aware Decision Support for Distributed Data Centers under Heterogeneous and Missing Telemetry  
**Student:** Mihir Chouhan · **Roll Number:** 25CSE2014  
**Department:** Computer Science and Engineering, National Institute of Technology Goa  
**Supervisor:** Dr. Pravati Swain  
**Presentation Date:** 19 December 2026  
**Document Revision:** Standardized Academic Research Edition (November 2026)  
**Package Directory:** `/media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission`  

---

## 1. Directory Organization and Package Contents

This folder consolidates all final, verified, and standardized academic deliverables for the M.Tech major project into an easily identifiable structure:

```
output/endsem_final_submission/
├── README.md                              # This master navigation and executive guide
│
├── 01_FINAL_PROJECT_REPORT/              # Final Thesis Report and Research Monograph
│   ├── FINAL_PROJECT_REPORT.md           # The authoritative, standardized project report
│   ├── BASE_PAPER_DETAILED_ANALYSIS.md   # Mathematical analysis of Liu et al. (arXiv:2511.00623v3)
│   ├── BASE_PAPER_VS_OUR_RESEARCH_COMPARISON.md # 7-table comparative analysis
│   ├── BIBLIOGRAPHY.md                   # Complete primary references and academic citations
│   └── FIGURES/                          # All 41 high-resolution publication figures & plots
│
├── 02_PRESENTATION_AND_DEFENSE/          # Slide Presentation and Defense Package
│   ├── FINAL_ENDSEM_PRESENTATION.pptx    # 24-slide OpenXML presentation with embedded speaker notes
│   ├── FINAL_ENDSEM_PRESENTATION.pdf     # 24-page high-resolution vector/image PDF presentation
│   ├── PRESENTATION_SCRIPT.md            # Word-for-word oral defense speaker script (19 min 45 sec)
│   ├── VIVA_QUESTIONS_AND_ANSWERS.md     # 14 comprehensive viva questions and evidence-based answers
│   ├── PRESENTATION_MAP.md               # Master slide index and timing allocation
│   ├── slide_content.json                # Machine-readable slide metadata
│   └── SLIDES/                           # All 24 rendered 16:9 PNG slide images (slide_01.png to slide_24.png)
│
├── 03_RESEARCH_REGISTRY_AND_GLOSSARY/    # Master Experiment Registry & Terminology Glossary
│   ├── EXPERIMENT_REGISTRY.md            # Master registry with 12 study profiles and terminology glossary
│   ├── EXPERIMENT_REGISTRY.csv           # Machine-readable database of all 12 standardized studies
│   ├── PRESENTATION_METRICS.csv          # Comprehensive metric database (214 reconciled rows)
│   └── EXPERIMENT_STATUS_REGISTER.csv    # Status register of all 190 experimental fits and runs
│
├── 04_CODEBASE_TECHNICAL_EXPLANATION/    # Deep Technical Architecture Monograph
│   └── CODEBASE_EXPLANATION.md           # Comprehensive architectural guide across 15 pipeline stages
│
└── 05_AUDITED_RESULTS_SUMMARIES/         # Cryptographically Audited Experiment Summaries
    ├── T1_2_RESULTS_SUMMARY.csv          # Lookback benchmark summary (1h vs 6h vs 24h)
    ├── T4_FINAL_RESULTS_SUMMARY.csv      # Residual learning summary (91.85% win rate)
    ├── T5_RESULTS_SUMMARY.csv            # Weather ablation summary (< 0.03 W delta)
    ├── T6_RESULTS_SUMMARY.csv            # Architecture comparison summary (GRU vs LSTM vs XGB)
    └── T7_RESULTS_SUMMARY.csv            # Missing telemetry stress testing summary
```

---

## 2. Executive Summary of Primary Research Contributions

1. **The Lookback Invariance Law (Study 11):**
   - Proved empirically in a controlled 9-fit multi-seed benchmark across 1h ($H=6$), 6h ($H=36$), and 24h ($H=144$) histories that expanding context beyond 1 hour yields **zero statistically significant gain** ($|\Delta 	ext{MAE}| < 0.084	ext{ W}$, well within the $\pm 0.63	ext{ W}$ seed variance).
   - In contrast, the 24-hour lookback incurs a **$+28.5\%$ training wall-clock penalty** ($1584	ext{ s}$ vs $1233	ext{ s}$) and quadruples recurrent tensor memory. Recent 1-hour workload momentum heavily dominates long-range telemetry for next-hour predictions.

2. **Solving the Persistence Barrier via Residual L1 Learning (Study 6):**
   - Naive persistence ($P[t-1]$) achieves an aggressive baseline ($33.07	ext{ W}$ Pooled MAE) in thermal and power systems governed by physical inertia. Direct power regression under MSE loss focuses on absolute power levels and fails to consistently beat persistence.
   - By reformulating the learning task as **residual delta forecasting** ($\hat{y}_{t+1} = p_{t,	ext{last}} + \hat{r}$) with **L1 loss**, FedAvg-GRU achieves **$24.81 \pm 0.04	ext{ W}$ Pooled MAE** ($25.22 \pm 0.03	ext{ W}$ Macro MAE), decisively outperforming persistence across **898.3 / 978 compute nodes (91.85% win rate)** with a **$+8.25	ext{ W}$ (+24.96%)** improvement.

3. **Environmental Telemetry Decoupling (Study 7):**
   - A 4-condition controlled ablation proved that outdoor temperature and relative humidity contribute zero predictive signal ($< 0.03	ext{ W}$ difference between masked and enabled models).
   - Outdoor meteorological fluctuations are completely buffered by computer hall CRAC units. Masking outdoor weather channels saves edge network bandwidth without accuracy loss.

4. **Architectural Pareto Efficiency (Study 8):**
   - 2-layer SequenceGRU (49,217 parameters) matches SequenceLSTM (63,297 parameters) within $0.035	ext{ W}$ ($24.81	ext{ W}$ vs $24.85	ext{ W}$ MAE) while requiring **22.2% fewer parameters** and eliminating cell-state communication overhead.
   - Both recurrent neural architectures decisively outperform centralized tabular XGBoost trees ($25.31	ext{ W}$ MAE).

5. **Telemetry Missingness Resilience (Study 9):**
   - Inference-time stress testing demonstrates strong resilience up to 20% random missing telemetry ($< 0.6\%$ degradation).
   - Power telemetry is proven to be the single mission-critical input channel ($+100.16	ext{ W}$ degradation under complete blackout); CPU, memory, and temperature blackouts add under $0.21	ext{ W}$ error.

6. **Base Paper Alignment and Operational Decision Boundaries (Study 12):**
   - Rigorously compared against the base paper (*Adaptive Federated Learning to Optimize Integrated Flows in Cyber–Physical Data Centers*, Liu et al., arXiv:2511.00623v3).
   - We maintain strict scientific integrity: our work performs pure empirical time-series forecasting across 979 real compute nodes. Downstream decision support is formulated mathematically; no live facility control loop or physical simulator was executed, and no unverified energy savings are claimed.

---

## 3. Key Document File Links

- **Final Thesis Report:** [01_FINAL_PROJECT_REPORT/FINAL_PROJECT_REPORT.md](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/01_FINAL_PROJECT_REPORT/FINAL_PROJECT_REPORT.md)
- **Base-Paper Analysis:** [01_FINAL_PROJECT_REPORT/BASE_PAPER_DETAILED_ANALYSIS.md](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/01_FINAL_PROJECT_REPORT/BASE_PAPER_DETAILED_ANALYSIS.md)
- **PowerPoint Presentation:** [02_PRESENTATION_AND_DEFENSE/FINAL_ENDSEM_PRESENTATION.pptx](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/02_PRESENTATION_AND_DEFENSE/FINAL_ENDSEM_PRESENTATION.pptx)
- **PDF Presentation:** [02_PRESENTATION_AND_DEFENSE/FINAL_ENDSEM_PRESENTATION.pdf](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/02_PRESENTATION_AND_DEFENSE/FINAL_ENDSEM_PRESENTATION.pdf)
- **Oral Defense Script:** [02_PRESENTATION_AND_DEFENSE/PRESENTATION_SCRIPT.md](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/02_PRESENTATION_AND_DEFENSE/PRESENTATION_SCRIPT.md)
- **Viva Defense Q&A:** [02_PRESENTATION_AND_DEFENSE/VIVA_QUESTIONS_AND_ANSWERS.md](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/02_PRESENTATION_AND_DEFENSE/VIVA_QUESTIONS_AND_ANSWERS.md)
- **Master Experiment Registry:** [03_RESEARCH_REGISTRY_AND_GLOSSARY/EXPERIMENT_REGISTRY.md](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/03_RESEARCH_REGISTRY_AND_GLOSSARY/EXPERIMENT_REGISTRY.md)
- **Codebase Technical Monograph:** [04_CODEBASE_TECHNICAL_EXPLANATION/CODEBASE_EXPLANATION.md](file:///media/csedept/cse2014/Project/m100_training_template/output/endsem_final_submission/04_CODEBASE_TECHNICAL_EXPLANATION/CODEBASE_EXPLANATION.md)
