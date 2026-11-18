# Detailed Analysis of the Mandatory Base Paper

Junhong Liu, Lanxin Du, Yujia Li, Rong-Peng Liu, Yunfeng Li, Fei Teng and Francis Yunhe Hou, *Adaptive Federated Learning to Optimize Integrated Flows in Cyber–Physical Data Centers*, arXiv:2511.00623v3, 14 July 2026. [Official PDF](https://arxiv.org/pdf/2511.00623v3).

## Document identity and reading boundary

The attachment named `Optimized integration Flows in Cyber–Physical Data Centers(4).pdf` was not found in the accessible attachments, Downloads, Desktop or project-file searches. Consequently, this analysis uses the actual official v3 PDF, downloaded to an isolated temporary directory and read page by page. Its title, authors and version match the requested reference; byte identity with the unavailable uploaded copy is **Not verified from available artifacts**. The official PDF has 14 pages, 1,069,514 bytes and SHA-256 `c081a18f0a7ac2ea920698904734f667981fe2f707fbb5d1d869efe8ec4d8aef`. Page references below are PDF/printed page numbers, starting at 1. Extracted text was cross-checked against its explicit page boundaries because embedded figure text introduces extra form-feed characters.

The comparative scope covers learning, physical optimization and evaluation. Privacy, key management, cryptographic protocols, cybersecurity and security mechanisms are **Out of Scope for this comparison**. Their implementation details and associated experiments are not compared with this project. The paper's reported overall runtime is retained as a complete reported workflow measurement without attributing any component speedup to this project.

## Problem and physical formulation

Section II-A, pp. 3–4, formulates day-ahead coordination of electricity, heat and computational workload. A utility coordinates modeled data centers, generators, storage and electricity markets; Figure 1, p. 4, depicts this physical organization. This is a decision problem, rather than next-hour sensor forecasting.

Equation (1a) minimizes a sum over operating periods of net imported/exported electricity expenditure, service-delay penalties, generator costs and battery degradation, less ancillary-service revenue. Schematically, with the paper's symbols abbreviated,

\[
J=\sum_t\Delta t\left[P_t^{imp}\lambda_t^{imp}-P_t^{exp}\lambda_t^{exp}
+\sum_i c_i^{pen}\delta_{i,t}+\sum_g(\alpha_gp_{g,t}+\beta_gp_{g,t}^2)
-\sum_sP_{s,t}^{dch}\lambda_t^{reg}+\sum_sc_s^{deg}(P_{s,t}^{ch}+P_{s,t}^{dch})\right].
\]

The constraints describe workload queues, server counts and capacity, processing efficiency, server electricity, temperature dynamics, cooling, service-delay penalties, generator bounds/ramping, battery state and power balance. For example, queue evolution in Eq. (1c) balances arrivals against effective processing; Eq. (1j) links delay to an SLA target; Eqs. (1p)–(1u) couple thermal evolution and cooling; Eq. (1ad) balances power. Server count is an integer decision. Trading and storage modes include binary decisions. These structures explain why the original model is computationally harder than a regression forward pass.

The effective-processing product is bilinear. The authors introduce McCormick-envelope inequalities in Eqs. (2a)–(2c), p. 4, and describe the resulting problem P2 in Eq. (3) as MIQP. Envelope reformulation should not be casually equated with exact recovery of every original nonconvex feasible point; a faithful reproduction would need to examine the bounds, auxiliary constraints and code. This project implements neither P1 nor P2.

## Learned mapping and system architecture

Section III-A, pp. 4–6, learns an environment-to-decision mapping from historical optimal operating decisions. Environmental inputs include import/export/regulation prices, queue state and workload arrivals. Learned outputs include operating servers, processing choices, temperature, electrical and cooling quantities, and service-delay variables. These are optimizer-derived decisions/states, not the same labels as observed IPMI next-hour means.

Equations (4a)–(4d), pp. 4–5, define three hidden transformations followed by an output transformation. Section IV-A, p. 9, specifies hidden widths `[256,1024,256]`, ReLU and dropout 0.15 after each hidden layer. Calling this a four-layer MLP counts the output affine layer. It is not a recurrent network and has no demonstrated equivalence to the project's GRU histories.

Each agent has an ensemble. Shared blocks are averaged within its ensemble before federation; only an indicated subset of blocks is shared, with the first three blocks given as an example in Eq. (5), p. 5. Ensemble mean and spread appear in Eqs. (12a)–(12b). Figure 5, p. 9, labels three ensemble members per each of five modeled data centers; the formulation itself uses a general ensemble size N. Figures 2 and 3, p. 6, describe offline/online learning organization. The figure's phase naming should not be used to infer that the repository implements the paper's online operational system.

## Adaptive update acceptance and local learning

The aggregate in Eqs. (9)–(10), p. 5, uses the paper's round-specific weights. It must not be relabeled as the project's accepted-window-count weighting. Algorithm 1, p. 6, combines independent learning and conditional participation. Its composite loss is `0.85 MSE + 0.15 Huber + λ||θ||²`. It requires at least three willing agents for a federated aggregation.

Equations (11a)–(11d), p. 5, define recent-history measures: relative median-loss improvement I; independent-minus-federated loss slope T; and a clipped relative volatility score S. A weighted score C combines indicators for favorable I and T with S. Algorithm 1 accepts when `(I>0.1 or T>0.01) and C>0.8`; weights are 0.4, 0.3 and 0.3, history length 20, momentum coefficient 0.9 and epsilon 10⁻⁶. Low acceptance below 0.1 after five rounds triggers a five-round opt-out. Nonparticipants continue independent local training. These are adaptive local acceptance/participation rules, not ordinary unconditional full-model FedAvg.

Equation (13), p. 6, gives decaying mixing `α_base=0.1×0.95^r`, `α=α_base C`, a momentum recurrence and the local parameter update. A reproduction caution is warranted: as printed, the recurrence uses `(θ_local−θ_aggregate)` followed by a plus update. This apparent direction convention should be reconciled with the authors' implementation before copying it. This review records the printed rule and does not independently certify its convergence proof or silently correct its sign. No corresponding momentum-acceptance algorithm was implemented in this reporting task.

Section IV-A specifies 1,500 independent warm-up steps, batch size 512, then 25 global rounds with 800 local steps per round. A learning step is not an epoch, and 25 rounds here is not a smaller but otherwise equivalent version of this project's 200 rounds. Dataset sizes, local workloads, objectives and participation rules differ.

## Learning-to-optimization stage

Section III-B and Eq. (14), p. 7, use inferred local decisions to simplify P2 into P3. The remaining dispatch variables concern generators, storage and grid import/export. Algorithm 2's relevant physical sequence is local prediction followed by the reduced dispatch solve. Thus the contribution joins supervised decision approximation to constrained system dispatch; a learned model alone is not the full method.

The complexity discussion in Section III-E, pp. 8–9, attributes the main simplification to removal of data-center server-count integers and associated variables from the remaining optimization. Local MLP inference depends on ensemble size, horizon and layer widths; the residual optimization still has combinatorial structure. No corresponding optimization variables, constraints or solver benchmark have been evaluated in the forecasting repository. Facility cooling/PUE channels discovered in its upstream archive do not supply the missing action-response formulation automatically.

## Dataset, simulation and evaluation

Section IV-A, p. 9, identifies market signals, load profiles and operational data for modeled centers/storage/generators and points to reference [29] for data. It reports Python 3.9, 15 Xeon Platinum 8474C vCPUs, RTX 4090D 24 GB and Gurobi 11.0. Five data-center agents are represented in Table II and Figure 5. These modeled sites cannot be equated to 979 real compute nodes from one facility.

A precise chronological train/validation/test sample count, original sensor-missingness policy and an independently validated real-site generalization claim are **Not verified from available artifacts** in this PDF review. The paper mentions augmentation and validation-based acceptance; it does not provide this repository's UTC ten-minute, 300-second hold and 50% coverage contract. Absence of an explicitly verified detail here is not evidence that the authors' source code lacks it; that separate codebase was not reproduced.

Equation (15a), p. 9, averages per-decision-dimension RMSE normalized by that dimension's standard deviation plus 10⁻⁸. Equation (15b) measures relative NRMSE improvement against independent learning. Equation (15c) gives R². These answer different questions from watt MAE, watt RMSE or macro-node errors. The paper's benefit percentage cannot be substituted for this project's percentage improvement over persistence.

## Verified numerical records

Table II, p. 9, reports R² across all decision variables for each center:

|Method|DC1|DC2|DC3|DC4|DC5|
|---|---:|---:|---:|---:|---:|
|Independent|0.920|0.630|0.959|0.941|0.916|
|Default FL|0.591|0.807|0.797|0.642|0.935|
|Adaptive FL|0.951|0.988|0.988|0.984|0.988|

Figure 5 reports approximately 15–30% NRMSE benefit over independent learning and unequal acceptance rates. Figure 6, p. 10, shows individual decision-variable examples for DC5; its variable-specific R² values are not Table II's combined values.

Table III, p. 10, reports aggregate cost $70,391.63 for reference centralized optimization, $70,304.49 for independent-learning-assisted optimization, $70,344.01 for default FL and $70,335.77 for adaptive FL. The displayed absolute relative errors are 0.12%, 0.07% and 0.08%. Recalculation from the displayed costs gives approximately 0.1238%, 0.0677% and 0.0794%, consistent with rounding. Smaller modeled cost than the reference must not automatically be called a real saving: it is reported as deviation from an optimization reference. Default FL's aggregate deviation is smaller than adaptive FL's; the authors emphasize component-wise balance rather than universal dominance. The service-penalty component has a small reference denominator and much larger relative deviations, another reason to inspect components.

Section IV-C and Figure 8, p. 10, describe ten simulation instances. The reported overall times are approximately 0.61 s and 5.30 s; `5.30/0.61=8.6885`, consistent with 8.69×. These are operating-workflow times under the paper's hardware/setup. They are not FL training durations, network throughput measurements or a speedup reproduced by this project.

## Limitations and implications for this M.Tech work

The useful transferable idea is to distinguish prediction from constrained operational consequences. However, the paper's labels are optimizer-derived decisions, whereas the project's labels are observed future telemetry. Site-level energy-flow equations and SLA penalties require measurements/assumptions at the matching physical scale. A node-temperature forecast cannot validate a facility cooling-control equation on its own.

The project offers a different empirical opportunity: coverage-aware temporal forecasting, explicit missing-input policies, observed-target preservation, node heterogeneity and same-window persistence/control comparisons. Its measured limitations include a one-date v5 test panel, B1's mixed persistence comparison and unresolved newer protocol/documentation differences. These limitations support further controlled work; they do not demonstrate that either system is universally superior.

Before operational adaptation, the project would need independently specified actions, causal response, service constraints, accounting boundaries, cost signals and held-out policy evaluation. Copying R² or an optimization runtime from the paper cannot replace these stages. No optimization, security-mechanism replication, model execution or dataset preparation occurred in this document-analysis task.
