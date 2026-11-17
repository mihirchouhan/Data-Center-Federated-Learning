# Complete presentation script

Mihir Chouhan · 25CSE2014 · NIT Goa

Presentation: 19 December 2026. Evidence cutoff: 17 November 2026.

24 slides · planned delivery: 19 minutes 55 seconds including pauses. Spoken narrative: 1,938 words. Rehearse at 110–125 words/minute with chart pauses; shorten technical backup to remain within 20 minutes.

Technical explanations are backup material. Source IDs resolve to exact paths and hashes in SOURCE_INDEX.md.

## Slide 01 — Federated Forecasting

**Approximate time:** 30 seconds

### What to say

Good morning. I am Mihir Chouhan, and this project studies federated forecasting under heterogeneous and missing telemetry. The completed system predicts node IT power, and in one separate experiment also recorded ambient temperature. The broader title describes an energy-aware decision-support direction. Today I will distinguish those measured forecasting results from the operational control work that remains future research. All clients here are simulated compute-node partitions from the same Marconi100 facility.

### Technical explanation

The presentation date is 19 December 2026; the evidence cutoff is 17 November 2026. No later completion is assumed. Student, supervisor and institutional details follow the updated final-report cover.

### Transition

I will first explain why a forecasting study needs careful experimental controls.

### Likely faculty question

Is an energy-management controller already deployed? No; only forecasting and controlled analysis are evidenced.

**Sources:** REPORT

## Slide 02 — Forecasting is a prerequisite, not a savings result

**Approximate time:** 40 seconds

### What to say

A useful energy-management system must anticipate demand and then choose safe actions. This project addresses the anticipation stage. Missing sensors and unequal client histories make even that stage difficult: an apparently better predictor may simply have been tested on easier windows. A lower forecast error does not establish lower energy consumption. That would require an action-response model, operational constraints and a controlled comparison of the resulting policy. This separation guides every result that follows.

### Technical explanation

Power is a rate in watts. Energy requires integration over a specified duration and measurement boundary. Node IT power does not by itself measure facility power, cooling energy, PUE or service-level outcomes.

### Transition

The experimental questions focus on what the telemetry can actually answer.

### Likely faculty question

Why not convert a watt-error reduction into energy savings? Forecast error is not an intervention or energy balance.

**Sources:** REPORT, PAPER

## Slide 03 — Three questions organize the research

**Approximate time:** 40 seconds

### What to say

The first question is whether FedAvg is competitive with an appropriate centralized control and persistence. The second asks how history length, missing telemetry and model formulation affect forecasting. The third asks what is needed before forecasts can inform energy decisions. The contribution is the implemented, evidence-traceable study: causal windows, frozen identities, separate experiment tracks and positive as well as negative outcomes. GRU and FedAvg are established methods; merely combining them is not the scientific claim.

### Technical explanation

The experiments concern known compute nodes at one facility. Across-seed variation describes randomization on the same dataset, not variation across independent sites or periods.

### Transition

The mandatory base paper motivates the final decision-support direction, but solves a different problem.

### Likely faculty question

What is novel here? The controlled telemetry study and its documented limits, not invention of GRU or FedAvg.

**Sources:** REPORT, REGISTRY

## Slide 04 — Base paper: learn operating decisions, then optimize

**Approximate time:** 55 seconds

### What to say

The base paper is Adaptive Federated Learning to Optimize Integrated Flows in Cyber–Physical Data Centers. It models five data centers and learns operating decisions using an ensemble of MLPs. Each MLP has three hidden layers followed by an output layer. Selected blocks are shared, while agents adaptively accept aggregated updates using recent validation behavior. Local inferred decisions then reduce a downstream optimization problem involving generators, storage and grid exchange. This is different from learning the next hour of measured sensor power.

### Technical explanation

The paper uses MLP hidden widths 256, 1024 and 256, followed by an output layer. ReLU and dropout 0.15 are specified. Selected shared blocks and adaptive local acceptance are described in Eq. (11) and Algorithm 1. Its five modeled data-center agents differ from our node-level clients.

### Transition

That distinction matters when comparing both architecture and reported numbers.

### Likely faculty question

Is this a reproduction of the paper? No; neither the MLP ensemble nor reduced physical optimization is reproduced.

**Sources:** PAPER

## Slide 05 — Same federated idea; different labels and evidence

**Approximate time:** 55 seconds

### What to say

The similarity is distributed local learning with aggregation. The paper predicts decision variables and evaluates decision accuracy and economic optimization. Our models predict observed future telemetry and report errors in watts or degrees Celsius. The paper reports adaptive R-squared values from 0.951 to 0.988, but these cannot be compared with our watt MAE. Its approximate optimization runtime reduction is also not a forecast-training speedup. We have not implemented adaptive acceptance, decision inference or the downstream physical solver.

### Technical explanation

Paper Table II (p. 9) supplies R². Table III (p. 10) reports aggregate cost deviations of 0.07% for default FL and 0.08% for adaptive FL, so adaptive FL does not win every reported quantity. Figure 8 compares approximately 0.61 s with 5.30 s for the optimization workflow, not model training.

### Transition

I will now show the implemented telemetry-to-results pipeline.

### Likely faculty question

What additional evidence connects this work to the paper? A validated decision layer using compatible physical objectives and constraints.

**Sources:** PAPER, COMPARISON

## Slide 06 — The implemented pipeline ends at verified forecasts

**Approximate time:** 45 seconds

### What to say

Raw Parquet telemetry is converted into causal ten-minute bins. Window builders retain node identities, timestamps, feature order and observed future targets. Each experiment freezes its cohort and training-only normalization. Local learning and aggregation use that declared contract. Validation chooses a checkpoint before the test interface is opened, and saved predictions support pooled and per-node analysis. Reports and locks bind these stages. The decision layer at the right is a proposal rather than an executed component.

### Technical explanation

Historical artifact preservation and scientific correctness are separate checks. A matching hash proves that bytes agree with a record; it does not establish every modeling assumption or a successful semantic replay.

### Transition

The preprocessing contract determines which forecasts are eligible.

### Likely faculty question

Are these independent distributed servers? The federation is simulated on one centrally accessible server.

**Sources:** REPORT, ARCHITECTURE, REGISTRY

## Slide 07 — Fourteen slots retain operational and thermal context

**Approximate time:** 50 seconds

### What to say

Inputs retain power, four CPU utilization channels, aligned memory-used fraction, recorded IPMI ambient temperature, two weather channels, calendar terms and node identity. The node index is categorical and becomes an embedding. Local sensors use causal time-weighted holds capped at five minutes; a ten-minute bin needs at least half its duration covered. Memory usage is computed only from valid aligned free and total memory bins. Weather is joined from a strictly earlier observation within thirty minutes. Indoor ambient and outdoor temperature are distinct channels.

### Technical explanation

Slots 0–13 preserve the original order. CPU channels are user, system, idle and wio. Memory fraction is 1 − aligned free/total, restricted to [0,1]. The IPMI ambient sensor’s exact placement and calibration remain incompletely documented. Weather masking and causal input filling vary by track; future targets remain observed.

### Transition

The next diagram makes the time convention explicit.

### Likely faculty question

Does valid bin mean all 600 seconds observed? No; the accepted coverage threshold is 50%.

**Sources:** PREPROCESS, FEATURES, REPORT

## Slide 08 — Predict the next-hour mean, not a point at +60 min

**Approximate time:** 55 seconds

### What to say

For B1, the input ends immediately before the forecast origin and contains thirty-six completed ten-minute bins. The target is the arithmetic mean of the next six observed power bins and, separately, the next six observed ambient-temperature bins. Origins are hourly, and the entire history and target lie inside the assigned split. Training ends on 28 January, validation ends on 4 February, and March is a separate retrospective test. These stricter B1 windows must not be confused with historical tracks that permit earlier-split history or overlapping forecast intervals.

### Technical explanation

B1 splits are half-open UTC intervals: train [1 Jan, 28 Jan), validation [28 Jan, 4 Feb), test [1 Mar, 1 Apr), all in 2022. Power targets are 1000 times the mean future kW bins; ambient targets are the mean future °C bins. T4 uses the separate Candidate 1 January 22 training cut and ten-minute origins.

### Transition

Clients are partitions of these windows, not separate physical facilities.

### Likely faculty question

Why can two six-hour studies have different window counts? Their splits, origin cadence, coverage and target contracts differ.

**Sources:** B1_PREP, B1_EVAL, REPORT

## Slide 09 — 979 clients means 979 mapped compute nodes

**Approximate time:** 45 seconds

### What to say

Each client corresponds to one original node identifier. Local windows stay within that node, and its ordinal is preserved in the shared mapping. The recurrent model uses an eight-dimensional node embedding, so it can model systematic differences among known nodes. Client data counts and error distributions differ. This is meaningful heterogeneity, but it is not evidence about unseen data centers. The local privacy boundary is simulated rather than a deployed cross-institution security boundary.

### Technical explanation

The 979×8 embedding replaces categorical slot 13. Each recurrent input therefore has 13 numeric values plus 8 embedding components. Only a client’s own row receives its data-dependent gradient, while full-table FedAvg also averages unchanged copies from other clients.

### Transition

The two principal GRU formulations use that mapping but predict different quantities.

### Likely faculty question

Can the trained embedding generalize to a new node? Not without an explicitly evaluated extension or alternative identity representation.

**Sources:** B1_MODEL, B1_RUNNER, REPORT

## Slide 10 — Two GRU formulations answer different questions

**Approximate time:** 55 seconds

### What to say

Both models use two GRU layers with sixty-four hidden units and an eight-dimensional node embedding. B1 directly predicts two standardized outputs through a linear head: power and recorded ambient temperature. Its loss balances outputs in their training-standardized units. T4 instead predicts a standardized correction to the last power bin and adds that correction back in physical units. T4’s winning configuration uses L1 loss. We must keep those targets, losses and checkpoints separate when interpreting their very different errors.

### Technical explanation

B1 has a Linear(64,2) head without a residual, standardized two-output MSE, SGD at 0.01 and batch size 128. T4 has a Linear(64,1) head, residual inversion, L1 loss, Adam at 0.001 and batch size 64. There is no additional hidden prediction-head layer. T4 has 49,561 parameters.

### Transition

The aggregation method is standard FedAvg; the optimization details still matter.

### Likely faculty question

Why residual prediction? It starts from an available persistence anchor and learns a correction; its benefit is empirical and track-specific.

**Sources:** B1_MODEL, B1_RUNNER, T4_SOURCE, T4_CONFIG

## Slide 11 — FedAvg averages complete local model states

**Approximate time:** 55 seconds

### What to say

At each B1 round, ninety-eight clients are sampled uniformly without replacement. Every selected client starts from the same global state, creates a fresh SGD optimizer and processes one local epoch, including its partial final batch. The coordinator weights each complete model state by the number of processed examples. B1 accumulates in CPU float64. A node’s embedding update is diluted by unchanged copies from the other clients; that is accepted standard full-table FedAvg behavior. The matched B1 central control replays the recorded batches on one continuing model instead of cloning and averaging.

### Technical explanation

Full-table FedAvg is θ_next = Σ n_k θ_k / Σ n_k. If only owner i updates embedding row i, its global change is (n_i / Σ n_k) times its local change when selected; otherwise it is zero. B1 uses CPU float64 accumulation. T4’s separate implementation uses float32 accumulation.

### Transition

Evaluation must preserve the same distinction between fitting, selection and reporting.

### Likely faculty question

Why not row-wise aggregation? It changes the algorithm and effective row learning rate; it is a custom alternative, not the adopted standard FedAvg.

**Sources:** B1_RUNNER, FEDAVG, REPORT

## Slide 12 — Frozen identities make comparisons interpretable

**Approximate time:** 50 seconds

### What to say

Each track fixes chronological windows and its scaler population. Validation selects checkpoints; test results are not used to choose a new checkpoint. B1 validates at rounds zero through two hundred in ten-round increments, selecting the earliest minimum joint standardized RMSE. We report power and temperature separately in physical units. Pooled metrics weight examples, while macro-node metrics give each represented node equal weight. Three-seed sample standard deviation describes those three scores, not a confidence interval or uncertainty across independent datasets.

### Technical explanation

MAE is mean absolute error; RMSE is the square root of mean squared error. Macro-node RMSE averages the individual node RMSEs equally. B1 selection uses sqrt(0.5 × sum over outputs of (RMSE / training standard deviation)²). A strict improvement comparison preserves the earliest checkpoint on a tie.

### Transition

The evidence registry therefore separates the completed studies rather than pooling them.

### Likely faculty question

Why can macro and pooled rankings differ? Unequal node sample counts and nonlinear square-root aggregation.

**Sources:** B1_RUNNER, B1_EVAL, REGISTRY

## Slide 13 — Completed evidence is a portfolio, not one leaderboard

**Approximate time:** 45 seconds

### What to say

The legacy shared LSTM, eighteen early architecture/weather fits, six one-hour federated/central controls, the twenty-four-fit v5 matrix, B1 and T4 are separate experiments. The completed history comparison has nine one-layer fits. T2 and T3 examine temporal extension and adaptation; the weekly variant has only seed42. T5 and T6 add weather and model-family evidence, partly reusing T4 reference checkpoints. T7 adds ninety-nine saved stress-evaluation records without new model fitting. The newer two-layer history benchmark is externally progressing, with no accepted completed matrix in this presentation. This registry prevents duplicated reference models from being counted as new independent experiments.

### Technical explanation

T5 adds nine weather fits and reuses three T4 references. T6 adds three LSTM and three XGBoost fits, reusing the same GRUs. T7 has 99 condition/seed/period records and no new fits. B1 central training is complete, but its March scoring output is not verified present. T3 weekly adaptation has seed 42 only.

### Transition

First, the earlier results show why federation itself cannot be assumed to improve accuracy.

### Likely faculty question

Did every CSV row represent a newly trained model? No; multiple periods and reused references generate multiple evaluation rows.

**Sources:** REGISTRY, V5, B1_EVAL, T4, T5, T6, T7

## Slide 14 — Earlier results already challenge an FL advantage

**Approximate time:** 55 seconds

### What to say

On its own one-hour cohort, the legacy shared LSTM beats persistence on both pooled errors. It is not a federated model. In the separate six-hour v5 experiment, centralized SGD has lower pooled MAE and RMSE than FedAvg in every policy-seed pair. Among central policies, FF10 has the lowest mean errors. However, even its mean MAE remains slightly above persistence, while its RMSE is lower. That test contains only one calendar date, so the finding is narrow and cannot rank all centralized and federated learning methods.

### Technical explanation

The legacy panel has 75,566 windows and 979 nodes. The v5 panel has 6,590 windows, 570 nodes and 17 ten-minute origins on 24 February. v5 persistence MAE is 27.1471 W, central FF10 mean MAE is 27.3466 W and FedAvg FF10 mean MAE is 29.8255 W. Error bars are sample SD across three seeds.

### Transition

B1 broadens the task to two outputs and a separate March cohort.

### Likely faculty question

Does forward-fill always help? No; effects are conditional on native coverage, fixed sampling budget and the common holdout.

**Sources:** LEGACY, V5, V5_SEEDS

## Slide 15 — B1 two-output forecasting has mixed outcomes

**Approximate time:** 60 seconds

### What to say

B1 uses no input forward-fill, full features, and two observed next-hour targets. All three seeds select round two hundred under the prescribed validation rule. On the same March windows, the GRU reduces power RMSE from about ninety-two point six to eighty-nine point eight watts, but increases power MAE. Ambient-temperature MAE and RMSE are both worse than persistence. This is a completed multi-output experiment with informative negative outcomes. It does not justify a claim that joint modeling uniformly improves either target.

### Technical explanation

B1 has 280,499 training and 82,993 validation windows. Pooled power MAE/RMSE are 53.3568 ± 1.5126 / 89.7978 ± 1.1387 W. Ambient MAE/RMSE are 0.44787 ± 0.00768 / 0.72480 ± 0.01021 °C. The companion table supplies macro-node metrics. Central selected blocks are 200/190/190; no central March scores are supplied.

### Transition

T4 explores a different power-only residual formulation.

### Likely faculty question

Why can power RMSE improve while MAE worsens? The error distribution may have fewer severe errors but larger typical absolute errors; inspect distribution rather than infer a cause.

**Sources:** B1_EVAL, B1_PERSIST, B1_LOCK

## Slide 16 — T4: residual L1 forecasting improves persistence

**Approximate time:** 60 seconds

### What to say

T4 is a different power-only FF30 study. Its residual L1 FedAvg model reaches a pooled MAE of twenty-four point eight one watts versus thirty-three point zero seven for persistence, a reduction of about twenty-five percent. Pooled RMSE also decreases. The macro-node comparison supports a broad improvement rather than a gain driven only by heavily sampled nodes. The centralized T4 model has lower mean errors still, so this result supports the residual forecasting configuration, not a causal claim that federated training produced the gain.

### Technical explanation

T4 screened 13 validation pilots with seed 42. Final FedAvg selected rounds are 197/198/200. The central models ran 20 full-union epochs and selected epochs 2/3/2; this is not exact schedule replay. Pooled 24.8133 W pairs with pooled persistence 33.0661 W; macro 25.2242 W pairs with macro persistence 33.3504 W. SD uses ddof=1.

### Transition

The next question is whether that result survives a later time period.

### Likely faculty question

Is this the best model across all project experiments? It is strongest against its own persistence baseline; different cohorts prevent a global ranking.

**Sources:** T4, T4_SEEDS, T4_AUDIT, T4_MANIFEST

## Slide 17 — T4 gains persist, but later-period errors increase

**Approximate time:** 55 seconds

### What to say

T4 remains below its same-window persistence baseline in both the extended and March views. March mean MAE rises to about thirty-eight point three watts from twenty-four point eight on the canonical period. The central model remains slightly lower. This is evidence of temporal sensitivity, but not proof of a particular cause of drift. The extended evaluation contains the canonical and March intervals; these are not three independent test datasets. The small three-seed spreads do not quantify uncertainty across future facilities or years.

### Technical explanation

T4 canonical coverage is 243,138 windows / 978 nodes; extended is 3,609,285 / 979; March is 2,229,042 / 976. FedAvg node MAE wins over persistence for seeds 42/43/44 are 900/897/898 on canonical and 955/953/953 on March. These March windows differ from B1’s 194,212 hourly windows.

### Transition

History-length results require the same care about model and cohort contracts.

### Likely faculty question

Why not average the three temporal scores? The overlapping windows and different periods would yield an uninterpretable aggregate.

**Sources:** T4_SEEDS, T4_AUDIT

## Slide 18 — Longer history shows a small conditional benefit

**Approximate time:** 55 seconds

### What to say

The completed one-layer history study holds its longest-history-eligible cohort fixed across one, six and twenty-four hours. Mean pooled MAE decreases from thirty-three point three zero to thirty-two point six five and then thirty-two point five seven watts. Most of the measured change occurs between one and six hours; the further change is small. This does not establish universal benefit from twenty-four hours. A separate newer two-layer residual-MSE benchmark is ongoing and has unresolved eligibility and provenance issues, so its provisional scores are excluded.

### Technical explanation

The old matched history panel uses 243,138 test windows / 978 nodes, a one-layer GRU, direct-power MSE, Adam 0.001, batch 64, FF30 and masked weather. Mean MAE improves about 1.95% from 1 h to 6 h and 0.23% from 6 h to 24 h. The newer residual runner is distinct; all clients fail its stated 1,024-window threshold, and nine manifest comparisons fail.

### Transition

Weather and model-family studies add two more controlled comparisons.

### Likely faculty question

Can no-fill support twenty-four hours? Coverage depends on split and feature contract; a prior historical test had zero windows, while a new common audit records a small nonzero cohort. Do not generalize across contracts.

**Sources:** HISTORY, HISTORY_SEEDS, NEW_STATUS, FAILED_BINDINGS

## Slide 19 — Weather and model families: gains depend on metric

**Approximate time:** 55 seconds

### What to say

In T5, the no-weather T4 reference has slightly lower canonical MAE than the temperature-only, humidity-only and combined-weather conditions. That small difference does not show weather is universally irrelevant. T6 compares the reference GRU, a federated LSTM and centralized XGBoost. GRU has the lowest mean MAE, but XGBoost has the lowest canonical RMSE. These are not equal-capacity or equal-algorithm models, and the LSTM also has lower RMSE in some later views. Model choice therefore depends on the criterion, period and control design.

### Technical explanation

T5 canonical MAEs are 24.8133 W without weather, 24.8836 with temperature, 24.9316 with humidity and 24.9593 with both. T6 compares a 49,561-parameter GRU, 63,449-parameter LSTM and centralized XGBoost using 504 flattened features. Reference GRUs are reused. XGBoost’s best-iteration prediction handling remains an implementation-review limitation.

### Transition

We then stress the frozen T4 models rather than train another model.

### Likely faculty question

Is GRUsuperior to XGBoost? Lower MAE here, but higher canonical RMSE and different training/representation make a universal ranking unsupported.

**Sources:** T5, T6, LATE_RESULTS

## Slide 20 — Missingness pattern matters as much as its rate

**Approximate time:** 55 seconds

### What to say

T7 applies synthetic input loss to the frozen T4 models. At twenty percent requested missingness, random loss increases MAE to about twenty-six point five watts, while one-hour burst masking raises it to about thirty-two point four. Longer contiguous losses are more damaging. The preprocessing first uses causal FF30 and then a standardized training-mean input fallback. Future targets remain observed. The mask is fixed across model seeds, so the error bars measure model-seed variation rather than variation across independent failures.

### Technical explanation

T7 has 75 canonical, 12 extended and 12 March records, with no new fits. Burst starts are distinct but their intervals may overlap; requested rates differ from realized removal rates. Clean RMSE is 57.5502 W from CSV, not the old narrative’s 44.97 W. Persistence uses the available input anchor for each condition.

### Transition

The node-wise view reveals where this robustness breaks down.

### Likely faculty question

Does this simulate actual packet loss? It is a synthetic corruption protocol; no causal identification of real failure mechanisms is established.

**Sources:** T7, T7_SUMMARY, T7_SOURCE

## Slide 21 — Broad node gains coexist with a power-sensor failure

**Approximate time:** 50 seconds

### What to say

On the clean canonical cohort, about ninety-two percent of nodes beat persistence by MAE on average across the three seeds. That fraction decreases as missingness becomes more difficult. Complete power-channel loss is especially damaging: mean MAE becomes about one hundred twenty-five watts, slightly worse than that condition’s persistence baseline. CPU, memory and ambient blackouts have much smaller effects in this setup. This supports a dependence on the power-history anchor, not a claim that the other sensors are universally unnecessary or that the system is production fault tolerant.

### Technical explanation

The mean fraction of nodes beating condition-specific persistence is 91.854% clean, 90.695% under random 20% loss and 66.326% under one-hour bursts at 20%. Power blackout MAE is 124.9714 W versus 124.8222 W persistence. The missingness audit denominator is 7,154,532 node-grid/channel values. These results do not establish a formal non-IID taxonomy.

### Transition

These forecasting findings define what a future decision layer must validate.

### Likely faculty question

Why does the power blackout hurt a residual model? It removes the anchor and key temporal signal; the fallback changes the persistence reference too.

**Sources:** T7_SUMMARY, T7_NODES, T7_SOURCE

## Slide 22 — Energy-aware decisions remain a separate research layer

**Approximate time:** 45 seconds

### What to say

The completed component supplies forecasts with known coverage and failure modes. A future decision layer would combine them with workloads, prices, operational constraints and a validated action-response model. It would then compare a forecast-informed policy against a defined baseline using actual energy, cost and service outcomes. No such closed-loop experiment is claimed here. The base paper motivates this architecture, but its optimizer and physical model cannot be assumed to transfer directly to M100 telemetry.

### Technical explanation

Energy, PUE and cooling objectives require compatible facility-level measurement boundaries. SLA benefits require an operational definition and an intervention study. Point forecasts and three-seed SD do not supply calibrated predictive uncertainty.

### Transition

Before that next stage, the existing evidence and remaining integrity limits must be explicit.

### Likely faculty question

What is the immediate next experiment? Complete the separately authorized same-window B1 centralized March comparison before adding further algorithm variants.

**Sources:** PAPER, REPORT, B1_CENTRAL_STATUS

## Slide 23 — Limits and integrity gates remain visible

**Approximate time:** 50 seconds

### What to say

All major conclusions remain conditional on one facility and already inspected retrospective periods. Test-window counts differ sharply across tracks. Small seed standard deviations are not evidence of independent replication. Historical strict preservation preflight remains failed and semantic reproduction remains unknown. Nine newer history-audit manifest bindings also fail, despite the referenced historical inputs matching older records. We have preserved these failures rather than rewriting hashes. The current presentation is an evidence snapshot; the externally progressing benchmark is not promoted to a completed result.

### Technical explanation

The presentation freshly checked 109 report artifacts, reproduced 638 matches and nine known failures in 647 prior content comparisons, and rechecked 279 source bindings. These are bounded byte checks, not a prediction replay or historical end-to-end reproduction. No model or GPU operation occurred during presentation creation.

### Transition

I will close with the strongest measured conclusions and the next defensible step.

### Likely faculty question

Can this work be called reproducible? The repository preserves substantial traceability, but failed historical preflight and missing semantic replay prevent an unconditional reproducibility claim.

**Sources:** REPORT_VALIDATION, FAILED_BINDINGS, AUDIT, NEW_STATUS

## Slide 24 — Conclusions: useful forecasts, bounded claims

**Approximate time:** 35 seconds

### What to say

The strongest result is that residual L1 power forecasting improves persistence within the T4 contract, while centralized controls remain competitive or better. B1 demonstrates two-output execution but has mixed baseline performance. Missingness pattern and the power-history anchor materially affect accuracy. Longer history and weather yield conditional results rather than universal rules. The immediate next step is the locked, separately authorized B1 centralized March comparison, followed by repaired provenance for new studies and eventually a validated decision layer. Thank you; I welcome your questions.

### Technical explanation

No universal FL superiority, energy saving, cooling/PUE/SLA gain, formal privacy guarantee, unseen-site generalization or universal 24-hour-history benefit is established. The presentation starts no experiment and grants no new authorization.

### Transition

Invite questions and use the companion viva answers and metric appendix for detail.

### Likely faculty question

What is the single main takeaway? Evaluate forecasting changes on identical windows with correct baselines, and separate accuracy from operational benefit.

**Sources:** REPORT, V5, B1_EVAL, T4, T7
