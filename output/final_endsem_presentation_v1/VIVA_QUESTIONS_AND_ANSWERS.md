# Viva preparation

43 evidence-bounded questions and answers. Results refer to the 17 November 2026 snapshot; consult SOURCE_INDEX.md for exact artifacts.

## 1. Why federated learning?

To study shared learning across node partitions with heterogeneous local observations. This simulation tests optimization and accuracy; it does not establish deployment privacy or a necessity to decentralize this centrally available dataset.

## 2. Why GRU?

It represents temporal dependence with a relatively compact recurrent model. T6 found lower mean MAE than its evaluated LSTM/XGBoost alternatives, but GRU did not minimize every RMSE and capacity was not matched.

## 3. Why standard FedAvg?

It provides a clear sample-weighted baseline with a defined local-training and aggregation protocol. Adaptive aggregation or FedProx would need their own controlled experiments.

## 4. What is a simulated FL client?

A partition of windows from one mapped compute-node string, executed within a server-side simulation. It is not an independently administered data center.

## 5. Why are 979 clients not 979 data centers?

The 979 identifiers refer to compute nodes of the single Marconi100 installation. Physical-site independence was never created by assigning client labels.

## 6. What does the model predict?

The next hour’s arithmetic mean of six valid future ten-minute bins. B1 predicts power in W and recorded IPMI ambient in °C; T4 predicts power only.

## 7. Is this instantaneous power at t+60 min?

No. The label averages bins spanning [t, t+60 min), which is a different forecasting task.

## 8. How are inputs binned?

Causal time-weighted holds of at most 300 seconds; a 600-second bin must have at least 50% coverage. Weather uses an earlier observation within 30 minutes.

## 9. How is memory utilization validated?

Use aligned valid binned mem_free and positive mem_total, compute 1−free/total, and reject fractions outside [0,1]. Do not silently drop memory.

## 10. What is the node embedding?

A 979×8 learned table replacing categorical slot 13. The GRU receives 13 numeric channels and 8 embedding components at each step.

## 11. Why does embedding dilution occur?

Only the owner client updates its row. Whole-table weighted averaging includes unchanged copies from other selected clients, multiplying its local delta by its processed-example weight.

## 12. Why not row-wise embedding averaging?

That would renormalize by only row-updating clients and change effective row learning rates. It is a custom aggregation algorithm rather than the adopted full-table FedAvg.

## 13. How is B1 local training performed?

98 of 979 clients are sampled per round. Each starts from the same global model, creates fresh SGD at lr 0.01, and runs one local epoch with batch 128 and drop_last=False.

## 14. How are model parameters aggregated?

Sample-weighted sums over processed examples. B1 uses CPU float64 accumulation then casts back; T4 uses its separately implemented float32 aggregation.

## 15. Are all clients trained equally often?

Uniform sampling gives equal expected participation, not identical realized counts. Over 200 rounds the expectation is 200×98/979≈20.02 selections per client, with random variation.

## 16. Does one local epoch mean equal local sample exposure?

Every selected B1 client processes all its accepted local windows once. Unequal client sizes mean unequal local examples and ceil(n/128) optimizer steps.

## 17. Why retain the partial final batch?

drop_last=False preserves declared example exposure and sample weights. Dropping it would change the matched budget and potentially exclude small clients.

## 18. Why is persistence essential?

It is a strong causal last-bin baseline for slowly changing telemetry. A learned model must be compared on identical target windows to show useful forecast gain.

## 19. Why can MAE and RMSE disagree?

RMSE gives large errors more influence. A model can reduce severe mistakes while increasing typical absolute error, as B1’s mixed power comparison illustrates.

## 20. What distinguishes pooled and macro-node metrics?

Pooled aggregates all errors, weighting nodes by window count. Macro computes each node’s metric then averages equally. Macro RMSE is not the square root of average node RMSE.

## 21. What does three-seed sample SD measure?

Variation among seeds 42, 43, 44 on one fixed dataset. It is not a confidence interval for unseen dates, sites or new missingness masks.

## 22. Did FedAvg beat centralized SGD?

No in the v5 paired pooled MAE/RMSE comparisons: central won all 12 policy-seed pairs. T4 also has lower central mean errors, but its budgets differ from exact B1 schedule replay.

## 23. Which fill policy is best?

Within v5, central FF10 has the lowest mean MAE/RMSE; FedAvg differences are small and its best mean RMSE is FF20. This does not generalize beyond that common one-date holdout.

## 24. Are targets ever forward-filled?

No for the reported forecast protocols. Causal input policies may fill histories; the future target bins remain original observed valid values.

## 25. What does B1 establish?

A complete two-output FedAvg pipeline and retrospective evaluation. It beats persistence on pooled power RMSE only among the four pooled target/error comparisons.

## 26. Why did all B1 seeds select round 200?

The prescribed earliest-minimum rule selected the final evaluated round for each seed. This does not authorize more rounds or March-informed tuning; convergence beyond the budget remains unestablished.

## 27. How does T4 differ from B1?

T4 is power-only residual L1 with Adam, batch 64, FF30, masked weather and a different cohort. B1 directly predicts two standardized targets with SGD, batch 128, no-fill and stricter hourly windows.

## 28. What exactly is T4’s persistence improvement?

Canonical pooled MAE 24.8133 vs 33.0661 W, about 24.96% lower; pooled RMSE 57.5502 vs 64.0451 W. Macro MAE 25.2242 vs 33.3504 W is a separate comparison.

## 29. Does 24-hour history always help?

No. The completed old FF30 panel shows a small conditional mean MAE reduction on one matched cohort. A separate newer residual benchmark is ongoing and has unresolved provenance/eligibility issues.

## 30. Does weather have no predictive information?

That is unsupported. The evaluated T5 enabled-weather variants have slightly worse canonical mean MAE than no-weather, but other models, regimes and objectives were not exhausted.

## 31. What did T7 reveal?

Burst masking is more harmful than random masking at comparable requested rates; power-channel blackout is especially damaging. These are fixed synthetic masks, not operational fault guarantees.

## 32. Was T7 trained on missingness?

No. It reuses three frozen T4 models and adds 99 stored condition/seed/period evaluations with causal FF30 and training-mean input fallback.

## 33. Does T7 use independent mask seeds?

No. A fixed seed 42 mask per condition is shared across model seeds; reported seed SD therefore excludes variation across independently simulated outages.

## 34. What is the difference from the base paper?

The paper uses adaptive federated MLP decision learning and a reduced physical optimization solver. We use standard FedAvg recurrent telemetry forecasts; the paper’s full framework is not reproduced.

## 35. Can paper R² be compared with our MAE?

No. The targets, units, references and datasets differ. R² and dispatch-cost deviations cannot rank watt/degree prediction errors.

## 36. Why is an energy-saving claim unsupported?

There is no validated action-response model and controlled policy/energy experiment. Forecast accuracy alone does not show changed energy consumption or service outcomes.

## 37. Is IPMI ambient a calibrated inlet measurement?

Documentation supports an ambient/inlet interpretation, but exact placement, calibration and drift remain incompletely established. Use the recorded metric name and °C rather than invent sensor semantics.

## 38. Is March untouched confirmatory data?

No. It was previously inspected and is explicitly presented as retrospective evaluation. Later model selection cannot be described as prospective confirmation.

## 39. What is the historical integrity status?

Strict historical preservation preflight remains FAILED; semantic reproduction remains UNKNOWN. New scoped hash matches do not erase those statuses.

## 40. What are the nine new manifest failures?

New history coverage/preflight manifests disagree with current source/test/log and historical input bindings in nine comparisons over eight unique files. Older historical input hashes still match their own records; causes of new expected-hash discrepancies remain unresolved.

## 41. What is the next best experiment?

Complete the separately authorized locked March evaluation of the already-trained B1 centralized schedule-replay control on exactly the existing B1 windows, before adding more method variants.

## 42. What remains unanswered?

Prospective multi-site generalization, reliable policy savings, calibrated uncertainty, controlled participation/FedProx effects, independent outage realizations and a fully reconciled new residual history benchmark.

## 43. What is the actual contribution?

A functioning causal forecasting/evidence pipeline plus controlled comparisons showing where federation, residual learning and missing-input handling help or fail. Negative and mixed results are part of the contribution.
