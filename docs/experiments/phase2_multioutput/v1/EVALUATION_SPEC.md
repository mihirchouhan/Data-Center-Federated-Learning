# Two-output evaluation specification v1

This freezes definitions, not results or March availability. The approved time intervals, features and cohort rules are in PROTOCOL.md and protocol.json. No models, real target windows, predictions or metrics are computed in Phase2.0. Mathematical metric functions are tested only on fabricated hand-calculated arrays.

## Shapes, observations and identities

Raw numerical input shape is [N,H,14], H in {6,36,144}; the final categorical slot is the frozen literal node mapping index. Targets and predictions have shape [N,2], ordered power W then node-inlet temperature degC. The node embedding is an independent parameter table, not extra saved features. A window identity is (literal node string, UTC datetime64[ns] forecast origin), ordered by original mapping then origin. The new versioned digest uses length-prefixed canonical JSON identity rows; it is not falsely equated with historical digest algorithms.

All six future ten-minute bin START timestamps must be exactly t+[0,1,2,3,4,5]*600s and belong to the same node. All required target observations are original, finite and valid; no target filling is allowed. y_power=1000*mean(power_kw in six future bins); y_temperature=mean(ambient_C in six future bins). Histories are exactly t+[-H,...,-1]*600s, contiguous and same-node. The entire history and future lie in the assigned split, including the half-open future endpoint. Raw contributor bounds must exclude future observations and earlier-split samples. Raw-to-bin coverage/hold semantics still require a separate implemented audit, not trust in a flag.

Comparisons verify identity, target timestamps, original target values, target means, split and mapping. Identical counts alone are insufficient. Masked weather is applied after standardization and changes only slots7/8, preserving input shape, workload/calendar/ID values and frozen scalers. Primary shared windows require enabled-weather validity; masked native availability is descriptive only.

## Metrics and persistence

Persistence uses each window's last original input bin: power_kw*1000 and inlet_C. It predicts those two scalar values for the corresponding next-hour means. It is not a prior-hour average, fitted model, random seed, initial neural checkpoint or energy forecast.

For each target j, compute e_ij=prediction_ij-observed_target_ij in physical units. MAE_j=sum(abs(e_ij))/N; RMSE_j=sqrt(sum(e_ij^2)/N). Never average W and degC into a physical scalar. Never drop nonfinite predictions/windows silently. Report sample and node counts and ordered identity digest with every result.

Per-node scores use that node's windows. Pooled MAE=sum_n(N_n/N)*MAE_n; pooled RMSE=sqrt(sum_n(N_n/N)*RMSE_n^2). Equal-node MAE and RMSE are the arithmetic means of node MAEs and node RMSEs respectively. A mean of node RMSEs is not pooled RMSE. Include per-origin/date sample/node coverage and signed/absolute error distributions in the future reporter; the present minimal pure scorer implements pooled/per-node/equal-node MAE and RMSE. It does not yet write plots or those additional diagnostics.

For each learned configuration retain all seeds42,43,44. Report individual scores, their arithmetic mean and sample SD(ddof=1), not an ensemble prediction or a confidence interval. Persistence has one deterministic vector per cohort; repeating its identical score is not three independently fitted baselines. Improvement percent, if later reported, is100*(reference_error-model_error)/reference_error, separately by output/metric. Zero references require an explicit undefined/zero policy; this release does not implement percentage scoring.

## Validation selection and test protection

Training loss is equal-output standardized MSE. For selection use J=0.5*((RMSE_power_val_W/sigma_power_train_W)^2+(RMSE_temp_val_C/sigma_temp_train_C)^2). Both immutable training scales must be nonconstant. Choose the earliest exact minimum; do not introduce a post-hoc tie tolerance. Report all physical metrics even when MAE disagrees with the RMSE-oriented criterion.

Neural candidates are round/block0,10,...,200. Round0 is a fresh randomly initialized non-residual two-output network, not persistence. No early stopping; a fixed cap avoids seed-dependent budget changes. A final trace must include every scheduled candidate; partial traces are not reported as completed selection.

XGBoost trains two scalar squared-error boosters on standardized targets. They share one selected boosting iteration chosen from the joint J trace at iterations0,...,500. Do not combine a power-optimal tree count with a different temperature-optimal count after inspection. The larger selection opportunity and different training budget are explicit limitations of between-family comparisons.

Before any eventual test examples are opened, a separately authorized immutable selection lock must bind all declared fit IDs/seeds, selected model exports and hashes (two boosters where relevant), validation trace and selected index, protocol/matrix/source/environment, scaler/fit population, node mapping/eligibility, ordered test IDs and original six-bin target provenance. Verify actual files, not only stored filenames or booleans. No lock is created now. The pure metadata lock validator is only one future component; it does not hash model bytes or provide a full test evaluator.

Test evaluation must consume exactly those locked models and identical ordered March windows. It cannot reselect checkpoints, choose a best test seed, retune hyperparameters, expand masks' cohorts or splice output heads. Interrupted evaluation preserves partial outputs; resume requires a separately implemented/tested mechanism, otherwise review a new unused retry directory with the unchanged lock. Never delete results to rerun.

## XGBoost feature mapping

Flatten thirteen standardized non-ID channels oldest-to-newest, followed by one categorical node index. Column13*s+j corresponds to feature j at10*(H-s) minutes before origin. Column13*H is node_index with native categorical feature type c, locked mapping and categorical-enabled serialization. Shapes are79,469,1873 columns for H6,36,144. Treating node IDs as an ordered numerical feature is prohibited. Store lag_mapping(H), source and mapping hashes with each future booster. Forecasting forecasts only two outputs; no cooling/PUE/SLA label is introduced.

## Interpretation limits

March is historical retrospective data whose prior exposure must be audited, not declared untouched by this document. January–February has already been observed. Seeds and nodes share one dataset/facility; targets are non-overlapping within a node, but histories and environmental conditions are dependent. No independent-dataset significance or confidence claim follows from three seeds. No measured energy savings, cooling outcome, PUE/SLA improvement, privacy guarantee or longer-history advantage exists here.
