# FedAvg-GRU and matched central control v1

Declared future protocol only. No model/state initialization, optimizer, local training, averaging or communication round runs in Phase2.0.

## Client population and participation

One literal mapped compute node is one simulated client at one facility. Candidate mapping has979 entries; actual eligible training clients and March evaluated nodes are UNKNOWN. Admit from accepted training windows only: >=24 windows and >=7 distinct UTC origin dates. Use the same admitted clients within each controlled suite and its common ordered evaluation windows. Report zero-coverage/unmapped nodes without changing their original mapping indices. Fewer than98 eligible training clients is a stop condition; never silently sample with replacement or reduce participation.

Per round, sample98 clients uniformly without replacement using a stream separate from initialization and shuffling. A selected client processes one complete local epoch over its accepted training windows, with batch512 and drop_last=False. Local training never receives validation/test arrays. Global complete validation belongs to the coordinator; the future data API must enforce that capability boundary. This is not OS isolation or proof of privacy.

## Model, optimization and aggregation

Saved input [N,H,14] supplies13 standardized numerical channels and categorical index13. Embedding979x8 (conditional on preserved mapping) replaces the index inside the network, giving21 recurrent inputs. GRU has hidden64, two layers, dropout0, final recurrent output and Linear64to2 head. No power-only residual buffer is imported. Each seeded paired central/FedAvg GRU starts from the same fresh initialization/fingerprint, not a historical checkpoint. LSTM is a central comparison with the same declared width/layers/head but different cell parameters.

Every selected client starts from a clone of the same immutable round-global state. SGD learning_rate0.01, momentum0, weight_decay0, one local epoch; initialize a fresh optimizer for every client session. Loss is equal-output standardized MSE. For client k with n_k accepted training windows, aggregate theta_next=sum_k(n_k/sum_j n_j)*theta_k over exact participating counts. Validate complete names, shapes, dtypes, finite values, round IDs, common-start fingerprint and counts before aggregation. Abort invalid rounds; no silent client substitution or reweighting. Use deterministic recorded client order, CPUfloat64 accumulation and cast back. Fixed buffers, if any, remain exact under the approved two-output state schema; the old baseline_coefficients buffer is neither required nor transplanted.

The entire shared embedding table undergoes standard FedAvg, including unchanged rows. A client-specific row correction is diluted by its sample weight. No row-wise normalization, personalization or hidden embedding aggregation rule is included. Its effect is a potential diagnostic, not an established explanation for results.

Central SGD replays the same seed-derived ordered clients, per-client row permutations, batch boundaries and example/step totals. One continuing central model processes the batches sequentially; it is not reset to a global snapshot between clients and has no averaging. Zero momentum/decay means optimizer reset per block carries no history. This matches exposure/nominal optimizer settings, not optimization trajectories or communication costs. The masked central control pairs with masked FedAvg; enabled variants pair separately. Each suite/history records its own schedule; fewer native windows cannot silently change a matched comparison.

## Determinism, validation and artifacts

Seeds42,43,44. SeedSequence domains:0 initialization,1 client selection,2 local row shuffling,3 XGBoost output streams. Future implementation must store exact derived seeds and library versions; shuffling coordinates include seed/domain/round/frozen node index. Client selection must depend only on the frozen training cohort. Repeated runs on different software/hardware are not promised bitwise identical merely by naming a seed.

Cap200 rounds/central blocks; complete global validation at0 and every10 through200. Select earliest exact minimum pooled equal-output standardized validation MSE; power/temperature physical MAE/RMSE always accompany it. Random round0 is not the persistence baseline. No test selection or early stopping. XGBoost's two-output shared-iteration selection is specified separately in EVALUATION_SPEC.md.

Log participants/order, mapping/cohort hashes, local training counts and row-order fingerprints, examples processed, optimizer steps, sample weights, start/end parameter fingerprints, finite-state checks, validation sample counts/errors/J, selected candidate, seeds/environment, elapsed time, measured memory and estimated payload bytes. Future atomic round checkpoints/receipt chains must include optimizer/RNG/selection/counters and interruption semantics. Replay a failed round from its unchanged committed start; do not commit partial aggregation. None of that training/checkpoint implementation exists in this release, and deterministic resume is not claimed tested for new two-output models.

Measure per-output pooled/per-node errors and descriptive heterogeneity, convergence by rounds/examples/steps, client participation distribution, state bytes up/down and elapsed/resource use where measurable. A simulated parameter-byte count is not a measured network deployment or energy saving. Existing old residual FedAvg classes require different fixed-buffer contracts and must not be reused without a separately reviewed new adapter.

## Execution gates

Require verified March acquisition; separate safe schema/coverage inspection; raw-causality/temperature-unit acceptance; approved new-scope provenance; accepted matched train/validation/test identities; eligibility-aware training-only scalers; synthetic tests of new trainable model/update/export/resume/evaluator adapters; and a separately approved measured validation-only pilot. Full fits, immutable final selection lock and fixed-model test access each need separate authorization. Phase2.0 supplies no fit command and never launches the36 planned fit identities.

Historical strict preflight remains FAILED and numerical reproduction UNKNOWN. Secure aggregation, differential privacy, cryptographic transport, malicious-client robustness and formal privacy are not implemented or claimed. No FL-superiority, physical multi-data-center, energy-management, cooling, PUE or SLA conclusion is warranted by this protocol.
