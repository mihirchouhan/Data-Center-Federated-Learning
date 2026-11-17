# Claims not verified or contradicted

These qualifications are retained instead of silently repairing old artifacts. `UNVERIFIED_CLAIMS.csv` provides the complete machine-readable register.

**Q01 — T7 narrative clean RMSE44.97 W.** Contradicted. Evidence: T7_RESULTS_PER_CONDITION.csv and SUMMARY give57.5502 W. Treatment: Use CSV-backed RMSE, preserve original narrative.

**Q02 — New history preflight all 979clients meet>=1024 windows.** Contradicted. Evidence: CLIENT_ELIGIBILITY: all False; counts12–985. Treatment: Mark readiness unresolved; do not change threshold.

**Q03 — New T1.1/T1.2 execution provenance verified/frozen.** Unverified / failed bindings. Evidence: Nine actual SHA-256 mismatches in their new manifests. Treatment: Retain failures; no hash regeneration.

**Q04 — Three-way common windows exist only underFF30.** Contradicted. Evidence: Positive no-fill/FF10/FF20 intersections in common CSV. Treatment: FF30 is largest here, not uniquely nonempty.

**Q05 — Late-February Ganglia daemon crash proven.** Not verified. Evidence: Channel gaps do not identify operator/software cause. Treatment: Describe measured telemetry gaps only.

**Q06 — T4winner and proposed history runner have identicalMSEloss.** Contradicted. Evidence: T4L1; new runner/config MSELoss. Treatment: Require explicit scientific decision before execution.

**Q07 — T7 tests21/21passed.** Not verified. Evidence: Current test source has4test_* methods; no supported21-test log found. Treatment: No fresh test execution; constructor accesses actual data.

**Q08 — Burst blocks never overlap.** Contradicted by implementation. Evidence: Unique start positions can produce overlapping spans. Treatment: Call it sampled-start burst masking; report requested versus realized rates.

**Q09 — T7 missingness audit denominator7,138,530 window elements.** Contradicted. Evidence: CSV:7,154,532 node-grid/channel values. Treatment: Distinguish grid accounting from windows.

**Q10 — Random real-world loss predominates / FF30 solves packet loss.** Not evaluated. Evidence: Synthetic generator, one fixed mask per condition. Treatment: Limit to this synthetic stress study.

**Q11 — Power blackout predicts exactly the training mean.** Too strong. Evidence: Source adds learned residual to mean-anchor fallback. Treatment: Separate anchor from final prediction.

**Q12 — T6 establishes a universal Pareto winner includingTransformer.** Not supported. Evidence: No Transformer result in verified matrix; LSTM lower RMSE in some views. Treatment: Restrict to actual models/metrics and measured resources.

**Q13 — New residual history memory/runtime estimates guarantee negligibleOOMrisk.** Not verified. Evidence: Numbers reused from different older1-layer runs. Treatment: Label estimates; no execution readiness inferred.

**Q14 — T7 audit proves prior T5/T6 contents unaltered.** Overstated. Evidence: audit_t7_integrity.py checks existence of three prior artifacts. Treatment: Use actual byte checks separately; no blanket audit claim.

**Q15 — All T7 predictions/identities independently replayed.** Not verified. Evidence: Saved metrics/counters; no frozen prediction archive replay in this task. Treatment: Counts/code intention do not prove ordered prediction equality.

**Q16 — Named uploaded base paper byte-identical to reviewedPDF.** Not verified. Evidence: Uploaded copy unavailable; cached officialv3 hash matches prior identity. Treatment: Keep documented official-PDF substitution.

**Q17 — Historical semantic reproduction or preflight repaired.** Not established. Evidence: Historical preflight FAILED; reproduction UNKNOWN. Treatment: Preserve status; report adds no approval.

**Q18 — Energy/SLA/PUE/cooling benefits or operational control demonstrated.** Not evaluated. Evidence: Forecast errors and proposed decisions only. Treatment: No savings,service or deployed-control claims.


**Q19 — Historical FF10 validation node count is impossible as labeled.** Three new common-cohort rows report 146 common windows across 979 common nodes; active nodes cannot exceed windows. Those historical node counts are not accepted. Candidate1 totals in the report are separately reconciled with its per-client table.
