# Evidence-led debugging and safe recovery

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

Read the exact error and version's source/config/receipt, preserve failure evidence, then propose the smallest change. Never recreate provenance, update expected hashes, delete locks or retrain as a diagnostic shortcut.

| Symptom | Evidence / mechanism | Safe next action |
| --- | --- | --- |
| Missing scripts/download_datasets.py | That is not this pipeline's entrypoint | Confirm root and [API](API.md), don't mix projects |
| Slow download / TLS chain failure | Provider throughput differs from local speed test; TLS fails explicitly | Inspect request/evidence and approved CA bundle; never disable TLS |
| NVML driver/library mismatch | Recorded historical server incident | Optional operator nvidia-smi; administrator fix, no driver update here |
| CUDA unavailable | Locked v5 forbids CPU fallback | Check interpreter/device visibility/admin environment; no plan edit |
| Execution source changed | training_io.sources hashes actual bytes | Compare recorded snapshot/backup; only restore verified originals under explicit repair approval or approved additive version |
| Matrix/manifest differs | Config/software/source/device binding mismatch | Inspect exact fields; don't re-sign or change receipts |
| Preservation failed | Hash/size/mode/mtime comparison; unresolved historical gate | Read discrepancy register, preserve failure; no touch/PID restoration/baseline reset |
| Missing nested run | Old evaluator assumed nested paths; actual names flat/legacy | Use existing v2 manifest discovery, don't rename runs |
| Evaluation output exists | Partial/complete outputs deliberately never overwritten | Verify existing inventory; preserve partial. Any approved repeat needs unused version |
| Missing metric/nonfinite history | Required channel/coverage absent | Read schema/exclusions; no invented zeros or gap bridge |
| IDs/targets/baseline mismatch | Counts alone don't prove same examples | Compare ordered node/origin, observed bins, split and content hashes |
| Nonfinite loss/update | Whole invalid round/session aborts | Preserve pending evidence; inspect input/scaling/state, don't silently drop client |
| Background fresh parent absent | Older statvfs assumption fixed in later wrappers | Use verified approved version; no manual lock/ownership bypass |
| Missing/changed validator, 200 on resume, 416 | Transfer checks fail closed | Preserve response/checkpoint; no guessing/truncation/budget reset |
| Disk/quota/invalid checkpoint | Capacity isn't quota proof; durable prefix mismatch can block resume | Stop for quota/recovery review; preserve all partials |
| Phase 2 data gate exit 2 | Deliberate missing-adapter/coverage blocker | Approved versioned implementation, not bypass |

## Read-only diagnosis

```bash
cd /media/csedept/cse2014/Project/m100_training_template
pwd
.venv/bin/python -B --version
.venv/bin/python -B -m json.tool output/phase1_integrity_review/integrity_summary.json
sha256sum src/v5_missing_data/evaluate_policy_models.py src/v5_missing_data/evaluate_policy_models_v2.py
rg -n 'def (load_plan|verify_run)|Execution source changed'   src/v5_missing_data/run_policy_experiments.py src/v5_missing_data/training_io.py
```

Compare against the affected run's expected snapshot/lock, not a guessed latest hash. Don't hash a growing March archive or invoke acquisition lifecycle actions without inspecting side effects. [API](API.md) separates lock and output verification.

## Resume boundaries

v5 explicit resume requires the original method/policy/seed/kind/cap, environment/source/data and contiguous committed ledger. Trace/state hashes and deterministic plans are checked. Preserve .pending-* work; don't promote it to a committed round. Smoke remains capped at 3, main at 200. No generic resume command is safe until the owned run is identified. Historical v1 partial fits have different no-auto-resume behavior.

Acquisition uses its source-pinned helper/wrapper/session with existing validators and request budget. Preserve V1/V2 sessions. SSH/laptop-independent background execution does not imply reboot restart. Exit status alone is not archive success; require exact-size/checksum publication receipt.

## Unresolved historical discrepancy

The investigation records 26 files/28 differing fields: 25 metadata-only files plus one changed operational cache lock. Re-extraction evidence supports 24 Parquet mtime changes. Exact last lock writer and source-mtime operator remain [UNKNOWN]. Matching saved bytes is valuable, but official semantic/numerical replay was not established. Use formal provenance reconciliation, never an informal harmlessness waiver.
