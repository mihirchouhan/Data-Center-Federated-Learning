# Guidance for research software contributors

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

The concise repository rulebook is [root AGENTS.md](../AGENTS.md). This guidance explains workflow; neither overrides current user authorization or immutable run evidence.

## Inspect before editing

Read [PRD](PRD.md), [Design](Design.md), [Architecture](Architecture.md), [Features](Features.md), then the exact versioned protocol/source/configuration/receipts for the affected track. Check destinations, symlinks, inventories and known failures. Use rg to discover code; don't select an old Desktop copy or infer identity from folder spelling.

Additive files inside sealed directories can still break exact inventory checks. Use unused approved versions and preserve failures. Do not edit source hashes or receipts to make newer code look historical.

## Coding and research conventions

Reuse existing interfaces, keep functions focused, validate temporal/shape/unit/count boundaries, and explain non-obvious aggregation/chronology logic. Follow existing Python/JSON conventions. Document public CLI side effects and failure behavior. Do not introduce frameworks/dependencies, shared-server changes or destructive operations without authorization.

Dataset/months, feature order/units, targets, history, cohort/eligibility, splits, normalization, weather policy, sampling budget, FL aggregation and checkpoint selection affect the research question. Explain and obtain approval for changes outside the session's existing authorization. Do not repeatedly ask for routine already-authorized work.

## Execution and evidence

- No GPU fitting/resume, large transfer or model/test inference without explicit authorization for that operation.
- Separate synthetic checks, coverage audits, validation-only pilots, fits, locks, final evaluation and saved-artifact reports.
- Inspect code before assuming verify/report/plan/dry-run is read-only. Some verifiers initialize/reload CPU tensors; some reports run inference.
- Use unused destinations; resume only with an exact supported owned manifest. Preserve partials and failed evidence.
- Never change lock PID/timestamp/expected hash/baseline to pass a check.
- Never publish credentials or imply privacy guarantees from local-data interfaces.

For approved changes, run focused tests and necessary preservation checks. Do not blindly rerun historical suites with occupied fixtures or real-data writes. Report exact commands, pass/fail/skip counts, changed paths, preservation scope and limitations. A prior passing log is historical evidence, not a fresh result.

Update affected interface/status/design docs after implementation, retaining sealed originals. Cite numerical evidence. Use the index's six status labels. Stop on unexplained protected mismatches or unresolved material design decisions.

Keep claims precise: one-facility simulated node clients; observed test periods; seeds on one dataset; no measured savings, general FL superiority or formal privacy. Current v5 is power-only; Phase 2 real two-output execution remains blocked.
