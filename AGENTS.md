# Research repository rules

Project: M100 node-level forecasting with simulated FedAvg and centralized controls.
Completed v5: 36 ten-minute inputs, next-hour mean IT power, 979 mapped node clients.
Clients are compute nodes at one facility, not independent data centers.
Real power/temperature training and energy-aware decisions remain future work.

## Read before changing anything

- Read docs/INDEX.md, docs/PRD.md, docs/Design.md, docs/Architecture.md and docs/Features.md.
- Inspect the affected version's protocol, source, config, manifests and receipts.
- User instructions/authorization take precedence. Preserve immutable run evidence;
  overview docs must not override code, saved results or a failed integrity status.
- Use /media/csedept/cse2014/Project/m100_training_template, not the older Desktop copy.

## Research and filesystem safety

- Never silently change datasets, feature order/units, splits, targets, history,
  cohorts/scalers, sampling budgets, FL settings or evaluation/selection protocol.
  Obtain explicit approval where not already authorized in the current session.
- Never delete/overwrite datasets, checkpoints, logs, outputs or historical evidence.
- Check unused destinations and sealed inventories before additive work; use new versions.
- Never alter expected hashes, baselines, timestamps or lock contents to pass a check.
- No GPU training/resume, model/test evaluation or large downloads without explicit
  authorization for that operation. CLI flags alone are not authorization.
- No destructive filesystem commands, Git history rewrites, unapproved dependency
  installation or shared-server configuration changes.
- Preserve failed/partial evidence; resume only the exact supported owned run.
- Stop on unexplained protected mismatches or unresolved material methodology choices.

## Implementation workflow

- Inspect first; explain scope, files, methodology and side effects before approved changes.
- Reuse existing interfaces; keep functions focused and validate temporal/shape/unit boundaries.
- Run appropriate focused tests and necessary non-destructive integrity checks.
- Inspect test/CLI side effects first: some reports infer and some suites use immutable fixtures.
- Document implementation changes and update affected navigation/interface/status guides.
- Report every changed file, actual commands, pass/fail/skip counts and limitations.
- Historical passing logs are not fresh results. Never fabricate metrics or conclusions.

## Claims and gates

- Keep one-hour historical, six-hour v5 power-only and future multi-output tracks separate.
- Existing test periods are observed; seeds share one dataset, not independent datasets.
- No unsupported FL superiority, 24-hour benefit, savings, PUE/SLA improvement,
  carbon prediction, formal privacy, secure aggregation or deployed-controller claims.
- Historical strict preflight remains FAILED; semantic reproduction is UNKNOWN.
  New scoped preservation cannot repair or bypass that status.
- Stop after the approved stage; provide evidence for review before the next gated operation.

Detailed guidance: docs/Agents.md. Tests/recovery: docs/Testing.md, docs/Debugging.md.
