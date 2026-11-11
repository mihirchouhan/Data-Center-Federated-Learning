# Documentation-oriented project overview

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

**Federated Multi-Output Forecasting and Energy-Aware Decision Framework for Distributed Data Centers**

This repository studies reproducible node IT-power forecasting with simulated FedAvg and centralized controls. Completed v5 uses six hours of fourteen-channel telemetry to predict next-hour mean power. The title includes later power/temperature forecasting and energy-aware decisions, which are not completed experiments.

## Start here

1. Read [INDEX](INDEX.md), [PRD](PRD.md) and [Design](Design.md).
2. Inspect [Architecture](Architecture.md), [storage](Database.md) and [CLI](API.md).
3. Use the existing server venv and safe [Setup](Setup.md) commands; don't run historical all/fit/report/download commands for orientation.
4. Review [Testing](Testing.md), [Debugging](Debugging.md) and [Plan](Plan.md) before another experiment.

[IMPLEMENTED] Causal preparation, frozen mapping/scalers, residual GRU, local training-data API, count-weighted FedAvg, matched centralized SGD, input-only bounded forward-fill, fixed sampled-pass budgets, immutable locks and saved-prediction evaluation/reporting.

[PARTIALLY IMPLEMENTED] Phase 2 two-output contracts/design and synthetic checks. Actual March bridge, eligibility/scalers, training/test adapters and pilot are [BLOCKED]. Controlled history/environmental studies and energy-aware decisions are [PLANNED].

## Current evidence and architecture

v5 has 24 completed training identities (two methods × four policies × three seeds) and saved locked test results for 6,590 common windows/570 nodes. Mapping includes 979 training-node clients from one facility. See the [saved v5 report](../output/experiments/v5_missing_data/model_comparison/v1/results_analysis/COMPLETE_V5_RESULTS_REPORT.md). The older one-hour test has 75,566 windows/979 nodes; scores/cohorts are not interchangeable.

The system uses Python/PyTorch/NumPy/Arrow and optional native XGBoost, explicit CLI stages and file ledgers on Linux. It has no traditional database, REST service, application login or project MCP server. See [Integration](Integration.md), [Authentication](Authentication.md) and [MCP](MCP.md).

## Limitations and roadmap

Historical strict preservation preflight is FAILED; semantic/numerical reproduction is [UNKNOWN]. New scoped byte checks do not repair that status. Sensor calibration, current March archive success/coverage and multi-output runtime remain unverified. The test period has already been observed, and seeds share one dataset. Simulated clients are not independent physical data centers or a privacy guarantee. Prediction proves neither measured savings nor an operational controller.

Next gates are verified acquisition, approved read-only schema/coverage, new-scope provenance acceptance, real adapters and a separately approved validation-only pilot. No experiment is launched by these docs. Root [README](../README.md) and sealed histories remain intact. [AGENTS](../AGENTS.md) governs contributions. A root LICENSE file was not present in the inspected inventory; licensing is [UNKNOWN], not inferred.
