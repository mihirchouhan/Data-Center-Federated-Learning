# Research documentation index

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

**Federated Multi-Output Forecasting and Energy-Aware Decision Framework for Distributed Data Centers**

The title describes the research direction. Completed current v5 predicts **one output: next-hour mean node IT power**, using six hours of telemetry. Two-output power/temperature forecasting and energy-aware decisions are later work. The 979 mapped compute nodes are simulated clients from **one facility**, not 979 physical data centers.

## Status and authority

| Label | Meaning |
| --- | --- |
| [IMPLEMENTED] | Code/artifact exists; cited evidence defines the verified scope |
| [PARTIALLY IMPLEMENTED] | Interfaces exist, but required execution/components are missing |
| [PLANNED] | Proposed work; no outcome implied |
| [BLOCKED] | Evidence, implementation or authorization gate unmet |
| [UNKNOWN] | Not established by this review |
| [NOT APPLICABLE] | No corresponding research-application component |

Authority: current user instructions/authorization; exact versioned protocol and immutable artifacts for the run; inspected source/configuration; these navigation documents; older explanatory prose. Resolve conflicts explicitly. Overview documentation never changes a receipt or failed integrity status. Implemented verification code does not imply every historical run passes it now.

| Track | State | Evidence |
| --- | --- | --- |
| v1 | [IMPLEMENTED] one-hour LSTM/GRU/XGBoost/weather comparison | [Original guide](../EXPERIMENTS.md) |
| v2/v3 | [IMPLEMENTED] one-hour FedAvg/matched centralized SGD | [Historical thesis](thesis/v1/THESIS_METHODS_RESULTS.md) |
| v5 | [IMPLEMENTED] 24 six-hour power-only fits and completed locked test outputs | [Saved v5 report](../output/experiments/v5_missing_data/model_comparison/v1/results_analysis/COMPLETE_V5_RESULTS_REPORT.md) |
| Historical preservation preflight | [BLOCKED] FAILED; numerical/semantic reproduction [UNKNOWN] | [Investigation](../output/phase1_integrity_review/integrity_review.md) |
| Phase 2 | [PARTIALLY IMPLEMENTED] two-output design/synthetic contracts; execution [BLOCKED] | [Frozen protocol](experiments/phase2_multioutput/v1/PROTOCOL.md) |
| Energy-aware decisions | [PLANNED] no controller or measured savings | [Roadmap](Plan.md) |

## Reading order and navigation

1. [README](README.md), [PRD](PRD.md): orientation, requirements and research scope.
2. [Design](Design.md), [Architecture](Architecture.md): methodology, model and modules.
3. [Database / storage](Database.md), [API / CLI](API.md): interfaces and artifacts.
4. [Setup](Setup.md), [Features](Features.md), [Testing](Testing.md), [Debugging](Debugging.md).
5. [Plan](Plan.md), [Integration](Integration.md), [Authentication](Authentication.md), [MCP](MCP.md).
6. [Contributor guidance](Agents.md), root [AGENTS.md](../AGENTS.md), [documentation audit](DOCUMENTATION_AUDIT.md).

## Research workflow and diagrams

Inspect evidence → declare protocol → coverage audit → preserve inputs → synthetic checks → separately approved pilot → separately approved fits → validation selection → immutable lock → separately approved test evaluation → saved-artifact analysis/thesis. Failed gates stop the workflow; documentation grants no experiment authorization.

Mermaid sources: [system](diagrams/system_architecture.mmd), [research workflow](diagrams/research_workflow.mmd), [data pipeline](diagrams/data_pipeline.mmd), [FedAvg](diagrams/federated_learning.mmd), [experiment workflow](diagrams/experiment_workflow.mmd). Important diagrams are embedded in Architecture.

The researcher/approved contributors maintain these guides alongside interface/status changes. Preserve sealed historical documentation and use new versions for changed experiments. Do not rewrite dated plans to imply later work had already been executed.
