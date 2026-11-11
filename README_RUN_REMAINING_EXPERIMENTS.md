# README: sequential remaining v5 experiments

`run_remaining_experiments.sh` automates only v5 **main fitting and read-only verification**. It does not freeze data, reconstruct raw inputs, evaluate the test split, generate reports, change configuration, or launch the built-in matrix command. Creating or syntax-checking the script does not train anything.

## Repository and fixed inputs

The active project is `/media/csedept/cse2014/Project/m100_training_template`. The older Desktop project copy does not contain the v5 runner. The workspace copy of this launcher targets the active project too; it does not depend on the terminal's current directory.

- Python: `.venv/bin/python -B`
- Plan: `configs/experiments/v5_missing_data/gru_gap_policy_v1.json`
- Cohorts: `output/experiments/v5_missing_data/model_comparison/v1/cohorts`
- Outputs: `output/experiments/v5_missing_data/model_comparison/v1/full_runs`
- `CUDA_VISIBLE_DEVICES=GPU-4d239ff8-8f62-c45c-7b52-23d05bc69842` is set for the entire launcher and every child. The unchanged plan's `cuda:0` is the logical device for this visible UUID. There is no automatic GPU selection or fallback.

This is an additive README. Historical `README.md`, all protected sources, the plan, frozen data and existing experiment results remain unchanged.

## Queue and naming

The two methods are `centralized_sgd` and `fedavg`, with seeds 42, 43 and 44. Folder policy `none` is translated to the runner's required `--policy no_fill`. New folders are exactly `{method}_{policy}_seed{seed}_v1`, for example `centralized_sgd_none_seed42_v1`.

The six requested historical exclusions are discovered by **manifest identity**, preserving legacy `centralized_...` folder names:

- centralized_sgd / ff20 / 42
- fedavg / ff20 / 42
- centralized_sgd / ff10 / 42
- fedavg / ff10 / 42
- centralized_sgd / ff10 / 43
- fedavg / ff10 / 43

Each of these six must exist as one unique completed 200-round main run and pass a fresh `--stage verify` returning `verified_complete` before any new fit starts. A directory or completion marker alone is never sufficient. Missing, duplicate, incompatible, corrupted or incomplete historical exclusions stop the launcher.

The 18-entry pending queue is deterministic:

1. FF10 seed44: centralized SGD, then FedAvg.
2. No-fill seeds42/43/44: both methods per seed.
3. FF20 seeds43/44: both methods per seed.
4. FF30 seeds42/43/44: both methods per seed.

For every new/partial fit, the launcher waits for successful `--stage fit`, requires a terminal JSON result showing 200 completed rounds, and automatically runs `--stage verify`. It requires `verified_complete`, completed round 200 and `model_forward_or_scoring: false` before proceeding. A zero exit code with an incomplete or malformed result also stops the queue.

## Preview safely: no training or output writes

```bash
cd /media/csedept/cse2014/Project/m100_training_template
bash run_remaining_experiments.sh --dry-run
```

The dry run checks the locked plan, source/evidence hashes and manifest/completion metadata, then prints six **skip candidates** and the queue. It imports no model code, runs no pipeline stage, queries no GPU and creates no logs or directories. Skip candidates still require real verification during execution. For an already started queue:

```bash
bash run_remaining_experiments.sh --dry-run --resume
```

## Start the queue manually

**The following command starts GPU model fitting. Execute it yourself only when ready.** Run it in the foreground in the college server terminal, or in an existing persistent terminal session. It does not detach itself.

```bash
cd /media/csedept/cse2014/Project/m100_training_template
bash run_remaining_experiments.sh
```

The script uses one child at a time and immediately waits for it. The child is placed in the background only to allow reliable signal forwarding; no second experiment is started concurrently. An advisory `flock` prevents two copies of this launcher from running against the same output root. This lock does not control unrelated manually launched programs or existing GPU watcher scripts; do not launch those alongside this queue.

## Resume after interruption or failure

First inspect the most recent master and per-stage logs. Then preview and resume:

```bash
cd /media/csedept/cse2014/Project/m100_training_template
bash run_remaining_experiments.sh --dry-run --resume
bash run_remaining_experiments.sh --resume
```

`--resume` rechecks every existing run's method, policy, seed, main-run kind, cap and locked configuration, then calls the real verifier:

- `verified_complete`: skip without fitting.
- `verified_partial`: pass `--resume` to the existing single-fit runner, then verify again after completion.
- Verification error, unrecognized status or incompatible provenance: stop; preserve the directory.

Without `--resume`, any existing non-excluded experiment directory causes a preflight failure. No directory is overwritten, deleted, renamed, silently repaired or automatically replaced. Duplicate identities require review; the launcher does not choose a model based on metrics or newest timestamps.

If interrupted before a complete round checkpoint exists, the verifier may refuse resume. Preserve the partial directory and request review; do not delete it or edit hashes. The launcher intentionally does not restart an unverified run from scratch. A disappeared inventoried run also stops execution. Actual runner resume requires the recorded software, source, configuration and device-name bindings to match.

Ctrl-C, TERM and HUP stop the active child and retain atomic checkpoints, any `.pending-*` directories and all logs. A forcibly killed parent may leave its one child running; its inherited queue lock prevents a second launcher from starting until it exits. Never remove a lock file to bypass an active process.

## Logs and progress

Each execution creates a new, non-conflicting directory under:

```text
output/experiments/v5_missing_data/model_comparison/v1/full_runs/_automation_logs/run_<UTC timestamp>_<unique suffix>/
    master.log
    progress.tsv
    <method>_<policy>_seed<seed>.fit.<unique suffix>.log
    <method>_<policy>_seed<seed>.verify.<unique suffix>.log
```

`master.log` contains timestamped status messages, exact commands, GPU UUID and log paths. `progress.tsv` records starts, outcomes, verified skips and completed entries. Full runner stdout/stderr goes to each per-stage log, rather than flooding the terminal. Open the printed path in VS Code, or use `tail -f` with that actual path from another terminal.

Every execution and stage uses a newly allocated path; previous logs are preserved. Logs are outside individual fit directories, because the runner must create each fit directory itself. Logs and progress are not trusted as proof of completion; fresh runner verification controls every skip.

## Validation and limitations

The script's syntax is checked with:

```bash
bash -n run_remaining_experiments.sh
```

During creation, the six existing main runs were individually verified with the real `--stage verify`; all returned `verified_complete` at round 200 with `model_forward_or_scoring: false`. No new model fit or inference was run. Dry-run and isolated mock-CLI tests check sequencing, verification gates, failures, policy translation, existing-directory handling, interruption, resume and locking. Mock fit calls are synthetic control-flow tests, not GPU/model experiments.

Creation checks (3 November 2026): Bash syntax passed; the real dry run printed 18 pending entries beginning with centralized-SGD FF10 seed44 and FedAvg FF10 seed44; all **nine mock tests passed**. The mock harness is `/tmp/m100-automation-tests-YrQvgN/test_automation.py`; the six real read-only verification results are `/tmp/m100-automation-verification-vzhcvgm_/results.json`. These temporary records are not historical experiment receipts. ShellCheck was not installed, so no ShellCheck result is claimed.

The launcher inherits the existing verifier's scope: it checks saved source/data/checkpoint hashes and state consistency, without rerunning predictions. It does not strengthen historical partial-receipt anchoring, prove GPU numerical resume equivalence, or recover lost resource measurements. The locked plan and its expected hashes must not be changed to resolve a failure. The launcher's GPU UUID in the new logs supplements the runner's logical-device/model-name manifest.

`M100_PROJECT_ROOT` can point to an explicitly reviewed relocation or an isolated mock fixture. Normal use should leave it unset; the launcher defaults to the active `/media` repository and still requires the exact locked plan/input hashes.
