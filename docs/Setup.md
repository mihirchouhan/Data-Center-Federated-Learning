# Verified server setup and safe orientation

Documentation review: 11 November 2026. See [index](INDEX.md) and [evidence and verification](DOCUMENTATION_AUDIT.md).

## Existing environment

Use `/media/csedept/cse2014/Project/m100_training_template/.venv/bin/python`, not an older Desktop copy or accidental Conda interpreter. pyvenv.cfg records Miniforge-derived Python 3.14.7. Distribution metadata was inspected without importing model frameworks:

| Dependency | Observed version | Declaration |
| --- | --- | --- |
| Python | 3.14.7 | Existing pyvenv.cfg |
| PyTorch | 2.14.1+cu130 | requirements.txt: >=2.6,<3 |
| NumPy | 2.5.3 | >=1.24,<3 |
| pandas | 2.3.3 | >=2.2,<3 |
| PyArrow | 23.0.1 | >=16,<24 |
| Matplotlib | 3.11.2 | >=3.8,<4 |
| XGBoost | 3.4.1 | Optional requirements exact pin |
| SciPy | 1.18.1 | Optional requirements exact pin |

These are metadata observations, not proof of current CUDA-driver compatibility or numerical reproduction. Broad core version ranges do not reconstruct every transitive historical dependency. No dependency was installed/upgraded.

## Safe first steps

```bash
cd /media/csedept/cse2014/Project/m100_training_template
.venv/bin/python -B --version
.venv/bin/python -B -m pip list
.venv/bin/python -B -m json.tool configs/experiments/v5_missing_data/gru_gap_policy_v1.json
.venv/bin/python -B -m src.phase2_multioutput.readiness --stage protocol
```

Protocol verification is not a model smoke. The data gate deliberately exits 2 with blockers. Commands are source-checked; experiment CLIs were not run in this task. There is no database setup/migration, .env template, webserver or browser URL. Read [API](API.md) before other stages.

Installation is already complete in the existing venv. Do not reinstall/upgrade it to repair unexplained provenance. For a separately approved new environment, inspect root README wheel guidance, requirements files and recorded full dependency map first; select compatible Python/CPU-or-CUDA wheels explicitly. This guide does not invent a clone URL, blanket platform instructions or a new installation recipe for untested hardware.

## Data and output paths

Original frozen cohort is `output/experiments/v1/frozen`; current v5 uses `output/experiments/v5_missing_data/model_comparison/v1/cohorts`. Its common arrays/scalers are authoritative, not regenerated equivalents. Phase 2's `configs/experiments/phase2_multioutput/v1` is design-only; it is not a ready March model dataset. Before a future authorized run, check both file existence and symlinks, then let the runner exclusively create its output. Never clean an existing directory for reuse.

## GPU and environment variables

An operator may run read-only `nvidia-smi` before an authorized GPU run. It was not run here. Historical NVML driver/library mismatch needs administrator attention, not an agent driver update. Locked v5 requires CUDA with no CPU fallback. Device visibility maps a physical GPU to logical cuda:0; do not edit the plan or assume hardware changes preserve bitwise resume.

| Variable | Actual scope |
| --- | --- |
| CUDA_VISIBLE_DEVICES | Launcher physical UUID → logical 0; empty value for pure CPU synthetic protocol suite |
| CUBLAS_WORKSPACE_CONFIG | v5 accepts :4096:8 or :16:8; deterministic algorithms, disabled TF32 |
| PYTHONDONTWRITEBYTECODE / -B | Avoid protected-source bytecode writes |
| TRAIN_OUTPUT_DIR / TRAIN_DATA_PATH | Original main.py defaults, not v5 overrides |
| CA bundle | Original pipeline --ca-bundle; never disable TLS/hostname validation |

## Smoke checks and common setup errors

The Phase 2 synthetic NumPy suite in [Testing](Testing.md) checks contracts without models/GPU. Existing v5 fits are complete; no duplicate real-data smoke is recommended. Phase 2 real model pilots remain blocked pending adapters/coverage/approval.

Common failures: wrong project root; wrong Python; changed dependencies; unavailable CUDA; NVML mismatch; TLS chain failure; missing required metric; occupied output; changed source/configuration; misaligned UTC IDs; historical preservation preflight. Diagnose via [Debugging](Debugging.md), preserve evidence and never delete locks or regenerate expected hashes.
