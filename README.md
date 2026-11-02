# Portable LSTM/GRU training template

Run the M100 download, preparation, training and evaluation pipeline from a VS Code Remote SSH / VS Code Server terminal. Downloads and training resume after interruptions; generated artifacts go to one persistent output directory. `pipeline.py` uses real Dataset 5 by default. `main.py` also offers a small synthetic deployment check. Synthetic metrics and miniature test fixtures are software checks, not M100 research results.

## Directory structure

```text
m100_training_template/
├── main.py                     # CLI, save_checkpoint(), load_checkpoint(), training
├── pipeline.py                 # Download → prepare → train → test evaluation
├── requirements.txt
├── README.md
├── .gitignore
├── configs/
│   ├── default.json            # Synthetic / prepared NPZ training
│   ├── gpu.json                # Require CUDA for M100 training
│   └── m100.json               # Real Dataset 5 pipeline settings
├── src/
│   ├── __init__.py
│   ├── data.py                 # Synthetic data / prepared NPZ validation
│   ├── download.py             # Resumable download, MD5 and selective extraction
│   ├── preprocess.py           # Node metric alignment / temporal split / scaling
│   ├── model.py                # LSTM / GRU regression model
│   └── io_utils.py             # Atomic writes / run lock
├── tests/
│   ├── test_recovery.py
│   ├── test_download.py
│   ├── test_preprocess.py
│   └── test_pipeline.py
├── tools/
│   └── package_project.py      # Source-only ZIP allowlist
├── data/                       # Runtime cache; excluded from ZIP
│   ├── archives/               # Verified monthly TARs / partial downloads
│   └── extracted/              # Content-specific dataset caches / selected metrics
└── output/                     # Runtime artifacts; excluded from ZIP
    └── m100/                   # pipeline.py's default run directory
        ├── .pipeline.lock
        ├── .run.lock
        ├── run_config.json
        ├── prepared/
        │   ├── dataset.npz
        │   └── preprocessing.json
        ├── stages/
        │   ├── extract_22-01.json
        │   ├── extract_22-02.json
        │   ├── prepare.json
        │   └── completed.json
        ├── checkpoints/
        │   ├── last.pt
        │   ├── previous.pt
        │   └── best.pt
        ├── logs/
        │   ├── pipeline.log
        │   └── train.log
        ├── models/
        │   └── final_weights.pt
        └── results/
            ├── data_audit.json
            ├── history.json
            ├── summary.json
            ├── val_predictions.npz
            ├── test_summary.json
            ├── test_predictions.npz
            ├── loss_curve.png
            └── test_forecast.png
```

The ZIP contains source, configuration and tests only. `pipeline.py` places downloaded archives under `data/archives/` and selectively extracted Parquet files under `data/extracted/<dataset-signature>/`; generated prepared datasets, checkpoints, plots and results go beneath `output/m100/`. Content-specific extraction prevents stale files from a different archive being mixed into a new dataset. A cache OS lock prevents two processes from writing the shared cache; it is released before training. Both base locations are configurable and excluded from the ZIP. `main.py` alone trains synthetic or already prepared NPZ data without downloading/preprocessing; it accepts `--data-path` or `TRAIN_DATA_PATH` and defaults to the `output/` run directory.

## Deploy on the server

Upload `m100_training_template.zip` to the remote server. In the remote VS Code terminal, use a directory with persistent storage and enough space for checkpoints and results. The commands below assume the ZIP is in the current directory.

```bash
unzip m100_training_template.zip
cd m100_training_template
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use Python 3.10 or later. If `venv` is unavailable, request the server administrator's supported Python environment. For a CPU smoke test, install:

```bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import torch; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

For an NVIDIA GPU run, check `nvidia-smi`, then choose the Linux/Pip/Python CUDA command from the [official PyTorch installer](https://pytorch.org/get-started/locally/) that matches the server's supported driver. Install that PyTorch wheel before `python -m pip install -r requirements.txt`. The template needs `torch` only; `torchvision` and `torchaudio` are optional installer components. `--device cuda` fails clearly if CUDA is unavailable; `--device auto` selects an available GPU or falls back to CPU. Confirm the selected device in the run's `logs/train.log`.

Both M100 configurations now select `configs/gpu.json`, which requires `device="cuda"`. Download and preparation use the CPU; model training requires an accessible NVIDIA GPU and stops if one is unavailable. `preprocessing.include_gpu_utilization=false` excludes a missing historical sensor feature; it does not disable GPU model training.

On this server, two NVIDIA RTX 6000 Ada GPUs were detected in `/proc/driver/nvidia/gpus/`. The environment originally contained `torch 2.14.1+cpu`. The [official CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/) provides the matching CUDA build for Python 3.14:

```bash
python -m pip install --upgrade 'torch==2.14.1+cu130' --index-url https://download.pytorch.org/whl/cu130
```

CUDA 13.x requires a compatible NVIDIA driver from version 580 onward; see [NVIDIA's compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). A sufficiently recent version alone does not repair inconsistent driver components. The loaded NVIDIA kernel module was **595.91.07**, while `libcuda.so.1` and `libnvidia-ml.so.1` resolved to **580.178.04**. The server administrator must align the loaded module and userspace libraries and arrange a reboot if needed, following [Ubuntu's driver mismatch guidance](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/#driver-library-version-mismatch-error). Installing PyTorch does not repair system drivers.

After the driver is repaired, verify GPU access from the activated project environment:

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda); assert torch.cuda.is_available(), 'GPU unavailable'; print(torch.cuda.get_device_name(0)); x = torch.ones(2, device='cuda'); print(x + x)"
```

For a fresh ten-epoch GPU run using the existing downloaded data, use a separate output directory:

```bash
python pipeline.py --config configs/m100.json --device cuda --data-dir data --output-dir output/m100_gpu
```

This reuses the archive/extraction cache, prepares the same dataset in the new output directory, trains on CUDA and exports held-out evaluation. Check `output/m100_gpu/logs/train.log` for `Device=cuda`. Rerun the same command to resume an interrupted GPU run. The completed ten-epoch CPU baseline remains in `output/m100`; restarting that baseline with ten epochs would only export/evaluate its completed model. To extend its learned weights with ten additional GPU epochs instead, run:

```bash
python pipeline.py --config configs/m100.json --stage train --device cuda --epochs 20 --data-dir data --output-dir output/m100
```

cc## Continue from the persistent project location

The project now lives at `/media/csedept/cse2014/Project/m100_training_template`. Open this folder in VS Code and select `.venv/bin/python` from this folder as the Python interpreter. In a new terminal, activate the environment from the new location:

```bash
cd /media/csedept/cse2014/Project/m100_training_template
source .venv/bin/activate
python -c "import sys; print(sys.executable); print(sys.prefix)"
```

Both printed paths should refer to the new project location. The copied environment's activation scripts and console-script interpreter paths have been updated for this move on the same server; its base Miniforge Python installation remains the same.

Resume the original January–February download with explicit cache/output locations so older terminal environment variables cannot redirect progress elsewhere:

```bash
python pipeline.py --config configs/m100.json --stage download --data-dir data --output-dir output/m100
```

For the January-only experiment, use:

```bash
python pipeline.py --config configs/m100_january.json --data-dir data --output-dir output/m100_january
```

The January-only command downloads, prepares, trains and evaluates. Keep the copied `data/` and `output/` folders: partial downloads, verified archives, preparation receipts and checkpoints are reused from them. For an existing prepared/training run, continue with its original configuration and corresponding copied output directory. Relative paths resolve from this new project directory. Stop any process launched from the old project before resuming from the new one; a process already running keeps its original paths.

## Download M100 and run the full pipeline

The default configuration selects [M100 Dataset 5, January–February 2022](https://zenodo.org/records/7589942), from the dataset's [official ExaData project](https://gitlab.com/ecs-lab/exadata). Its monthly archives total approximately **24 GB**: January is 15.8 GB and February is 8.2 GB. These archives are downloaded only when you launch the pipeline; they are not included in this project ZIP. Reserve space for both archives, selected extraction and generated outputs. Check storage and server download policies first:

```bash
df -h .
python pipeline.py --config configs/m100.json
```

That command runs download, preparation, training and final held-out evaluation. It saves progress under `output/m100/`. After a server or network interruption, run the same command again: existing downloads are checked, partial downloads resume using HTTP Range where supported, completed stages are reused when their fingerprints match, and training restores its latest checkpoint. The default downloads are verified against the official archive MD5 checksums before extraction.

Run individual stages when you want to inspect the data audit before training:

```bash
python pipeline.py --config configs/m100.json --stage download
python pipeline.py --config configs/m100.json --stage prepare
python pipeline.py --config configs/m100.json --stage train
```

If a download stops with `CERTIFICATE_VERIFY_FAILED` / `self-signed certificate in certificate chain`, keep `data/archives/*.tar.part`: the next download attempt resumes from the saved byte offset when the server supports HTTP Range. A read timeout is retried automatically, but certificate validation failures stop with the partial-file path and saved byte count. If the connection verifies again, rerun the same download command above.

For a network that requires an institution/proxy CA, obtain its trusted PEM CA certificate from your administrator and add it with:

```bash
python pipeline.py --config configs/m100.json --stage download --ca-bundle /path/to/institution-ca.pem
```

`--ca-bundle` adds certificates to Python's default trust store and keeps certificate/hostname verification enabled. It follows the same project-relative path rules as other CLI paths. Python also supports `SSL_CERT_FILE` and `SSL_CERT_DIR` to configure its default CA locations; `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` do not configure this standard-library downloader. In a Conda/Miniforge-based environment, Python's default CA file can differ from the OS bundle. On Ubuntu, you can explicitly add the OS bundle with `--ca-bundle /etc/ssl/certs/ca-certificates.crt`; this helps only if that bundle trusts the CA presented on your network. See [Python's TLS context documentation](https://docs.python.org/3/library/ssl.html#ssl.create_default_context).

`--stage all` is the default. Preparation uses files already downloaded/extracted, and training requires prepared data. Use persistent server locations through environment variables or CLI options:

```bash
export M100_DATA_DIR="data"
export TRAIN_OUTPUT_DIR="output/m100_run01"
python pipeline.py --config configs/m100.json --device auto --epochs 10
```

`--data-dir` overrides `M100_DATA_DIR`, and `--output-dir` overrides `TRAIN_OUTPUT_DIR`. Relative CLI, environment and configuration paths resolve from the project directory containing the scripts, even when launched from another directory. Supply the server's actual persistent mount through these settings if needed. Keep both the data cache and output directory on persistent storage. A separate output directory isolates another experiment; a data cache may be reused. Changing preprocessing settings requires a new compatible prepared run rather than silently reusing stale output.

In `configs/m100.json`, `preprocessing.node` selects anonymized node ID `"47"`, chosen after checking its CPU telemetry coverage in the downloaded January–February archives. Node `"0"` has no GPU-utilization rows, and its CPU/memory data end on January 27, so it cannot produce the original two-month validation/test windows. `preprocessing.months` must match the months listed in `dataset.archives`. To run one month, edit both lists together in a copied configuration. The default processes January and February. `training_config` selects the trainer configuration file.

`preprocessing.include_gpu_utilization` is explicitly `false` in both supplied M100 configurations. January's archive has no `Gpu0_gpu_utilization` through `Gpu3_gpu_utilization` partitions; February contains only short, all-zero series for a subset of nodes. The default preparation therefore omits the GPU feature and uses nine observed input features. This is a configuration choice recorded in the audit and preprocessing metadata; missing GPU readings are never replaced with zeros. Setting the option to `true` (also the behavior for older configurations that omit it) requires all four GPU series and complete windows in every chronological split. Only enable it for a dataset/node with verified coverage. Historical dataset GPU telemetry is separate from the NVIDIA driver used to train the model on this server.

For an earlier first experiment, `configs/m100_january.json` selects January only and reuses the same January archive/partial download in the data cache. If the two-month downloader is running, stop it with Ctrl+C before switching. Run:

```bash
python pipeline.py --config configs/m100_january.json --output-dir output/m100_january
```

This finishes the January download, prepares its data, trains and evaluates in a separate output directory. It avoids waiting for February's approximately 8.2 GB archive; it does not increase the transfer speed. Its time splits and results cover January only. For the original January–February experiment, use `configs/m100.json` and a separate compatible output directory; the verified January archive can be reused.

The initial experiment uses **one configurable node**, ten required raw metric partitions, ten-minute alignment, six history bins (one hour), and the following six bins' mean node power (next hour). For the inspected node, 24 and 12 hours produced no validation/test windows; six hours produced no test windows. Three hours worked but left only 19 test windows. One hour provides 4,442 training, 265 validation and 87 test windows for the January–February node-47 run. This is an explicit baseline configuration change, rather than a claim that the archives contain continuous readings for every sensor. Increase the history only after checking the resulting audit and split coverage. The default input metrics are:

- IPMI: `total_power` and node-inlet `ambient` temperature.
- Ganglia: `cpu_user`, `cpu_system`, `cpu_idle`, `cpu_wio`, `mem_free`, and `mem_total`.
- Weather: outdoor `temp` and `humidity`.

Extraction retains these partitions and any available GPU 0–3 utilization partitions for optional use; it does not unpack the entire dataset. The default ten raw metrics become nine input features: historical power in kW, four CPU percentages, used-memory fraction, node-inlet temperature, outdoor temperature and outdoor humidity. This gives seven operational and two weather features. Enabling GPU utilization adds its four raw metrics and their mean as a tenth feature. Prepared data record the actual feature names and dimensions; the trainer constructs its model from those prepared dimensions. The regression target is next-hour mean node power in kW. The pipeline uses fixed chronological boundaries: 70% training, 15% validation and 15% held-out test. Normalization is fit only from training data. It writes `results/data_audit.json`, `prepared/preprocessing.json` and the train/validation/test arrays in `prepared/dataset.npz`, then exports held-out predictions and error metrics in kW.

Inspect `results/data_audit.json` before treating the run as an experiment: it records selected-node rows, timestamps, sampling/coverage and missingness. The configured node and metrics must be present and produce enough valid windows; the pipeline fails clearly when those prerequisites do not hold, including an absent default node ID. A failure before all metrics are read may leave only a log rather than a completed audit. A ten-minute model grid does not prove that every sensor natively sampled at that interval. This is a single-facility, single-node LSTM/GRU forecasting baseline. Federated training, XGBoost comparisons and cooling/decision optimization remain research extensions; this pipeline provides the training/deployment basis for them.

The downloader/preprocessor is tested with miniature fixtures that mirror the official partition/schema layout. The downloaded January–February archives were also inspected to select the default node and exclude unavailable GPU utilization. Check the real-data audit for the resulting sample counts and coverage before interpreting model results. Prepared artifacts and final evaluation remain under the selected output directory; raw archive/extraction caches remain under the selected data directory.

## Check training and recovery

To train one shared GPU model across all usable nodes in the downloaded January–February archives, use the separate all-node configuration and output directory:

```bash
python pipeline.py --config configs/m100_all_nodes.json --device cuda --data-dir data --output-dir output/m100_all_nodes
```

`configs/m100_all_nodes.json` uses `preprocessing.nodes="all"`; a list of node ID strings can select a subset. Use either `node` for the original single-node path or `nodes` for the shared model. The all-node path scans each selected metric once, retains separate node histories, and uses common calendar train/validation/test cuts. All input features and future power bins must have valid sensor coverage. A node must have at least ten usable windows in each split; the audit records every excluded node and its reason. This covers usable readings from the downloaded two months, rather than every archive/month offered by the publisher. Missing sensors are not replaced with zeros. Longer histories should be chosen only after checking the coverage audit.

The shared model includes the nine operational/weather features, four known UTC calendar features, and a categorical node index. The index feeds a learned eight-dimensional embedding; it is not treated as an ordinal measurement or standardized. Other features and the target use pooled training-only scaling. Predictions still mean each node's average IT power over the next hour. This follows the shared-model forecasting approach described by [Montero-Manso and Hyndman](https://arxiv.org/abs/2008.00444); time-based evaluation avoids training on future observations, as explained in the [time-series validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

`configs/all_nodes_gpu.json` requires CUDA, uses two 64-unit LSTM layers, batches of 512, and at most 30 epochs. `early_stopping_patience=5` stops after five epochs without a lower validation MSE. `forecast_mode="residual"` begins at the persistence forecast and learns a correction. The untrained persistence model is also a validation candidate. `prediction_weights="best"` selects the weights with the lowest validation MSE, including that initial candidate; the final test split is used only for reporting. Accuracy improvements are measured, not guaranteed. The original training configurations retain their final-epoch export behavior, while this new experiment exports its selected weights in `models/final_weights.pt` and records which weights were selected. The latest resume checkpoint still preserves the final optimizer/training state so interruption recovery and epoch extension remain valid. A rerun of an early-stopped experiment exports the same selected weights; increasing its epoch limit resets the patience window. `--early-stopping-patience 0` disables early stopping when intentionally extending a run.

`results/test_summary.json` reports pooled and average-per-node MAE plus persistence comparisons. `results/test_by_node.json` reports each node separately. `results/test_predictions.npz` includes node IDs and UTC forecast origins. The forecast plot shows one node at a time instead of connecting predictions from unrelated nodes. Training and test inference use the run's selected GPU. Existing `output/m100`, `output/m100_run01` and `output/m100_january` experiments are preserved. Rerun the same all-node command after an interruption to resume its checkpoint.

Run the automated checks:

```bash
python -m unittest discover -s tests -v
```

Use a new output directory for this smoke test. The first command stops after step 3 and leaves a checkpoint; the second resumes that same run. Omit `--stop-after-step` when resuming.

```bash
python main.py --config configs/default.json --device cpu --epochs 2 --output-dir output/smoke --checkpoint-every-steps 1 --stop-after-step 3
python main.py --config configs/default.json --device cpu --epochs 2 --output-dir output/smoke --checkpoint-every-steps 1
```

Inspect `output/smoke/logs/train.log` for resumed progress, and `output/smoke/results/summary.json` for the final status and validation results. The `output/smoke/` contents follow the same structure shown above. The stop option simulates a controlled interruption; recovery tests also cover checkpoint consistency.

The default `num_threads=1` keeps CPU smoke tests modest; adjust it in your copied configuration for the server's assigned CPU resources. For a normal synthetic training run:

```bash
python main.py --config configs/default.json
```

Run the same command again after an interruption. Startup checks for `last.pt` and restores model weights, optimizer state, epoch/batch position, loss accumulation, random-number states and data/config fingerprints. Best validation loss and its model weights are retained in each resume checkpoint. If the latest checkpoint is unreadable, the retained previous checkpoint provides a recovery point. A completed run exports its artifacts without repeating optimization. Increase `--epochs` to extend that same compatible run.

## Train directly from an existing prepared dataset

Use `main.py` directly if preprocessing is already complete. Supply one numeric NumPy `.npz` file containing these four keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X_train` | `[N_train, T, F]` | Training sequences |
| `y_train` | `[N_train, K]` | Training regression targets |
| `X_val` | `[N_val, T, F]` | Validation sequences |
| `y_val` | `[N_val, K]` | Validation regression targets |

Use finite numeric values and consistent sequence, feature and target dimensions. No object arrays are supported. Set `sequence_length=T`, `input_size=F` and `target_size=K` in a copied configuration. The default `T=144` can represent 24 hours at ten-minute sampling; it has that interpretation only if preprocessing establishes that interval. Define next-hour targets during preprocessing. The template does not infer timestamps, horizons, units or metric meaning from array shapes.

For research use, split by time before fitting scalers; fit normalization on training data only and preserve preprocessing metadata separately. `main.py` optimizes/evaluates using only the train/validation keys; held-out `X_test` and `y_test` keys, when present, are reserved for the pipeline's final evaluation. The direct trainer's exported validation predictions stay in the supplied target scale, so physical-unit errors require the preprocessing inverse transform. The M100 pipeline performs that inverse transform for its held-out power results.

Set paths in the current shell rather than editing source code:

```bash
export TRAIN_DATA_PATH="../prepared_data/m100_windows.npz"
export TRAIN_OUTPUT_DIR="output/m100_lstm_run01"
python main.py --config configs/default.json --device auto
```

Replace the example dataset location with its actual server path and adjust the copied configuration to match. Relative CLI, environment and configuration paths resolve from the project directory containing `main.py`. An absolute path may be supplied through either environment variable when a persistent server mount is required. CLI `--data-path` and `--output-dir` override their environment variables.

## Keep runs isolated and recoverable

- Save checkpoints every configured number of training steps and epochs, plus an exit checkpoint on a controlled stop. `last.pt` tracks progress; `previous.pt` retains the prior save; `best.pt` tracks the best validation loss.
- Checkpoint files are written to temporary files in the same directory and atomically replaced. An abrupt power failure or process kill can lose work since the last successful save; those steps are replayed from the saved state.
- The output directory uses OS locks (`.pipeline.lock` for pipeline stages, `.run.lock` for training) to reject concurrent writers. Process death releases the OS lock; a lock file may remain on disk and does not by itself mean a run is active. Never remove a lock file while its process is running.
- Resume fingerprints the full dataset contents and rejects changed data or incompatible model/training settings. Epoch count may be increased; the device and checkpoint intervals may be changed. Use a new directory to change the model, learning rate, batch size, seed or data.
- `--fresh` requires a new or empty run directory and never deletes existing checkpoints. For example: `python main.py --config configs/default.json --output-dir output/another_run --fresh`.
- Use persistent disk storage for `TRAIN_OUTPUT_DIR` and `M100_DATA_DIR`. Local temporary disks, machine loss and storage corruption require a separate backup policy. Exact bitwise replay across different devices or PyTorch versions is not guaranteed.
- Checkpoint loading uses PyTorch's restricted `weights_only=True` mode, with RNG state represented as primitive values/tensors. Keep checkpoint provenance: load files generated by this project and controlled by you rather than replacing them with an unknown `.pt` download.

## Run beyond a terminal session

On a server that permits persistent interactive jobs, a `tmux` session keeps training alive when VS Code disconnects:

```bash
tmux new -s m100-training
source .venv/bin/activate
python pipeline.py --config configs/m100.json --output-dir output/long_run
```

Detach with `Ctrl+B`, then `D`; reconnect with `tmux attach -t m100-training`. If the college server uses a scheduler, submit the same command through its approved job script instead.

Alternatively, redirect console output into the run's log directory:

```bash
mkdir -p output/long_run/logs
nohup .venv/bin/python -u pipeline.py --config configs/m100.json --output-dir output/long_run > output/long_run/logs/console.log 2>&1 &
```

`tmux` and `nohup` handle terminal disconnections. Checkpoint recovery handles process/server interruptions: launch the same command again when the server becomes available. A server outage is not automatically restarted by this template.

## Outputs and packaging

During training, progress goes to `logs/train.log`, `results/history.json` and checkpoints. On successful completion, `models/final_weights.pt`, `results/summary.json` and `results/val_predictions.npz` are exported beneath the same output directory. `run_config.json` records the effective settings. The full pipeline also writes `logs/pipeline.log`, `prepared/dataset.npz`, `prepared/preprocessing.json`, `results/data_audit.json`, stage receipts and held-out evaluation outputs shown in the tree. `results/loss_curve.png` shows scaled-target train/validation MSE; `results/test_forecast.png` shows held-out node power and forecasts in kW.

The held-out evaluation uses the predetermined **final epoch** weights in `models/final_weights.pt`, with a persistence baseline reported in `results/test_summary.json`. `checkpoints/best.pt` separately preserves the best validation checkpoint. The test set is evaluated after optimization and is not used to select the model. Exported weights need the corresponding model architecture/configuration; `last.pt` is the full resume checkpoint.

Rebuild the source distribution from inside the project using its explicit source-file allowlist:

```bash
python tools/package_project.py
```

The default archive is `../m100_training_template.zip`. The packager includes source, README, requirements, configuration and tests; it excludes runtime output, environments and datasets even if they are present in the project directory. The prebuilt distribution ZIP follows the same rule.

## Frozen LSTM / GRU / XGBoost experiments

For the separate January–February all-node model/weather comparison, immutable manifests, pilot measurements and commands, see [EXPERIMENTS.md](EXPERIMENTS.md). New experiments use `experiments.py` and `output/experiments/v1/`; historical runs stay in their existing directories. Fitting/validation and held-out reporting are separate commands. No FedAvg is included.
