bbcM100 frozen model and weather comparison, version 1
=================================================

Use this runner for the new comparison. Historical `pipeline.py` runs and their artifacts are preserved. No FedAvg is included. Full comparisons have not been launched by this implementation.

The data contract references the reviewed January–February 2022 prepared NPZ (SHA-256 `b0b34c9b3af706d1f0062278ece1eb2a865c2e7f6dde03bb15c8257c4c7101e9`). Freeze makes read-only NPY copies for memory mapping, without re-extraction, filtering, normalization or window generation. It records hashes of X, y, node and origin arrays for each split, the stable node mapping, scalers and preprocessing metadata. IDs are `(node, origin UTC)` in frozen array order. Every primary fit uses 4,506,133 training, 199,670 validation and 75,566 test windows from 979 known nodes. History and horizon are six ten-minute bins each. Training target intervals end by 2022-02-11 07:10 UTC, validation is between that cut and 2022-02-20 03:30 UTC, and test begins at the latter cut. Accepted windows remain sparse; test has only 94 distinct origins. Its previous examination makes it a development benchmark, not a new untouched confirmatory period.

`weather` must be `enabled` or `masked`. The latter sets standardized outdoor temperature (index 7) and outdoor humidity (8) to zero, representing their training means. It preserves all 14 input slots, calendar features, inlet temperature, node identity, histories, labels, IDs and fitted scalers. Masking happens on batch/lag copies, never on frozen arrays. Identical seeds give paired neural variants identical initialization and shuffle sequences. No weather-specific tuning is performed.

Each new fit exclusively creates its output directory and a read-only `manifest.json`: exact effective config, source hashes and source snapshot, frozen-contract hash, package versions, Python/device/dtype, seed policy, train-index hash and feature mask. Files are created with no replacement, then fsynced/read-only. These are application-enforced immutable records, not storage hardware write-once guarantees. `fit_completed.json` binds the manifest and selected export, history, measurements, validation evaluator outputs and source snapshot with hashes. Reporting checks provenance again and refuses changed code, environment, data or exports. Training checkpoints are atomic within new directories. A completed matrix restart skips verified fits; a partial failed fit is retained and cannot be overwritten or automatically resumed. Use a new output root to repeat an experiment. Preserve exact package versions and recorded execution sources to reproduce an earlier report.

LSTM and GRU share two 64-unit layers, an eight-dimensional node embedding, batch 512, learning rate .001, 30-epoch cap, patience 5 and residual forecasts. Training minimizes scaled MSE; epoch selection uses physical validation RMSE. The zero-correction initial network is a selection candidate. Ties retain the earliest candidate. Inputs/targets stay on CPU and batches move to the requested CUDA device; CPU fallback is disabled. PyTorch deterministic algorithms and CuDNN deterministic mode are enabled with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Exact reproducibility still depends on hardware, library versions and backend; cross-device predictions need not be bit identical.

XGBoost uses the native API, no scikit-learn dependency. Its 79-column float32 matrix is:

- Columns `13*s+j`, for s=0..5 and j=0..12: feature j in chronological input bin s. Bin starts are origin minus 60,50,40,30,20,10 minutes; the final bin ends at the origin. All 13 numeric channels are retained, including weather-mask columns.
- Column 78: one frozen integer node index, marked categorical with `feature_types=['q']*78+['c']`. Codes are exactly the `node_names.npy` positions, consistent in every split. They are not treated as ordered numeric distances. Training covers every recorded node; unseen-node forecasting is outside this experiment.

The regressor predicts standardized target minus standardized persistence, starts with `base_score=0`, and adds persistence back at inference. Frozen persistence is the last completed ten-minute binned power mean, not the previous-hour average. `QuantileDMatrix` uses training quantile cuts for validation. Settings are GPU hist, max depth 6, max bin 256, eta .05, subsample .8, column sample .8, 500-tree cap and patience 30. The custom validation metric reconstructs target predictions in W, never evaluates test data. It selects the earliest minimum and compares it with zero correction. Native inference explicitly uses selected iteration counts; the UBJSON export is sliced to selected trees and preserves categorical splits. A recorded iteration 0 means use persistence rather than the saved tree artifact. Details follow the [official categorical documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) and [native Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html). The numeric/categorical mapping is also saved in each boosted fit.

The model-neutral evaluator verifies prediction IDs and array lengths, rejects nonfinite outputs, inverts target normalization and converts kW to W. It reports pooled MAE/RMSE, persistence MAE/RMSE, equal-node MAE, per-node scores and daily window/origin coverage. Forecast plots use real UTC origins and NaN breaks wherever the ten-minute grid has a gap. Validation export is reloaded and evaluated to check both adapters. `fit` writes validation only; `report` is the explicit separate test command. Pilot test reporting is forbidden. Frozen-array integrity verification reads all split hashes and checks alignment; no test predictions or scores enter fitting or pilot decisions. Matrix reporting requires all 18 fits and writes a hash-bound selection lock before any test inference. It then produces individual reports, paired weather RMSE differences and across-seed RMSE mean/std. It does not use test scores to change settings or retrain.

Run from `/media/csedept/cse2014/Project/m100_training_template` with its venv. The installed optional dependencies are pinned in `requirements-experiments.txt`; XGBoost 3.4.1 and SciPy 1.18.1 binary wheels support this Python 3.14 installation. Existing NumPy 2.5.3, PyTorch 2.14.1+cu130 and CUDA libraries are preserved. XGBoost's installed binary reports CUDA support. GPU execution is additionally checked in the pilot and recorded saved booster config.

```bash
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements-experiments.txt
python -m unittest discover -s tests -v
```

The following freeze and pilots are run as part of verification. They must not be rerun into existing directories:

```bash
python experiments.py freeze --source-run output/m100_all_nodes --output-dir output/experiments/v1/frozen
python experiments.py fit --config configs/experiments/v1/pilot_gru.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v1/pilots/gru_smoke_seed42
python experiments.py fit --config configs/experiments/v1/pilot_xgboost.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v1/pilots/xgboost_100_seed42
```

The GRU smoke uses one epoch and 32,768 fixed, evenly distributed training IDs with all validation windows. This subset tests execution and is not a full accuracy experiment. The XGBoost pilot uses all training/validation windows, exactly 100 trees, and patience 100 so it cannot stop before reaching the cap. Both retain validation selection, export-reload checks and common evaluator output. Pilot settings are predefined; test metrics are unavailable to them. Resource reports distinguish wall time, lifetime process RSS, sampled process GPU memory (0.5-second sampling, possible missed peaks), PyTorch allocator peak when applicable, neural train/validation epoch times and boosted quantization/boosting times. Other users' GPU memory is excluded from the per-PID sample.

After inspecting the pilot report and deciding to launch the full 18-fit comparison, run these commands. The first fits all variants serially on CUDA GPU 0 and selects solely on validation. The second locks all selected fits and evaluates the frozen benchmark. Neither has been launched by the agent.

```bash
python experiments.py matrix --plan configs/experiments/v1/matrix.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v1/comparison --stage fit
python experiments.py matrix --plan configs/experiments/v1/matrix.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v1/comparison --stage report
```

For an individual full fit, use a unique output directory:

```bash
python experiments.py fit --config configs/experiments/v1/gru_masked_seed43.json --data-dir output/experiments/v1/frozen --output-dir output/experiments/v1/individual/gru_masked_seed43
python experiments.py report --output-dir output/experiments/v1/individual/gru_masked_seed43
```

The freeze adds about 2 GB on disk; fit outputs add checkpoints and validation predictions, and later test predictions. Inputs total about 1.50 GiB, and the dense boosted training lag matrix alone is 1.33 GiB before quantization and histogram working space. Final computational estimates and actually observed pilot measurements are in `output/experiments/v1/PILOT_REPORT.md`; they are estimates, not claims about completed full comparisons. The current label coverage and retrospective node-cohort selection remain limitations. Time-block uncertainty and coverage-cause auditing are follow-up analyses rather than changes to this primary frozen comparison.
