# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNN-1D Autoencoder pipeline for anomaly detection in time-series sensor data (Cabiunas/Petrobras), integrated with ClearML for dataset versioning, experiment tracking, and remote execution via ClearML Agent.

- **ClearML project**: `TesteMLCab`
- **ClearML dataset**: `Cabiunas 2025` (ID: `e2765c3eef2349cda5f5cbcb0fcd5a40`)
- **Remote queue**: `default`
- **Docker image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

## Environment Setup

```bash
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
```

`clearml.conf` must exist at the project root but is excluded from Git.

## Common Commands

**Run pipeline (local or submit remote):**
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json
```

**Run tests:**
```bash
PYTHONPATH=. python -m pytest tests/
# or a single test file:
PYTHONPATH=. python -m pytest tests/test_split_and_threshold.py
```

**Validate syntax and config:**
```bash
python3 -m compileall src/main.py src/cnn1d_ae
python3 -m json.tool configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json
```

**Upload/update dataset:**
```bash
PYTHONPATH=. python scripts/upload_cabiunas_dataset.py --config configs/calibracao_v4_eq/record2025_tzm3_interpolado_v4_T5_AVG_A.json
```

**Generate plots from a completed ClearML task:**
```bash
PYTHONPATH=. python scripts/plot_task_artifacts.py --task-id <TASK_ID>
PYTHONPATH=. python scripts/plot_task_zoom_series.py --task-id <TASK_ID>
# --include-raw also downloads dataset to plot the raw time series
```

**Start remote ClearML worker:**
```bash
clearml-agent daemon --queue default
```

## Architecture

### Execution Flow

`src/main.py` is the entrypoint. It:
1. Loads a JSON config into `PipelineConfig` (a dataclass in `src/cnn1d_ae/config.py`)
2. Initializes a ClearML `Task`, connects config parameters, sets docker image
3. If `RUN_REMOTE=true` and running locally, calls `task.execute_remotely()` — the local process exits and the worker picks it up from Git
4. Otherwise calls `pipeline.run(cfg)` which iterates over all discovered sensors

### Per-Sensor Pipeline (`src/cnn1d_ae/pipeline.py: run_one_sensor`)

For each sensor the pipeline:
1. Builds a clean DataFrame excluding alarm windows and long gaps (`preprocess.py`)
2. Clips outliers, normalizes using train-only statistics (`zscore` or `robust`)
3. Creates sliding-window sequences (`sequences.py: make_sequences`)
4. Runs KerasTuner hyperparameter search (`tuning.py: run_tuner`)
5. Refits best model and saves to `best_model/model.keras`
6. Scores all data (including anomaly periods) via MAE reconstruction error
7. Applies threshold (`scoring.py: compute_threshold`) and optional operational mask
8. Maps sequence-level anomalies to point-level anomalies (`POINT_RULE`, `POINT_WINDOW`, `POINT_MIN_COUNT`)
9. Saves CSVs, figures, and JSON reports; uploads artifacts to ClearML

### Key Modules

| Module | Responsibility |
|---|---|
| `config.py` | `PipelineConfig` dataclass — all hyperparameters and paths |
| `io.py` | Data loading (from ClearML Dataset or local paths), timezone conversion, time integrity audit |
| `preprocess.py` | Alarm exclusion mask, outlier clipping, train-only normalization |
| `sequences.py` | Sliding window sequence creation, train/val split (temporal or random) |
| `model.py` | CNN-1D autoencoder architecture (KerasTuner), GPU setup |
| `tuning.py` | KerasTuner `RandomSearch` execution, trial ranking |
| `scoring.py` | MAE scoring, threshold modes, seq→point anomaly mapping, operational state mask |
| `plots.py` | Loss curves, MAE histograms, series+anomaly overlays, alarm comparison subplots |

### Output Structure

Each sensor produces an output directory (`OUTPUT_DIR_TEMPLATE`, default `OUTPUT_CNN1D_AE_{sensor}` under `OUTPUT_ROOT`):
```
OUTPUT_CNN1D_AE_<sensor>/
  tuner/               # KerasTuner trial artifacts
  best_model/          # model.keras + best_hyperparameters.json
  figs/                # loss_curve, train_mae_hist, series_with_anomalies, series_alarm_anomaly_subplots
  csv/                 # run_config.json, trials_ranking.csv, sequence_scores_all.csv,
                       # point_anomalies_all.csv, calibration_report.json, evaluation_alarm_hit_rate.json
```

### Config System

All pipeline behavior is driven by JSON configs in `configs/`. Keys map directly to `PipelineConfig` fields. Two config families exist:
- `calibracao_v3_on/` — operational mask enabled (`ENABLE_OPERATIONAL_MASK: true`)
- `calibracao_v4_eq/` — equalized calibration configs

`RUN_REMOTE: true` in a config causes the local process to enqueue the task for the ClearML worker. The worker clones the repo from Git and runs with `USE_CLEARML_DATASET: true` to pull data from ClearML.

### Time Handling

All timestamps are normalized to UTC-naive on load (`io.py`). Source data is `America/Sao_Paulo`. The `APPLY_HOUR_SHIFT` + `SHIFT_HOURS` parameters handle edge cases where raw data has an additional offset (e.g., `SHIFT_HOURS: -3` for `tzm3` datasets).

### Parallel Sensor Execution

`N_WORKERS > 1` spawns a `ProcessPoolExecutor`. Each worker reloads all data independently (required because TensorFlow/Keras cannot be forked cleanly). Keep `N_WORKERS=1` for debugging.
