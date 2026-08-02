# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anomaly detection pipeline for industrial time-series sensor data (Petrobras Cabiunas facility). Uses a CNN-1D Autoencoder trained on sensor readings (temperature, pressure, vibration) to detect equipment anomalies. Integrated with ClearML for experiment tracking, dataset versioning, and remote GPU execution.

### Tech Stack

- **Deep learning: TensorFlow / Keras** (`from tensorflow import keras`; Docker `tensorflow/tensorflow:2.16.1-gpu` → Keras 3). **NOT** PyTorch, **NOT** AutoKeras.
- **Hyperparameter search: KerasTuner** (`keras_tuner`, `kt.RandomSearch` over `val_loss`) — low-level tuning over hand-defined architectures, not AutoML/NAS.
- Model builders live in `model.py` (`build_cnn1d/gru/lstm/dense_autoencoder`), registered in `tuning.py::_BUILDERS` and selected via `MODEL_ARCH`.
- Note: `automl_dense_report.md` describes a **different** project (Transpetro, PyTorch + custom AutoML). Only its *concepts* (load-residual, CUSUM, persisting the scaler) were borrowed — not its code or framework.

## Commands

### Setup
```bash
source ../venv/bin/activate
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"
```

### Run Pipeline (local)
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v4_eq/<config_file>.json
```

### Run Tests
```bash
PYTHONPATH=. python -m pytest tests/
# Single test file:
PYTHONPATH=. python -m pytest tests/test_preprocess.py
```

### Upload Dataset to ClearML
```bash
PYTHONPATH=. python scripts/upload_cabiunas_dataset.py --config configs/calibracao_v4_eq/<config_file>.json
```

### Remote Execution (ClearML Agent)
Set `"RUN_REMOTE": true` in the config JSON, then run the pipeline locally — it enqueues the task to the configured `REMOTE_QUEUE` instead of training locally.

## Architecture

The pipeline runs in this sequence:

```
Config JSON → ClearML Task → Load CSV (io.py) → Preprocess (preprocess.py)
→ Feature Engineering (feature_engineering.py) → Sequencing (sequences.py)
→ Hyperparameter Search (tuning.py) → Train CNN-1D AE (model.py)
→ Anomaly Scoring (scoring.py) → Visualization (plots.py)
```

### Core Modules (`src/cnn1d_ae/`)

- **`config.py`** — `PipelineConfig` dataclass with 90+ parameters. All pipeline behavior is controlled here. Read this first when changing behavior.
- **`preprocess.py`** — Most complex module. Applies: sentinel value masking (e.g., -40.5°C = broken thermocouple), Hampel filter (spike removal), interpolation, quantile/MAD outlier clipping, gradient-based stable-period detection, z-score/robust normalization.
- **`scoring.py`** — Threshold selection (max_train, p95/p97/p99, or target anomaly rate) and mapping sequence-level reconstruction MAE → point-level anomalies via voting (all_of_window / k_of_window).
- **`model.py`** — CNN-1D autoencoder: 2 Conv1D layers → bottleneck → 2 Conv1DTranspose layers. GPU memory growth enabled.
- **`tuning.py`** — KerasTuner RandomSearch over filter counts, kernel sizes, strides, dropout, learning rate, and regularization.
- **`sequences.py`** — Sliding window batching with configurable TIME_STEPS and STRIDE; temporal train/val split.

### Entry Point (`src/main.py`)

Parses `--config`, initializes `PipelineConfig`, creates/reconnects ClearML task, calls pipeline stages in order, uploads artifacts. `RUN_REMOTE=true` causes it to enqueue to a ClearML worker and exit.

### Configs (`configs/calibracao_v4_eq/`)

Production JSON configs, one per sensor/equipment variant. Key parameters to know:
- `SENTINEL_MODE`, `INTERPOLATE_LIMIT`, `HAMPEL_FILTER`, `NORMALIZE_MODE` — preprocessing behavior
- `TIME_STEPS` (default 60), `STRIDE`, `VAL_FRAC` — sequence construction
- `THRESH_MODE` (default `p99`), `POINT_RULE`, `POINT_WINDOW` — anomaly detection logic
  - `THRESH_MODE="mean_std"` + `THRESH_STD_MULT` (y) gives an operator-facing sensitivity
    knob (`threshold = mean + y·std` of the training MAE). Calibrate y with
    `scripts/sweep_threshold_mean_std.py` — the reconstruction error has a heavy right
    tail, so the textbook y=2–3 is far too conservative (y=3 detected nothing on
    TC382_03_A/2025, where the calibrated point sits at y≈0.2).
  - To retune a **deployed** bundle without retraining:
    `scripts/set_bundle_threshold.py <bundle.json> --std_mult Y | --scale F | --abs V`
- `CLEARML_DATASET_ID`, `RUN_REMOTE`, `REMOTE_QUEUE` — ClearML integration

## Key Design Decisions

- **Sentinel detection**: Physical sensor failure values (e.g., exactly -40.5°C for a broken thermocouple) are masked before normalization to avoid poisoning the model.
- **Stable period gating**: Only data segments with low gradient variance are used for training, preventing the model from learning transient startup/shutdown behavior.
- **Sequence-to-point mapping**: Anomaly decisions are made per-window then projected back to timestamps using a voting rule to reduce false positives.
- **ClearML artifacts**: All outputs (CSVs, best hyperparameters, trained model weights, plots) are uploaded as ClearML task artifacts for reproducibility.
