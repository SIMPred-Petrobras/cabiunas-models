# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anomaly detection for industrial time-series sensor data (Petrobras Cabiunas, Turbina A). Autoencoders trained on sensor readings (thermocouples, average exhaust temp, vibration, pressure) flag equipment anomalies from reconstruction error. Integrated with ClearML for experiment tracking, dataset versioning, and remote GPU execution.

The repo covers the **whole lifecycle**, not just training: train → self-contained inference bundle → calibrated production alerting → offline evaluation against DCS alarms. The detector is already handed off (see `relatorio_anexos/HANDOFF_DETECTOR_TURBINA_A.md`); most current work is calibration and evaluation, not new modeling.

Code comments, docstrings, config names, and reports are in **Portuguese** — match that when editing.

### Tech Stack

- **Deep learning: TensorFlow / Keras** (`from tensorflow import keras`, Keras 3). **NOT** PyTorch, **NOT** AutoKeras.
- **Hyperparameter search: KerasTuner** (`kt.RandomSearch` over `val_loss`) — tuning over hand-defined architectures, not AutoML/NAS.
- Model builders live in `model.py` (`build_{cnn1d,gru,lstm,dense}_autoencoder`), registered in `tuning.py::_BUILDERS`, selected via `MODEL_ARCH`. `MODEL_ARCH="isolation_forest"` bypasses Keras entirely and routes to `model_if.py`.
- **Docker image**: every config sets `tensorflow/tensorflow:2.16.1-gpu`. The `PipelineConfig` *default* is a stale `pytorch/pytorch:...` string — ignore it, don't "fix" configs to match it.
- `automl_dense_report.md` (in the legacy sibling checkout `analise_cabiunas/cabiunas-models/`, not here) describes a **different** project (Transpetro, PyTorch + custom AutoML). Only its *concepts* (load-residual, CUSUM, persisting the scaler) were borrowed — not its code or framework.

## Commands

### Setup
```bash
source ../../venv/bin/activate          # venv lives at analise_cabiunas/venv
export CLEARML_CONFIG_FILE="$(pwd)/clearml.conf"   # gitignored; must be created locally
```

### Run pipeline
```bash
PYTHONPATH=. python src/main.py --config configs/calibracao_v14_full2024/v14_full2024.json
```
`"RUN_REMOTE": true` makes the local process enqueue the task to `REMOTE_QUEUE` and exit; a `clearml-agent daemon --queue default` worker then pulls the code from git, downloads the ClearML dataset, and trains.

### Tests
```bash
PYTHONPATH=. python -m pytest tests/
PYTHONPATH=. python -m pytest tests/test_preprocess.py
PYTHONPATH=. python -m pytest tests/test_preprocess.py::TestHampelFilter
```
Tests are `unittest.TestCase` classes, so a single test is addressed as `<file>::<Class>::<test_method>`. All 9 test files are fast pure-function tests (preprocess invariants, time handling, feature engineering, split/threshold, inference, plots). Nothing trains a model, nothing needs a GPU or ClearML.

### Upload dataset to ClearML
```bash
PYTHONPATH=. python scripts/upload_cabiunas_dataset.py --config configs/<gen>/<config>.json
```

## Architecture

`src/main.py` loads the config JSON into `PipelineConfig`, opens a ClearML task, then **branches**:

```
MULTIVARIATE_JOINT=false  →  pipeline.run(cfg)                     # default
MULTIVARIATE_JOINT=true   →  pipeline_multi.run_pipeline_multivariado(...)
```

Both end in `main.py::_upload_run_artifacts`, which uploads per-sensor CSVs, figures, best hyperparameters, the inference bundle, and the model. Note the comment at `main.py:117` — models must be uploaded as *named* artifacts (`{sensor}_model_keras`) because `output_uri` writes them all to the same colliding path.

### Per-sensor path (`pipeline.py`, the default)

`run()` calls `discover_sensors()` (filtered by `SENSOR_REGEX` / `SENSOR_LIST` / `SENSOR_EXCLUDE`), then `run_one_sensor()` for each. Inside `run_one_sensor`:

```
build_sensor_dataframe → Hampel filter → build `exclude` mask → df_normal / df_all
→ clip bounds (train only) → normalize_train_only → make_sequences → train_val_split
→ run_tuner + refit_best_model → reconstruction MAE → threshold → anomaly_seq
→ map_seq_to_point_anomalies → predictive layer → plots → inference_bundle.json
```

**The `exclude` mask (`pipeline.py:207-297`) is the thing to read before changing any training-data behavior.** It is one boolean series OR-ed together from: alarm windows (`EXCLUDE_MINUTES_BEFORE/AFTER_ALARM`), long interpolation gaps, post-startup ramps, non-operating points, forward-fill constant runs, and gradient spikes. `df_normal = df_use[~exclude]` is the training set; `df_all` (unfiltered) is what gets scored. `TRAIN_START_DATE` / `TRAIN_END_DATE` then cut the training window further — that's how out-of-sample and backcast evaluations are set up without touching the scoring set.

Operating-state detection has two modes and **`RUNNING_COL` wins over `ENABLE_OPERATIONAL_MASK`** when set. In production configs it is `RUNNING_A > 0.5` or `NGP_A > 50`.

### Multivariate path (`pipeline_multi.py`)

One run over all `SENSOR_LIST` channels at once. `MODEL_MODE` picks the backend:
- `per_sensor` (default) — N tiny fixed univariate AEs (`per_sensor.py`), combined by MAX ≡ "OR of any sensor above its own quantile". Validated better than multivariate: +3pp recall, 5× fewer false alarms/day.
- `multivariate` — one AE with N channels; `TARGET_SENSOR` optionally restricts the threshold to a single channel's MAE (detects a sensor diverging from its neighbours).

### Core modules (`src/cnn1d_ae/`)

- **`config.py`** — `PipelineConfig`, 130+ fields, every one commented with *why*. Read this first; it is the real documentation of pipeline behavior.
- **`preprocess.py`** — the densest module: sentinel masking, Hampel, interpolation, fabricated-gap detection (upstream linear fills that hide real archiving holes), outlier clipping, stable-gradient / constant-run / startup / gradient-spike masks, train-only normalization.
- **`scoring.py`** — threshold selection (`max_train`, `p95/p97/p99`, `target_rate`, `mean_std`, `alarm_f2`), adaptive/regime-band thresholds, CUSUM alarm policy, debounce, sequence→point voting, and the alarm-hit evaluation.
- **`predictive.py`** — the *official* metric layer (see Evaluation below).
- **`inference.py`** — production scoring from a bundle, no retraining.
- Supporting: `sequences.py`, `tuning.py`, `model.py`, `model_if.py`, `feature_engineering.py`, `io.py`, `plots.py`.

## Production lifecycle

1. Training writes `best_model/inference_bundle.json` — self-contained: `feature_columns`, scaler `center`/`scale`, `clip_bounds`, `threshold`, `time_steps`/`stride`, `running_col`/`running_threshold`.
2. `scripts/finalize_bundle.py` injects a `production_alerting` block, converting the eval's *rank-quantile* `thr_q` (which doesn't exist in streaming) into an absolute `ewma_abs_threshold`, plus `half_life_hours` and `debounce_hours`.
3. `inference.py::score_production(model, bundle, df)` runs the deployable chain: reconstruction MAE → EWMA(half-life) → `>= ewma_abs_threshold` → mask OFF → debounce. `score_dataframe` is the simpler raw-threshold variant.
4. Shipped artifacts are committed under `production_bundles/` (17 sensors; see its README for the per-sensor operating points and source ClearML task).

**Never clip scoring data to training bounds.** `transform_features` defaults to `clip=False` on purpose — clipping would erase exactly the out-of-range anomalies (UNDER dips, drift) the detector exists to catch. `clip_bounds` in the bundle only documents the scaler.

Retuning a deployed bundle needs no retraining and no ClearML:
```bash
PYTHONPATH=. python scripts/set_bundle_threshold.py <bundle.json> --std_mult Y | --scale F | --abs V
```

## Evaluation

`hit_rate@single_threshold` is **not** the metric of record — it is too sensitive to tuning variance. The project metric is the predictive curve in `predictive.py`: extract incidents from the alarm CSV → EWMA health index → recall × false-alarms/day across a threshold sweep → `pick_operating_point()` under `PREDICTIVE_FA_BUDGET_PER_DAY`. Enabled by default (`ENABLE_PREDICTIVE_LAYER=True`).

`scripts/eval_per_sensor_level.py` is the per-sensor reference implementation (gap-based recall, FA/day, duty-cycle, `--sticky_hours`, `--ok_aware`) and is imported by many other scripts. `scripts/validate_deployed_2024.py` is the honest end-to-end check: deployed bundles, frozen thresholds, scored on 2024 data the model never saw.

`scripts/` holds ~90 mostly single-purpose analysis/eval/plot scripts, each with a usage docstring at the top. Many read cached ClearML artifacts straight from `~/.clearml/cache`, so they still run when the ClearML server is down. Prefer reading a script's docstring over inferring its contract.

## Configs

`configs/calibracao_v<N>_<theme>/` are append-only **experiment generations** — older directories are history, not dead code. Current frontier: `v14_full2024` / `v13_train2024` (training-window studies), `v12_pressao` (pressure sensors), `v11_residual` (common-mode residual), `v10_alarm_context`, `v9_*` (architecture comparison). **Copy the newest relevant config into a new file rather than editing an old one** — reruns of past generations must stay reproducible.

Parameters worth knowing beyond `config.py`'s own comments:
- `THRESH_MODE="mean_std"` + `THRESH_STD_MULT` (y) is the operator-facing sensitivity knob (`threshold = mean + y·std` of training MAE). Calibrate y with `scripts/sweep_threshold_mean_std.py` — the reconstruction error has a heavy right tail, so the textbook y=2–3 is far too conservative (y=3 detected nothing on TC382_03_A/2025, where the calibrated point sits at y≈0.2).
- `EXCLUDE_MINUTES_BEFORE/AFTER_ALARM` should be **asymmetric**: a wide symmetric window strips normal high-load operation from training and concentrates false positives there.
- `PREDICTIVE_EWMA_HALF_LIFE_HOURS_PER_SENSOR` — sustained thermal drift wants ~4h; brief UNDER dips (TC382_04_A) want ~0.5h or the EWMA smooths them away.

## Timezones — read before touching any data ingestion

Internally everything is UTC (`SOURCE_TZ` → `TARGET_TZ`, `APPLY_HOUR_SHIFT`/`SHIFT_HOURS`). Two source families need **opposite-sign** corrections, both established by cross-probing overlapping sensor values, not by assumption:

- `record_*` exports are UTC−3 and need **−3h**.
- The 2024 monthly XLSX remessa is local time and needs **+3h** (proven in `scripts/build_consolidated_2024_2026.py`, p50 |error| = 0.000 °C at +180 min).

Applying one by analogy to the other produces a silent 6-hour misalignment. This has already cost the project time twice.

## Key design decisions

- **Sentinel detection** — physical failure values (e.g. an open thermocouple reading) are masked to NaN before normalization so they don't poison the scaler or the model.
- **Fabricated-gap detection** — upstream sources sometimes fill archiving holes with a straight line and no NaN, invisible to gap exclusion. `ENABLE_FABRICATED_GAP_DETECTION` finds abnormally low curvature runs and reopens the hole. Off by default so it can't silently change validated configs.
- **Train-only statistics** — scaler, clip bounds, and threshold are all fit on `df_normal` and persisted; applying training statistics (rather than refitting on new data) is what keeps a calibrated threshold meaningful out-of-sample.
- **Sequence-to-point mapping** — anomaly decisions are made per window, then projected back to timestamps through a voting rule (`all_of_window` / `k_of_window`) to suppress false positives.
- **ClearML artifacts** — CSVs, hyperparameters, weights, bundles, and plots all upload per task; `artifacts_local/<task_id>/` keeps a local copy when upload fails.

## Known dead ends (from the handoff report — don't re-litigate)

More preprocessing, architecture swaps (GRU / LSTM / Dense / Transformer), ensembles, multivariate + `TARGET_SENSOR`, raising the absolute threshold, and baseline-relative detection were all tried and **none moved the metric**. The model is not the bottleneck; incident scarcity and data coverage are. Only 7 of 17 sensors have evaluable incidents — the 10 `TV_*` vibration channels have ~1 alarm/year, while the equipment is off.

Also: GPU/cuDNN + KerasTuner are non-deterministic, so per-sensor recall swings ~±10pt between retrains of the same config. A single retrain delta is not evidence of an improvement.

## Outputs

Gitignored (regenerable): `OUTPUT*/`, `runs*/`, `artifacts_local/`, `data*/`, `dados/`, `record_*/`, `task_plots/`, `clearml.conf`.
Committed (results of record): `production_bundles/`, `eval_predictive_out/`, `eval_pressure_out/`, `relatorio_anexos/`.

Caveat: `.gitignore` matches `OUTPUT*/` case-sensitively, but recent configs set `OUTPUT_ROOT` to lowercase `outputs/...` (e.g. `v14_full2024.json`), which is therefore **not** ignored. Don't commit it if a local run creates it.
