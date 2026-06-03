from __future__ import annotations

import os
import json
import re
from dataclasses import asdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import ensure_sensor_dirs, save_run_config, load_data, resolve_output_dir
from .model import setup_gpu
from .preprocess import (
    build_sensor_dataframe,
    apply_hampel_filter,
    build_exclusion_mask,
    build_startup_exclusion_mask,
    build_stable_gradient_mask,
    build_constant_run_mask,
    build_gradient_spike_mask,
    clip_outliers,
    apply_feature_engineering,
    normalize_train_only,
)
from .sequences import make_sequences, train_val_split
from .tuning import run_tuner, refit_best_model
from .scoring import (
    reconstruction_mae_per_seq,
    compute_threshold,
    compute_threshold_alarm_optimized,
    apply_adaptive_monthly_threshold,
    map_seq_to_point_anomalies,
    build_sequence_scores_df,
    compute_anomaly_rate_per_day,
    evaluate_alarm_detection,
    compute_monthly_mae_drift,
    build_operational_state,
    mask_anomaly_seq_by_operational_state,
)
from .plots import (
    plot_loss,
    plot_hist_mae,
    plot_series_with_anomalies,
    plot_series_alarm_anomaly_subplots,
    plot_series_with_mae_reconstruction,
)


def discover_sensors(cfg: PipelineConfig, df_feat: pd.DataFrame, df_raw: pd.DataFrame) -> List[str]:
    source_df = df_raw if cfg.TRAIN_SOURCE.lower() == "raw" else df_feat
    cols = [c for c in source_df.columns if c != cfg.TIME_COL]

    if cfg.SENSOR_REGEX:
        pat = re.compile(cfg.SENSOR_REGEX)
        cols = [c for c in cols if pat.search(c)]

    if cfg.MODE.lower() == "local":
        if cfg.SENSOR_LIST:
            want = set(cfg.SENSOR_LIST)
            cols = [c for c in cols if c in want]

    if cfg.SENSOR_EXCLUDE:
        bad = set(cfg.SENSOR_EXCLUDE)
        cols = [c for c in cols if c not in bad]

    return sorted(cols)


def eval_alarm_hit_rate(df_alarm: pd.DataFrame, df_point: pd.DataFrame, minutes: int) -> Dict:
    return evaluate_alarm_detection(df_alarm, df_point, minutes)


def _infer_sampling_interval_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("Nao ha pontos suficientes para inferir granularidade temporal.")
    deltas = index.to_series().diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        raise ValueError("Nao foi possivel inferir granularidade temporal: deltas invalidos.")
    median_dt = float(deltas.median())
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("Granularidade temporal inferida invalida.")
    return median_dt


def _validate_or_resolve_time_steps(cfg: PipelineConfig, index: pd.DatetimeIndex, sensor: str) -> Dict[str, Any]:
    median_dt_seconds = _infer_sampling_interval_seconds(index)
    report = {
        "sensor": sensor,
        "median_sampling_interval_seconds": float(median_dt_seconds),
        "configured_time_steps": int(cfg.TIME_STEPS),
        "context_hours": float(cfg.CONTEXT_HOURS) if cfg.CONTEXT_HOURS is not None else None,
        "inferred_time_steps_from_context": None,
        "effective_time_steps": int(cfg.TIME_STEPS),
    }

    if cfg.CONTEXT_HOURS is None:
        return report

    inferred = int(max(1, round((float(cfg.CONTEXT_HOURS) * 3600.0) / median_dt_seconds)))
    report["inferred_time_steps_from_context"] = inferred
    diff_ratio = abs(inferred - int(cfg.TIME_STEPS)) / max(1, int(cfg.TIME_STEPS))
    report["time_steps_diff_ratio"] = float(diff_ratio)

    if diff_ratio > float(cfg.TIME_STEPS_TOLERANCE):
        msg = (
            f"TIME_STEPS={cfg.TIME_STEPS} difere do contexto desejado "
            f"CONTEXT_HOURS={cfg.CONTEXT_HOURS}h para sensor={sensor}. "
            f"Granularidade mediana={median_dt_seconds:.3f}s, TIME_STEPS inferido={inferred}."
        )
        if cfg.REQUIRE_CONTEXT_MATCH:
            raise ValueError(msg)
        print(f"[TIME-STEPS] {msg} Usando TIME_STEPS inferido.")
        cfg.TIME_STEPS = inferred

    report["effective_time_steps"] = int(cfg.TIME_STEPS)
    return report


def run_one_sensor(cfg: PipelineConfig, df_alarm: pd.DataFrame, df_feat: pd.DataFrame, df_raw: pd.DataFrame, sensor: str) -> Dict:
    cfg = PipelineConfig(**asdict(cfg))
    out_dirs = ensure_sensor_dirs(cfg, sensor)
    save_run_config(cfg, out_dirs)

    model_path = os.path.join(out_dirs["best_model"], "model.keras")
    if (not cfg.OVERWRITE) and os.path.exists(model_path):
        print(f"[SKIP] {sensor} (modelo ja existe: {model_path})")
        return {"sensor": sensor, "skipped": True, "reason": "model_exists", "model_path": model_path}

    print("\n==============================")
    print(f"[SENSOR] {sensor}")
    print(f"[OUT]    {out_dirs['root']}")
    print("==============================")

    df_use, long_gap_mask = build_sensor_dataframe(cfg, df_feat, df_raw, sensor)
    time_steps_report = _validate_or_resolve_time_steps(cfg, df_use.index, sensor)
    save_run_config(cfg, out_dirs)

    # Carrega RUNNING_COL do DataFrame-fonte (df_use não inclui essa coluna).
    _running_col_series: pd.Series | None = None
    if cfg.RUNNING_COL:
        _src = df_raw if cfg.TRAIN_SOURCE.lower() == "raw" else df_feat
        if cfg.RUNNING_COL in _src.columns:
            _running_col_series = (
                _src.drop_duplicates(subset=[cfg.TIME_COL])
                .set_index(cfg.TIME_COL)[cfg.RUNNING_COL]
            )
            _running_col_series = pd.to_numeric(_running_col_series, errors="coerce")
            _running_col_series = _running_col_series.reindex(df_use.index).fillna(0.0)
        else:
            print(f"[RUNNING_COL] Coluna '{cfg.RUNNING_COL}' nao encontrada na fonte — ignorando.")

    # Filtro de Hampel para spikes isolados (pós-interpolação, pré-divisão normal/all)
    df_use = apply_hampel_filter(df_use, sensor, cfg)

    if "Tag" in df_alarm.columns:
        df_alarm_sensor = df_alarm.loc[df_alarm["Tag"] == sensor].copy()
    else:
        df_alarm_sensor = df_alarm.copy()
    if "Data da Ocorrencia" in df_alarm_sensor.columns:
        df_alarm_sensor = df_alarm_sensor.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")

    if float(df_use[sensor].std()) < cfg.MIN_STD:
        print(f"[SKIP] {sensor} (std muito baixo, sensor provavelmente travado)")
        return {"sensor": sensor, "skipped": True, "reason": "low_std"}

    exclude_alarm = build_exclusion_mask(
        df_use.index,
        df_alarm_sensor["Data da Ocorrencia"] if "Data da Ocorrencia" in df_alarm_sensor.columns else pd.Series(dtype="datetime64[ns]"),
        cfg.EXCLUDE_MINUTES_AROUND_ALARM,
    )

    exclude = exclude_alarm.copy()
    if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
        long_gap_mask = long_gap_mask.reindex(df_use.index).fillna(False)
        exclude = exclude | long_gap_mask

    # Exclusão pós-startup: remove rampa de temperatura do conjunto de treino
    if cfg.EXCLUDE_STARTUP_MINUTES > 0:
        off_thr = (
            float(cfg.OFF_ABS_THRESHOLD)
            if cfg.OFF_ABS_THRESHOLD is not None
            else float(df_use[sensor].quantile(cfg.OFF_VALUE_QUANTILE))
        )
        startup_excl = build_startup_exclusion_mask(
            df_use.index, df_use[sensor], off_thr, cfg.EXCLUDE_STARTUP_MINUTES
        )
        n_startup_excl = int(startup_excl.sum())
        if n_startup_excl:
            print(
                f"[STARTUP-EXCL] sensor={sensor}: {n_startup_excl} pontos excluidos do treino "
                f"({n_startup_excl*30/60:.0f} min, EXCLUDE_STARTUP_MINUTES={cfg.EXCLUDE_STARTUP_MINUTES})."
            )
        exclude = exclude | startup_excl

    # Exclusão de períodos não-ON do treino.
    # Modo 1 — RUNNING_COL: usa coluna binária direta do dado (ex: RUNNING_A > 0.5).
    # Modo 2 — ENABLE_OPERATIONAL_MASK: infere estado por limiar do próprio sensor.
    # Modo 1 tem prioridade; é mais preciso quando disponível.
    if _running_col_series is not None:
        exclude_off_train = _running_col_series <= 0.5
        n_off = int(exclude_off_train.sum())
        print(
            f"[RUNNING_COL] sensor={sensor}: {n_off} pontos OFF excluidos do treino "
            f"via coluna '{cfg.RUNNING_COL}' ({100*n_off/max(len(df_use),1):.1f}%)."
        )
        exclude = exclude | exclude_off_train
    elif cfg.ENABLE_OPERATIONAL_MASK:
        train_op_state = build_operational_state(
            index=df_use.index,
            sensor_series=df_use[sensor],
            off_value_quantile=cfg.OFF_VALUE_QUANTILE,
            off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
            off_long_min_hours=cfg.OFF_LONG_MIN_HOURS,
            transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
            transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
        )
        exclude_off_train = train_op_state != "on"
        n_off_excl = int(exclude_off_train.sum())
        print(
            f"[OP-MASK-TRAIN] sensor={sensor}: {n_off_excl} pontos OFF/transiente "
            f"excluidos do treino ({100*n_off_excl/max(len(df_use),1):.1f}% do total)."
        )
        exclude = exclude | exclude_off_train

    # Exclusão de runs de forward-fill upstream (dado pré-interpolado).
    # Runs de ≥ CONSTANT_RUN_MIN_LENGTH valores idênticos durante ON são artefatos:
    # o sensor não mediu nada novo — o sistema repetiu o último valor.
    # Incluí-los no treino faz o AE aprender plateaus como "normal" e
    # flagrar esses mesmos plateaus como anomalias na inferência.
    if cfg.EXCLUDE_CONSTANT_RUNS:
        const_mask = build_constant_run_mask(df_use[sensor], min_length=cfg.CONSTANT_RUN_MIN_LENGTH)
        const_mask = const_mask.reindex(df_use.index).fillna(False)
        n_const = int(const_mask.sum())
        if n_const:
            print(
                f"[CONST-RUN] sensor={sensor}: {n_const} pontos de forward-fill "
                f"(runs>={cfg.CONSTANT_RUN_MIN_LENGTH}) excluidos do treino."
            )
        exclude = exclude | const_mask

    # Exclusão de ±GRADIENT_SPIKE_SUPPRESS_MINUTES ao redor de spikes de gradiente local.
    # Calibrado sobre estado ON para não confundir rampas de startup com spikes.
    if cfg.ENABLE_GRADIENT_SPIKE_MASK:
        spike_mask = build_gradient_spike_mask(
            df_use, sensor, _running_col_series,
            cfg.GRADIENT_SPIKE_STD_MULT,
            cfg.GRADIENT_SPIKE_SUPPRESS_MINUTES,
        )
        spike_mask = spike_mask.reindex(df_use.index).fillna(False)
        n_spike = int(spike_mask.sum())
        if n_spike:
            print(
                f"[GRAD-SPIKE] sensor={sensor}: {n_spike} pontos excluidos do treino "
                f"ao redor de spikes de gradiente "
                f"(std_mult={cfg.GRADIENT_SPIKE_STD_MULT}, suppress={cfg.GRADIENT_SPIKE_SUPPRESS_MINUTES} min)."
            )
        exclude = exclude | spike_mask

    df_normal = df_use.loc[~exclude].copy()
    df_all = df_use.copy()

    if cfg.TRAIN_END_DATE:
        cutoff = pd.Timestamp(cfg.TRAIN_END_DATE, tz=df_normal.index.tz)
        n_before = len(df_normal)
        df_normal = df_normal[df_normal.index <= cutoff]
        print(f"[TRAIN_END_DATE] {sensor}: treino restrito a ≤{cfg.TRAIN_END_DATE} "
              f"({len(df_normal)}/{n_before} pontos normais mantidos)")

    if len(df_normal) < cfg.TIME_STEPS + 10:
        print(f"[SKIP] {sensor} (poucos dados normais apos exclusao)")
        return {"sensor": sensor, "skipped": True, "reason": "few_normal_points"}

    df_normal = clip_outliers(df_normal, cfg)
    df_all = clip_outliers(df_all, cfg)

    df_normal, df_all = apply_feature_engineering(df_normal, df_all, sensor, cfg)
    feature_engineering_report = {
        "rolling_features_enabled": bool(cfg.ENABLE_ROLLING_FEATURES),
        "rolling_window": int(cfg.ROLLING_WINDOW if cfg.ROLLING_WINDOW is not None else cfg.TIME_STEPS),
        "trend_features_enabled": bool(cfg.ENABLE_TREND_FEATURES),
        "spectral_features_enabled": bool(cfg.ENABLE_SPECTRAL_FEATURES and cfg.SENSOR_TYPE.lower() == "vibration"),
        "spectral_window": int(cfg.SPECTRAL_WINDOW if cfg.SPECTRAL_WINDOW is not None else cfg.TIME_STEPS),
        "spectral_stride": int(cfg.SPECTRAL_STRIDE if cfg.SPECTRAL_STRIDE is not None else cfg.STRIDE),
        "sensor_type": cfg.SENSOR_TYPE,
        "context_features_enabled": bool(cfg.ENABLE_CONTEXT_FEATURES),
        "context_cols": [c for c in (cfg.CONTEXT_COLS or []) if c in df_normal.columns and c != sensor],
        "feature_columns": list(df_normal.columns),
        "n_features": int(df_normal.shape[1]),
        "sentinel_mode": cfg.SENTINEL_MODE,
        "hampel_filter_enabled": bool(cfg.ENABLE_HAMPEL_FILTER),
        "normalize_on_stable_only": bool(cfg.NORMALIZE_ON_STABLE_ONLY),
        "exclude_startup_minutes": int(cfg.EXCLUDE_STARTUP_MINUTES),
    }

    # Máscara de gradiente estável para normalização robusta (anti-bimodalidade)
    stable_mask = None
    if cfg.NORMALIZE_ON_STABLE_ONLY:
        stable_mask = build_stable_gradient_mask(
            df_normal, sensor, cfg.STABLE_ON_GRADIENT_QUANTILE
        )

    df_normal_z, df_all_z, _, _ = normalize_train_only(cfg, df_normal, df_all, stable_mask=stable_mask)

    values_normal = df_normal_z.values.astype(np.float32)
    values_all = df_all_z.values.astype(np.float32)

    x_train_full = make_sequences(values_normal, cfg.TIME_STEPS, cfg.STRIDE)
    x_train, x_val = train_val_split(
        x_train_full,
        cfg.VAL_FRAC,
        cfg.SHUFFLE_TRAIN,
        cfg.RANDOM_SEED,
        split_mode=cfg.SPLIT_MODE,
    )
    n_features = x_train.shape[-1]

    best_hp, best_model, df_trials = run_tuner(cfg, out_dirs, x_train, x_val, n_features)
    df_trials.to_csv(os.path.join(out_dirs["csv"], "trials_ranking.csv"), index=False)

    with open(os.path.join(out_dirs["best_model"], "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hp.values, f, indent=2, ensure_ascii=False)

    history = refit_best_model(cfg, best_model, x_train, x_val)
    best_model.save(model_path)

    plot_loss(history, os.path.join(out_dirs["figs"], "loss_curve.png"))

    train_mae_seq = reconstruction_mae_per_seq(best_model, x_train_full, cfg.BATCH_SIZE)

    # x_all calculado antes do threshold para suportar calibração alarm_f2
    x_all = make_sequences(values_all, cfg.TIME_STEPS, cfg.STRIDE)
    mae_seq_all = reconstruction_mae_per_seq(best_model, x_all, cfg.BATCH_SIZE)

    _alarm_times_for_thresh = (
        df_alarm_sensor["Data da Ocorrencia"]
        if "Data da Ocorrencia" in df_alarm_sensor.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    _eval_win = int(cfg.EVAL_WINDOW_MINUTES) if cfg.EVAL_WINDOW_MINUTES is not None else int(cfg.EXCLUDE_MINUTES_AROUND_ALARM)

    if cfg.THRESH_MODE.lower() == "alarm_f2":
        threshold = compute_threshold_alarm_optimized(
            mae_seq=mae_seq_all,
            index=df_all_z.index,
            alarm_times=_alarm_times_for_thresh,
            time_steps=cfg.TIME_STEPS,
            eval_window_minutes=_eval_win,
            target_recall=cfg.ALARM_F2_TARGET_RECALL,
            max_fp_per_day=cfg.ALARM_F2_MAX_FP_PER_DAY,
            incident_gap_hours=cfg.ALARM_F2_INCIDENT_GAP_HOURS,
            stride=cfg.STRIDE,
        )
    else:
        threshold = compute_threshold(train_mae_seq, cfg.THRESH_MODE, target_rate=cfg.TARGET_ANOMALY_RATE)

    plot_hist_mae(train_mae_seq, threshold, os.path.join(out_dirs["figs"], "train_mae_hist.png"))

    # Threshold adaptativo mensal: recalibra sem retreinar, resolve drift sazonal
    monthly_thresholds: dict = {}
    if cfg.ADAPTIVE_THRESHOLD_MODE.lower() == "monthly":
        anomaly_seq, monthly_thresholds = apply_adaptive_monthly_threshold(
            mae_seq=mae_seq_all,
            index=df_all_z.index,
            time_steps=cfg.TIME_STEPS,
            alarm_times=_alarm_times_for_thresh,
            eval_window_minutes=_eval_win,
            percentile=cfg.ADAPTIVE_THRESHOLD_PERCENTILE,
            stride=cfg.STRIDE,
        )
        print(f"[ADAPTIVE] Thresholds mensais: { {k: f'{v:.5f}' for k, v in monthly_thresholds.items()} }")
    else:
        anomaly_seq = mae_seq_all > threshold
    state = None
    if cfg.ENABLE_OPERATIONAL_MASK:
        state = build_operational_state(
            index=df_all_z.index,
            sensor_series=df_all[sensor],
            off_value_quantile=cfg.OFF_VALUE_QUANTILE,
            off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
            off_long_min_hours=cfg.OFF_LONG_MIN_HOURS,
            transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
            transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
        )
        anomaly_seq = mask_anomaly_seq_by_operational_state(
            anomaly_seq=anomaly_seq,
            index=df_all_z.index,
            time_steps=cfg.TIME_STEPS,
            state=state,
            stride=cfg.STRIDE,
        )
    elif _running_col_series is not None:
        # Constrói estado operacional direto do RUNNING_COL para plots e avaliação.
        state = _running_col_series.reindex(df_all_z.index).fillna(0.0).map(
            lambda x: "on" if x > 0.5 else "off"
        )
        anomaly_seq = mask_anomaly_seq_by_operational_state(
            anomaly_seq=anomaly_seq,
            index=df_all_z.index,
            time_steps=cfg.TIME_STEPS,
            state=state,
            stride=cfg.STRIDE,
        )

    all_index = df_all_z.index
    df_seq_scores = build_sequence_scores_df(all_index, mae_seq_all, anomaly_seq, stride=cfg.STRIDE)
    df_seq_scores.to_csv(os.path.join(out_dirs["csv"], "sequence_scores_all.csv"), index=False)

    df_point = map_seq_to_point_anomalies(
        anomaly_seq,
        all_index,
        cfg.TIME_STEPS,
        cfg.POINT_RULE,
        cfg.POINT_WINDOW,
        cfg.POINT_MIN_COUNT,
        stride=cfg.STRIDE,
    )
    if state is not None:
        df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
        df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0

    # Nível de warning: threshold mais permissivo, min_count menor → detecção antecipada
    if cfg.ENABLE_WARN_LEVEL:
        warn_thresh = compute_threshold(train_mae_seq, "target_rate", target_rate=cfg.WARN_TARGET_ANOMALY_RATE)
        warn_seq = mae_seq_all > warn_thresh
        if state is not None:
            warn_seq = mask_anomaly_seq_by_operational_state(
                anomaly_seq=warn_seq, index=all_index, time_steps=cfg.TIME_STEPS, state=state, stride=cfg.STRIDE
            )
        df_warn = map_seq_to_point_anomalies(
            warn_seq, all_index, cfg.TIME_STEPS,
            cfg.POINT_RULE, cfg.POINT_WINDOW, cfg.WARN_POINT_MIN_COUNT,
            stride=cfg.STRIDE,
        )
        if state is not None:
            df_warn["operational_state"] = state.reindex(df_warn.index).fillna("on")
            df_warn.loc[df_warn["operational_state"] != "on", "is_anom_point"] = 0
        df_point["is_warn_point"] = df_warn["is_anom_point"]
        df_point["is_warn_only"] = ((df_point["is_warn_point"] == 1) & (df_point["is_anom_point"] == 0)).astype(int)

    df_point.to_csv(os.path.join(out_dirs["csv"], "point_anomalies_all.csv"))

    anomalous_times = df_point.index[df_point["is_anom_point"] == 1]
    plot_series_with_anomalies(
        df_all[sensor],
        anomalous_times,
        os.path.join(out_dirs["figs"], "series_with_anomalies.png"),
        title=f"Serie + anomalias (CNN1D-AE) | sensor={sensor}",
        operational_state=state,
    )
    plot_series_alarm_anomaly_subplots(
        df_all[sensor],
        anomalous_times,
        df_alarm_sensor["Data da Ocorrencia"] if "Data da Ocorrencia" in df_alarm_sensor.columns else pd.Series(dtype="datetime64[ns]"),
        os.path.join(out_dirs["figs"], "series_alarm_anomaly_subplots.png"),
        title=f"{sensor}",
        operational_state=state,
    )
    plot_series_with_mae_reconstruction(
        df_all[sensor],
        df_seq_scores.set_index("seq_start_time") if "seq_start_time" in df_seq_scores.columns else df_seq_scores,
        threshold,
        anomalous_times,
        df_alarm_sensor["Data da Ocorrencia"] if "Data da Ocorrencia" in df_alarm_sensor.columns else pd.Series(dtype="datetime64[ns]"),
        os.path.join(out_dirs["figs"], "series_mae_reconstruction.png"),
        title=f"{sensor} | Série + Erro de Reconstrução MAE",
        operational_state=state,
    )

    eval_window_minutes = (
        int(cfg.EVAL_WINDOW_MINUTES)
        if cfg.EVAL_WINDOW_MINUTES is not None
        else int(cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    )
    eval_stats = eval_alarm_hit_rate(df_alarm_sensor, df_point, eval_window_minutes)

    df_monthly_drift = compute_monthly_mae_drift(df_seq_scores, threshold)
    monthly_drift_path = os.path.join(out_dirs["csv"], "monthly_mae_drift.csv")
    df_monthly_drift.to_csv(monthly_drift_path, index=False)
    drift_summary = {
        "n_months": int(len(df_monthly_drift)),
        "n_months_p99_above_threshold": (
            int(df_monthly_drift["drift_flag_p99_above_threshold"].sum())
            if not df_monthly_drift.empty
            else 0
        ),
        "max_p99_threshold_ratio": (
            float(df_monthly_drift["p99_threshold_ratio"].max())
            if not df_monthly_drift.empty
            else None
        ),
    }

    # Warning-level evaluation (medido separadamente sobre is_warn_point)
    warn_eval_stats: dict = {}
    if cfg.ENABLE_WARN_LEVEL and "is_warn_point" in df_point.columns:
        df_warn_eval = df_point.rename(columns={"is_warn_point": "is_anom_point"})[["is_anom_point"]]
        warn_eval_stats = eval_alarm_hit_rate(df_alarm_sensor, df_warn_eval, eval_window_minutes)
        warn_eval_stats = {f"warn_{k}": v for k, v in warn_eval_stats.items()}

    calibration_report = {
        "sensor": sensor,
        "threshold": float(threshold),
        "THRESH_MODE": cfg.THRESH_MODE,
        "TARGET_ANOMALY_RATE": float(cfg.TARGET_ANOMALY_RATE),
        "POINT_RULE": cfg.POINT_RULE,
        "POINT_WINDOW": int(cfg.POINT_WINDOW),
        "POINT_MIN_COUNT": int(cfg.POINT_MIN_COUNT),
        "anomaly_rate_points_per_day": compute_anomaly_rate_per_day(df_point),
        "operational_mask_enabled": bool(cfg.ENABLE_OPERATIONAL_MASK),
        "feature_engineering": feature_engineering_report,
        "time_steps_report": time_steps_report,
        "monthly_mae_drift_summary": drift_summary,
        "monthly_thresholds": {str(k): float(v) for k, v in monthly_thresholds.items()},
        **eval_stats,
        **warn_eval_stats,
    }
    if state is not None:
        counts = state.value_counts().to_dict()
        calibration_report["operational_state_counts"] = {str(k): int(v) for k, v in counts.items()}
    with open(os.path.join(out_dirs["csv"], "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    report = {
        "sensor": sensor,
        "output_dir": out_dirs["root"],
        "model_path": model_path,
        "threshold": float(threshold),
        "THRESH_MODE": cfg.THRESH_MODE,
        **eval_stats,
        "skipped": False,
    }
    with open(os.path.join(out_dirs["csv"], "evaluation_alarm_hit_rate.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _worker_entry(cfg_dict: Dict, sensor: str) -> Dict:
    cfg = PipelineConfig(**cfg_dict)
    setup_gpu()
    df_alarm, df_feat, df_raw, _ = load_data(cfg)
    return run_one_sensor(cfg, df_alarm, df_feat, df_raw, sensor)


def run(cfg: PipelineConfig) -> Dict[str, Any]:
    setup_gpu()
    df_alarm, df_feat, df_raw, time_report = load_data(cfg)

    sensors = discover_sensors(cfg, df_feat, df_raw)
    if not sensors:
        raise ValueError("Nenhum sensor encontrado apos filtros do config.")

    if cfg.SENSOR_FILTER_HAS_ALARMS and df_alarm is not None and not df_alarm.empty:
        tag_col = next((c for c in df_alarm.columns if "tag" in c.lower() and "alarm" in c.lower()), None)
        if tag_col:
            alarm_tags = set(df_alarm[tag_col].dropna().unique())
            before = len(sensors)
            sensors = [s for s in sensors if s in alarm_tags]
            print(f"[SENSOR_FILTER_HAS_ALARMS] {before} → {len(sensors)} sensores "
                  f"(mantidos: {sensors})")

    print(f"[MODE] {cfg.MODE} | sensors={len(sensors)} | N_WORKERS={cfg.N_WORKERS}")
    print(f"[SENSORS] primeiros 20: {sensors[:20]}")

    summary_out_root = cfg.OUTPUT_ROOT if cfg.OUTPUT_ROOT else "."
    os.makedirs(summary_out_root, exist_ok=True)
    summary_path = os.path.join(summary_out_root, "summary_all_sensors.csv")
    time_report_path = os.path.join(summary_out_root, "time_integrity_report.json")
    with open(time_report_path, "w", encoding="utf-8") as f:
        json.dump(time_report, f, indent=2, ensure_ascii=False)

    rows = []

    if cfg.N_WORKERS <= 1:
        for s in sensors:
            try:
                rows.append(run_one_sensor(cfg, df_alarm, df_feat, df_raw, s))
            except Exception as e:
                print(f"[ERROR] sensor={s} -> {e}")
                rows.append({"sensor": s, "skipped": True, "reason": f"exception: {e}", "output_dir": resolve_output_dir(cfg, s)})
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        cfg_dict = asdict(cfg)
        with ProcessPoolExecutor(max_workers=cfg.N_WORKERS) as ex:
            futures = {ex.submit(_worker_entry, cfg_dict, s): s for s in sensors}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    print(f"[ERROR] sensor={s} -> {e}")
                    rows.append({"sensor": s, "skipped": True, "reason": f"exception: {e}", "output_dir": resolve_output_dir(cfg, s)})

    df_summary = pd.DataFrame(rows).sort_values(["skipped", "sensor"])
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[DONE] Summary salvo em: {summary_path}")

    return {
        "summary_path": summary_path,
        "time_report_path": time_report_path,
        "sensor_outputs": rows,
        "sensors": sensors,
    }
