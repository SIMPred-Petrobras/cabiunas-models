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
    compute_clip_bounds,
    apply_clip_bounds,
    apply_feature_engineering,
    normalize_train_only,
)
from .sequences import make_sequences, train_val_split
from .tuning import run_tuner, refit_best_model
from .model_if import fit_and_score as if_fit_and_score
from .scoring import (
    reconstruction_mae_per_seq,
    compute_threshold,
    compute_threshold_alarm_optimized,
    apply_adaptive_monthly_threshold,
    apply_rolling_percentile_threshold,
    apply_cusum_alarm,
    apply_debounce,
    assign_regime_bands,
    compute_regime_band_thresholds,
    apply_regime_band_threshold,
    map_seq_to_point_anomalies,
    build_sequence_scores_df,
    compute_anomaly_rate_per_day,
    evaluate_alarm_detection,
    compute_monthly_mae_drift,
    filter_short_anomaly_runs,
    build_operational_state,
    mask_anomaly_seq_by_operational_state,
)
from .predictive import (
    extract_incidents,
    compute_health_index_ewma,
    compute_predictive_curve,
    pick_operating_point,
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
    _arch = getattr(cfg, "MODEL_ARCH", "cnn1d")
    if _arch != "isolation_forest" and (not cfg.OVERWRITE) and os.path.exists(model_path):
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

    # Remove condições que não devem gerar janela de exclusão de treino (ex: LOLO = parada planejada)
    if cfg.TRAIN_SKIP_CONDITIONS:
        _cond_col = next((c for c in df_alarm_sensor.columns if "condi" in c.lower()
                          and "não" not in c.lower()), None)
        if _cond_col:
            _skip = {c.upper() for c in cfg.TRAIN_SKIP_CONDITIONS}
            _before = len(df_alarm_sensor)
            df_alarm_sensor = df_alarm_sensor[
                ~df_alarm_sensor[_cond_col].str.upper().fillna("").isin(_skip)
            ]
            print(f"  [TRAIN_SKIP] Removidas {_before - len(df_alarm_sensor)} linhas de alarme "
                  f"({cfg.TRAIN_SKIP_CONDITIONS}) da janela de exclusão de treino")

    if float(df_use[sensor].std()) < cfg.MIN_STD:
        print(f"[SKIP] {sensor} (std muito baixo, sensor provavelmente travado)")
        return {"sensor": sensor, "skipped": True, "reason": "low_std"}

    exclude_alarm = build_exclusion_mask(
        df_use.index,
        df_alarm_sensor["Data da Ocorrencia"] if "Data da Ocorrencia" in df_alarm_sensor.columns else pd.Series(dtype="datetime64[ns]"),
        cfg.EXCLUDE_MINUTES_AROUND_ALARM,
        minutes_before=cfg.EXCLUDE_MINUTES_BEFORE_ALARM,
        minutes_after=cfg.EXCLUDE_MINUTES_AFTER_ALARM,
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
        exclude_off_train = _running_col_series <= cfg.RUNNING_THRESHOLD
        n_off = int(exclude_off_train.sum())
        print(
            f"[RUNNING_COL] sensor={sensor}: {n_off} pontos OFF excluidos do treino "
            f"via coluna '{cfg.RUNNING_COL}' (thr={cfg.RUNNING_THRESHOLD}, {100*n_off/max(len(df_use),1):.1f}%)."
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
            getattr(cfg, "GRADIENT_SPIKE_TRANSITION_MINUTES", 0),
        )
        spike_mask = spike_mask.reindex(df_use.index).fillna(False)
        n_spike = int(spike_mask.sum())
        if n_spike:
            trans_min = getattr(cfg, "GRADIENT_SPIKE_TRANSITION_MINUTES", 0)
            steady_label = f", steady_trans={trans_min}min" if trans_min > 0 else ""
            print(
                f"[GRAD-SPIKE] sensor={sensor}: {n_spike} pontos excluidos do treino "
                f"(std_mult={cfg.GRADIENT_SPIKE_STD_MULT}, suppress={cfg.GRADIENT_SPIKE_SUPPRESS_MINUTES}min"
                f"{steady_label})."
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

    if getattr(cfg, "TRAIN_START_DATE", None):
        cutoff_lo = pd.Timestamp(cfg.TRAIN_START_DATE, tz=df_normal.index.tz)
        n_before = len(df_normal)
        df_normal = df_normal[df_normal.index >= cutoff_lo]
        print(f"[TRAIN_START_DATE] {sensor}: treino restrito a ≥{cfg.TRAIN_START_DATE} "
              f"({len(df_normal)}/{n_before} pontos normais mantidos)")

    if len(df_normal) < cfg.TIME_STEPS + 10:
        print(f"[SKIP] {sensor} (poucos dados normais apos exclusao)")
        return {"sensor": sensor, "skipped": True, "reason": "few_normal_points"}

    # Bounds de clip de TREINO (df_normal): persistidos no bundle só para documentar
    # as estatísticas do scaler. NÃO clipar df_all/scoring com esses limites: cortaria
    # as anomalias fora-de-faixa (UNDER do TC, drift do T5) que precisam aparecer no
    # erro de reconstrução. df_all clipa com os próprios limites (de-glitch suave).
    clip_bounds = compute_clip_bounds(df_normal, cfg)
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

    df_normal_z, df_all_z, center, scale = normalize_train_only(cfg, df_normal, df_all, stable_mask=stable_mask)

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

    # x_all necessário para ambas as arquiteturas
    x_all = make_sequences(values_all, cfg.TIME_STEPS, cfg.STRIDE)

    arch = getattr(cfg, "MODEL_ARCH", "cnn1d")
    if arch == "isolation_forest":
        train_mae_seq, mae_seq_all = if_fit_and_score(x_train_full, x_all, cfg)
        best_model = None
        best_hp    = None
    else:
        best_hp, best_model, df_trials = run_tuner(cfg, out_dirs, x_train, x_val, n_features)
        df_trials.to_csv(os.path.join(out_dirs["csv"], "trials_ranking.csv"), index=False)

        with open(os.path.join(out_dirs["best_model"], "best_hyperparameters.json"), "w", encoding="utf-8") as f:
            json.dump(best_hp.values, f, indent=2, ensure_ascii=False)

        history = refit_best_model(cfg, best_model, x_train, x_val)
        best_model.save(model_path)

        plot_loss(history, os.path.join(out_dirs["figs"], "loss_curve.png"))

        train_mae_seq = reconstruction_mae_per_seq(best_model, x_train_full, cfg.BATCH_SIZE)
        mae_seq_all   = reconstruction_mae_per_seq(best_model, x_all, cfg.BATCH_SIZE)

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
    elif cfg.ADAPTIVE_THRESHOLD_MODE.lower() == "rolling_p99":
        anomaly_seq, monthly_thresholds = apply_rolling_percentile_threshold(
            mae_seq=mae_seq_all,
            index=df_all_z.index,
            time_steps=cfg.TIME_STEPS,
            stride=cfg.STRIDE,
            window_days=cfg.ADAPTIVE_THRESHOLD_WINDOW_DAYS,
            percentile=cfg.ADAPTIVE_THRESHOLD_PERCENTILE,
        )
        print(f"[ROLLING-THR] window={cfg.ADAPTIVE_THRESHOLD_WINDOW_DAYS}d p{cfg.ADAPTIVE_THRESHOLD_PERCENTILE:g} fallback={monthly_thresholds['rolling_p99_global_fallback']:.5f}")
    elif cfg.ALARM_POLICY.lower() == "cusum":
        anomaly_seq, _cusum = apply_cusum_alarm(mae_seq_all, train_mae_seq, cfg.CUSUM_K, cfg.CUSUM_H)
        print(f"[CUSUM] k={cfg.CUSUM_K} h={cfg.CUSUM_H} disparos={int(anomaly_seq.sum())}/{len(anomaly_seq)}")
    elif cfg.ENABLE_REGIME_BAND_THRESHOLD and _running_col_series is not None and cfg.REGIME_BANDS:
        train_bands = assign_regime_bands(len(train_mae_seq), df_normal_z.index, cfg.TIME_STEPS, cfg.STRIDE, _running_col_series, cfg.REGIME_BANDS)
        all_bands   = assign_regime_bands(len(mae_seq_all),  df_all_z.index,    cfg.TIME_STEPS, cfg.STRIDE, _running_col_series, cfg.REGIME_BANDS)
        band_thr, gthr = compute_regime_band_thresholds(train_mae_seq, train_bands, cfg.THRESH_MODE, cfg.TARGET_ANOMALY_RATE, cfg.REGIME_BAND_MIN_SAMPLES)
        anomaly_seq, _thr_seq = apply_regime_band_threshold(mae_seq_all, all_bands, band_thr, gthr)
        print(f"[REGIME-THR] col={cfg.RUNNING_COL} bands={cfg.REGIME_BANDS} thr_por_banda={ {k: round(v,4) for k,v in sorted(band_thr.items())} } global={gthr:.4f}")
    else:
        anomaly_seq = mae_seq_all > threshold

    if cfg.DEBOUNCE_POINTS and cfg.DEBOUNCE_POINTS > 1:
        _before_db = int(np.sum(anomaly_seq))
        anomaly_seq = apply_debounce(anomaly_seq, cfg.DEBOUNCE_POINTS)
        print(f"[DEBOUNCE] n={cfg.DEBOUNCE_POINTS}: {_before_db} -> {int(np.sum(anomaly_seq))} sequências anômalas")
    state = None
    if _running_col_series is not None:
        # RUNNING_COL tem prioridade: usa NGP_A (ou similar) como indicador de operação.
        # Espelha o comportamento do treino e de pipeline_multi.py.
        state = _running_col_series.reindex(df_all_z.index).fillna(0.0).map(
            lambda x: "on" if x > cfg.RUNNING_THRESHOLD else "off"
        )
        anomaly_seq = mask_anomaly_seq_by_operational_state(
            anomaly_seq=anomaly_seq,
            index=df_all_z.index,
            time_steps=cfg.TIME_STEPS,
            state=state,
            stride=cfg.STRIDE,
        )
    elif cfg.ENABLE_OPERATIONAL_MASK:
        # Fallback: infere estado pelo valor do próprio sensor (para sensores sem RUNNING_COL).
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

    # Supressão de scoring durante spikes de gradiente.
    # Reduz FP causados por transientes abruptos sem comprometer recall
    # (anomalias reais têm MAE elevado por horas, não por instantes).
    _suppress_sensors = getattr(cfg, "GRADIENT_SPIKE_SUPPRESS_SENSORS", None)
    _scoring_suppress = (
        cfg.ENABLE_GRADIENT_SPIKE_MASK
        and getattr(cfg, "GRADIENT_SPIKE_SUPPRESS_SCORING", False)
        and (_suppress_sensors is None or sensor in _suppress_sensors)
    )
    if _scoring_suppress:
        spike_mask_full = build_gradient_spike_mask(
            df_all, sensor, _running_col_series,
            cfg.GRADIENT_SPIKE_STD_MULT,
            cfg.GRADIENT_SPIKE_SUPPRESS_MINUTES,
            getattr(cfg, "GRADIENT_SPIKE_TRANSITION_MINUTES", 0),
        ).reindex(df_all_z.index).fillna(False)
        stride_int = max(1, int(cfg.STRIDE))
        seq_positions = np.arange(len(anomaly_seq)) * stride_int
        valid_pos = seq_positions[seq_positions < len(df_all_z.index)]
        seq_spike = np.zeros(len(anomaly_seq), dtype=bool)
        seq_spike[:len(valid_pos)] = spike_mask_full.iloc[valid_pos].values
        n_suppressed = int((anomaly_seq & seq_spike).sum())
        if n_suppressed:
            print(f"[GRAD-SPIKE-SCORE] {sensor}: {n_suppressed} sequências suprimidas no scoring.")
        anomaly_seq = anomaly_seq & ~seq_spike

    if getattr(cfg, "MIN_ANOMALY_RUN_STEPS", 0) > 1:
        _before_run = int(np.asarray(anomaly_seq).sum())
        anomaly_seq = filter_short_anomaly_runs(anomaly_seq, cfg.MIN_ANOMALY_RUN_STEPS)
        print(f"[MIN-RUN] min_steps={cfg.MIN_ANOMALY_RUN_STEPS}: {_before_run} -> {int(np.asarray(anomaly_seq).sum())} sequências")

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

    # Não flagar anomalia em gaps longos: esses pontos são valores fabricados pela
    # interpolação (além de INTERPOLATE_LIMIT), não dado real → alarme seria ruído.
    if cfg.EXCLUDE_LONG_GAPS_FROM_SCORING:
        _lg = long_gap_mask.reindex(df_point.index).fillna(False)
        _n = int((df_point["is_anom_point"].astype(bool) & _lg).sum())
        df_point.loc[_lg, "is_anom_point"] = 0
        if _n:
            print(f"[LONG-GAP-SCORE] {sensor}: {_n} anomalias suprimidas em gaps longos (dado interpolado).")

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
    _arch_label = getattr(cfg, "MODEL_ARCH", "cnn1d").upper()
    plot_series_with_anomalies(
        df_all[sensor],
        anomalous_times,
        os.path.join(out_dirs["figs"], "series_with_anomalies.png"),
        title=f"Serie + anomalias ({_arch_label}) | sensor={sensor}",
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

    # ------------------------------------------------------------------
    # Camada preditiva (métrica de produção): EWMA do MAE → episódios com
    # debounce → recall sobre incidentes genuínos (PREDICTIVE_INCIDENT_CONDITIONS,
    # gap 4h, onset em ON) no ponto de operação sob orçamento de FA/dia.
    # É a métrica PRINCIPAL do summary; o hit_rate pontual abaixo é debug
    # (threshold fixo, sem EWMA, denominador com todas as linhas de alarme).
    # ------------------------------------------------------------------
    predictive_summary: dict = {}
    pred_headline: dict = {}
    if cfg.ENABLE_PREDICTIVE_LAYER:
        incidents = extract_incidents(
            df_alarm_sensor,
            priorities=cfg.PREDICTIVE_INCIDENT_PRIORITY,
            conditions=getattr(cfg, "PREDICTIVE_INCIDENT_CONDITIONS", []) or None,
            incident_gap_hours=cfg.ALARM_F2_INCIDENT_GAP_HOURS,
        )
        if len(incidents):
            _idx_tz = getattr(all_index, "tz", None)
            if incidents.tz is None and _idx_tz is not None:
                incidents = incidents.tz_localize(_idx_tz)
            elif incidents.tz is not None and _idx_tz is None:
                incidents = incidents.tz_localize(None)
            n_total = len(incidents)
            incidents = incidents[(incidents >= all_index.min()) & (incidents <= all_index.max())]
            n_range = len(incidents)
            # Onset em OFF: fora do escopo do modelo (máscara operacional) — sai do denominador
            if state is not None and len(incidents):
                on_at_inc = state.reindex(incidents, method="nearest").eq("on").to_numpy()
                incidents = incidents[on_at_inc]
            print(f"[PRED] {sensor}: incidentes genuinos={n_total} no_range={n_range} ON={len(incidents)}")
        if len(incidents):
            stride_int = max(1, int(cfg.STRIDE))
            seq_starts = np.arange(len(mae_seq_all)) * stride_int
            seq_ends_pos = np.clip(seq_starts + cfg.TIME_STEPS - 1, 0, len(all_index) - 1)
            seq_end_seconds = pd.DatetimeIndex(all_index[seq_ends_pos]).values.astype("datetime64[s]").astype("int64")
            if state is not None:
                rv = state.eq("on").to_numpy(dtype=float)
                seq_run_frac = np.array([rv[s:s + cfg.TIME_STEPS].mean() for s in seq_starts])
            else:
                seq_run_frac = np.ones(len(mae_seq_all))
            seq_run_full = seq_run_frac >= 0.999
            dt_seconds = stride_int * _infer_sampling_interval_seconds(all_index)
            _hl = (cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS_PER_SENSOR or {}).get(
                sensor, cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS)
            health_ewma = compute_health_index_ewma(
                mae_seq_all, seq_run_frac, half_life_hours=float(_hl), dt_seconds=dt_seconds)
            inc_seconds = pd.DatetimeIndex(incidents).values.astype("datetime64[s]").astype("int64").astype(float)
            for h in cfg.PREDICTIVE_HORIZONS_HOURS:
                curve = compute_predictive_curve(
                    health_ewma=health_ewma,
                    seq_running_full=seq_run_full,
                    t_end_seconds=seq_end_seconds.astype(float),
                    incident_seconds=inc_seconds,
                    horizon_hours=float(h),
                    debounce_hours=cfg.PREDICTIVE_ALERT_DEBOUNCE_HOURS,
                    sigma_y_min=getattr(cfg, "PREDICTIVE_SIGMA_Y_MIN", 0.5),
                    sigma_y_max=getattr(cfg, "PREDICTIVE_SIGMA_Y_MAX", 5.0),
                )
                if curve is not None and len(curve):
                    curve.to_csv(os.path.join(out_dirs["csv"], f"predictive_curve_H{int(h)}h.csv"), index=False)
                op = pick_operating_point(curve, cfg.PREDICTIVE_FA_BUDGET_PER_DAY)
                if op:
                    op["n_incidents_on"] = int(len(incidents))
                    predictive_summary[f"H{int(h)}h"] = op
                    print(f"[PRED] {sensor} H={int(h)}h | recall={op['recall']:.2f} "
                          f"fa/dia={op['fa_per_day']:.3f} lead={op['median_lead_hours']:.1f}h "
                          f"eps={int(op['n_episodes'])}")
            _hs = [float(h) for h in cfg.PREDICTIVE_HORIZONS_HOURS]
            _h_head = 8.0 if 8.0 in _hs else (_hs[0] if _hs else None)
            _op_head = predictive_summary.get(f"H{int(_h_head)}h") if _h_head is not None else None
            if _op_head:
                pred_headline = {
                    "pred_horizon_hours": float(_h_head),
                    "pred_n_incidents_on": int(_op_head["n_incidents_on"]),
                    "pred_recall": float(_op_head["recall"]),
                    "pred_fa_per_day": float(_op_head["fa_per_day"]),
                    "pred_median_lead_hours": float(_op_head["median_lead_hours"]),
                }
        else:
            # Sem incidente genuíno ON na janela: não há denominador — recall
            # indefinido (não zero). FP continua visível via fa_per_day do debug.
            pred_headline = {"pred_n_incidents_on": 0}

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
        "predictive_operating_points": predictive_summary,
        **pred_headline,
        **eval_stats,
        **warn_eval_stats,
    }
    if state is not None:
        counts = state.value_counts().to_dict()
        calibration_report["operational_state_counts"] = {str(k): int(v) for k, v in counts.items()}
    with open(os.path.join(out_dirs["csv"], "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    # Bundle de inferência: scaler + threshold + parâmetros para reproduzir o
    # scoring em dados novos (sem refitar). Sem ele, a normalização recalculada
    # numa nova distribuição invalida o threshold calibrado.
    _hl_over = (cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS_PER_SENSOR or {}).get(sensor)
    inference_bundle = {
        "sensor": sensor,
        "model_arch": getattr(cfg, "MODEL_ARCH", "cnn1d"),
        "model_file": "model.keras",
        "feature_columns": [str(c) for c in df_normal_z.columns],
        "n_features": int(df_normal_z.shape[1]),
        "time_steps": int(cfg.TIME_STEPS),
        "stride": int(cfg.STRIDE),
        "normalize_mode": cfg.NORMALIZE_MODE,
        "center": {str(k): float(v) for k, v in center.to_dict().items()},
        "scale": {str(k): float(v) for k, v in scale.to_dict().items()},
        "outlier_mode": cfg.OUTLIER_MODE,
        "clip_bounds": clip_bounds,
        "threshold": float(threshold),
        "thresh_mode": cfg.THRESH_MODE,
        "monthly_thresholds": {str(k): float(v) for k, v in monthly_thresholds.items()},
        "running_col": cfg.RUNNING_COL,
        "running_threshold": float(cfg.RUNNING_THRESHOLD),
        "predictive_ewma_half_life_hours": float(
            _hl_over if _hl_over is not None else cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS
        ),
        "alarm_policy": cfg.ALARM_POLICY,
        "point_rule": cfg.POINT_RULE,
        "point_window": int(cfg.POINT_WINDOW),
        "point_min_count": int(cfg.POINT_MIN_COUNT),
    }
    with open(os.path.join(out_dirs["best_model"], "inference_bundle.json"), "w", encoding="utf-8") as f:
        json.dump(inference_bundle, f, indent=2, ensure_ascii=False)

    report = {
        "sensor": sensor,
        "output_dir": out_dirs["root"],
        "model_arch": getattr(cfg, "MODEL_ARCH", "cnn1d"),
        "model_path": model_path if arch != "isolation_forest" else None,
        "threshold": float(threshold),
        "THRESH_MODE": cfg.THRESH_MODE,
        **pred_headline,
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
