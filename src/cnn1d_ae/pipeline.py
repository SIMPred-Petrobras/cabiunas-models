from __future__ import annotations

import os
import json
import re
from dataclasses import asdict, replace
from typing import Dict, List, Any, Set, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import ensure_sensor_dirs, save_run_config, load_data, resolve_output_dir
from .model import setup_gpu
from .preprocess import (
    build_sensor_dataframe,
    build_group_dataframe,
    build_exclusion_mask,
    clip_outliers,
    normalize_train_only,
)
from .sequences import make_sequences, train_val_split
from .tuning import run_tuner, refit_best_model
from .scoring import (
    reconstruction_mae_per_seq,
    compute_threshold,
    map_seq_to_point_anomalies,
    build_sequence_scores_df,
    compute_anomaly_rate_per_day,
    build_operational_state,
    mask_anomaly_seq_by_operational_state,
)
from .plots import (
    plot_loss,
    plot_hist_mae,
    plot_series_with_anomalies,
    plot_series_alarm_anomaly_subplots,
    plot_series_failure_zoom,
    plot_signal_mae_anomaly,
)
from .model_card import write_model_card


def parse_failure_dates(failure_date: str) -> List[pd.Timestamp]:
    """Converte o campo FAILURE_DATE (datas separadas por ';', com ou sem hora)
    em uma lista de Timestamps. Entradas inválidas são ignoradas."""
    if not failure_date:
        return []
    out: List[pd.Timestamp] = []
    for part in str(failure_date).split(";"):
        part = part.strip()
        if not part:
            continue
        ts = pd.to_datetime(part, errors="coerce", dayfirst=False)
        if pd.notna(ts):
            out.append(ts)
    return out


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
    win = pd.Timedelta(minutes=minutes)
    hits = 0
    for t in df_alarm["Data da Ocorrencia"]:
        t0 = t - win
        t1 = t + win
        if df_point.loc[(df_point.index >= t0) & (df_point.index <= t1), "is_anom_point"].sum() > 0:
            hits += 1

    total = len(df_alarm)
    hit_rate = hits / total if total > 0 else np.nan
    return {
        "n_alarms": int(total),
        "alarms_with_detected_anomaly_in_window": int(hits),
        "hit_rate": float(hit_rate) if np.isfinite(hit_rate) else None,
    }


def run_one_sensor(cfg: PipelineConfig, df_alarm: pd.DataFrame, df_feat: pd.DataFrame, df_raw: pd.DataFrame, sensor: str) -> Dict:
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

    df_normal = df_use.loc[~exclude].copy()
    df_all = df_use.copy()

    if len(df_normal) < cfg.TIME_STEPS + 10:
        print(f"[SKIP] {sensor} (poucos dados normais apos exclusao)")
        return {"sensor": sensor, "skipped": True, "reason": "few_normal_points"}

    df_normal = clip_outliers(df_normal, cfg)
    df_all = clip_outliers(df_all, cfg)

    df_normal_z, df_all_z, _, _ = normalize_train_only(cfg, df_normal, df_all)

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

    plot_loss(history, os.path.join(out_dirs["figs"], "loss_curve.png"), title=f"sensor={sensor}")

    train_mae_seq = reconstruction_mae_per_seq(best_model, x_train_full, cfg.BATCH_SIZE)
    threshold = compute_threshold(train_mae_seq, cfg.THRESH_MODE, target_rate=cfg.TARGET_ANOMALY_RATE)

    plot_hist_mae(train_mae_seq, threshold, os.path.join(out_dirs["figs"], "train_mae_hist.png"))

    x_all = make_sequences(values_all, cfg.TIME_STEPS, cfg.STRIDE)
    mae_seq_all = reconstruction_mae_per_seq(best_model, x_all, cfg.BATCH_SIZE)
    anomaly_seq = mae_seq_all > threshold
    state = None
    if cfg.ENABLE_OPERATIONAL_MASK:
        ref_sensor = cfg.OPERATIONAL_REF_SENSOR
        if ref_sensor and ref_sensor != sensor:
            df_ref, _ = build_sensor_dataframe(cfg, df_feat, df_raw, ref_sensor)
            ref_series = df_ref[ref_sensor]
        else:
            ref_series = df_all[sensor]
        state = build_operational_state(
            index=df_all_z.index,
            sensor_series=ref_series,
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
        )

    all_index = df_all_z.index
    df_seq_scores = build_sequence_scores_df(all_index, mae_seq_all, anomaly_seq)
    df_seq_scores.to_csv(os.path.join(out_dirs["csv"], "sequence_scores_all.csv"), index=False)

    df_point = map_seq_to_point_anomalies(
        anomaly_seq,
        all_index,
        cfg.TIME_STEPS,
        cfg.POINT_RULE,
        cfg.POINT_WINDOW,
        cfg.POINT_MIN_COUNT,
    )
    if state is not None:
        df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
        df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0
    df_point.to_csv(os.path.join(out_dirs["csv"], "point_anomalies_all.csv"))

    anomalous_times = df_point.index[df_point["is_anom_point"] == 1]
    plot_series_with_anomalies(
        df_all[sensor],
        anomalous_times,
        os.path.join(out_dirs["figs"], "series_with_anomalies.png"),
        title=f"Serie + anomalias (CNN1D-AE) | sensor={sensor}",
        operational_state=state,
    )
    alarm_series = (
        df_alarm_sensor["Data da Ocorrencia"]
        if "Data da Ocorrencia" in df_alarm_sensor.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    plot_series_alarm_anomaly_subplots(
        df_all[sensor],
        anomalous_times,
        alarm_series,
        os.path.join(out_dirs["figs"], "series_alarm_anomaly_subplots.png"),
        title=f"{sensor}",
        operational_state=state,
    )

    # Série do MAE (erro de reconstrução) alinhada ao tempo, p/ o plot duplo-eixo.
    mae_series = pd.Series(mae_seq_all, index=all_index[: len(mae_seq_all)])
    _title = f"{cfg.EQUIPMENT_ID or ''} | {sensor}".strip(" |")

    failure_times = parse_failure_dates(cfg.FAILURE_DATE)
    if failure_times:
        plot_series_failure_zoom(
            df_all[sensor],
            anomalous_times,
            alarm_series,
            failure_times,
            os.path.join(out_dirs["figs"], "series_failure_zoom.png"),
            title=_title,
            zoom_days=cfg.FAILURE_ZOOM_DAYS,
            operational_state=state,
        )

    # Plots duplo-eixo (bruto + MAE + anomalia): visão geral e zoom(s) na falha.
    plot_signal_mae_anomaly(
        df_all[sensor], mae_series, float(threshold), anomalous_times,
        os.path.join(out_dirs["figs"], "signal_mae_anomaly.png"),
        title=_title, windows=None, failure_times=failure_times,
        alarm_times=alarm_series, operational_state=state,
    )
    if failure_times:
        z = pd.Timedelta(days=float(cfg.FAILURE_ZOOM_DAYS))
        wins = [(f - z, f + z, f"falha {pd.Timestamp(f).strftime('%d/%m/%Y %H:%M')}") for f in failure_times]
        plot_signal_mae_anomaly(
            df_all[sensor], mae_series, float(threshold), anomalous_times,
            os.path.join(out_dirs["figs"], "signal_mae_anomaly_zoom.png"),
            title=_title, windows=wins, failure_times=failure_times,
            alarm_times=alarm_series, operational_state=state,
        )

    eval_stats = eval_alarm_hit_rate(df_alarm_sensor, df_point, cfg.EXCLUDE_MINUTES_AROUND_ALARM)

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
        **eval_stats,
    }
    if state is not None:
        counts = state.value_counts().to_dict()
        calibration_report["operational_state_counts"] = {str(k): int(v) for k, v in counts.items()}
    with open(os.path.join(out_dirs["csv"], "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    try:
        card_path = write_model_card(
            cfg, out_dirs,
            kind="sensor", name=sensor, sensors=[sensor],
            best_hp=best_hp.values, threshold=float(threshold),
            calibration_report=calibration_report, failure_times=failure_times,
            n_train_seq=int(len(x_train_full)), n_features=int(n_features),
            data_period=f"{df_all.index.min()} → {df_all.index.max()}",
        )
        print(f"[CARD] {card_path}")
    except Exception as exc:
        print(f"[WARN] falha ao gerar MODEL_CARD para {sensor}: {exc}")

    report = {
        "sensor": sensor,
        "output_dir": out_dirs["root"],
        "model_path": model_path,
        "card_path": os.path.join(out_dirs["root"], "MODEL_CARD.md"),
        "threshold": float(threshold),
        "THRESH_MODE": cfg.THRESH_MODE,
        "anomaly_rate_points_per_day": calibration_report.get("anomaly_rate_points_per_day"),
        **eval_stats,
        "skipped": False,
    }
    with open(os.path.join(out_dirs["csv"], "evaluation_alarm_hit_rate.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def run_one_group(
    cfg: PipelineConfig,
    df_alarm: pd.DataFrame,
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    group: Dict[str, Any],
) -> Dict:
    """
    Treina um autoencoder multicanal para um grupo de sensores fisicamente
    relacionados. Suporta overrides por grupo:
      time_steps, stride, thresh_mode, target_anomaly_rate,
      point_window, point_min_count
    """
    group_name = group["name"]
    sensors = list(group["sensors"])

    # Monta config efetivo com overrides do grupo
    _OVERRIDE_KEYS = {
        "time_steps": "TIME_STEPS",
        "stride": "STRIDE",
        "thresh_mode": "THRESH_MODE",
        "target_anomaly_rate": "TARGET_ANOMALY_RATE",
        "point_window": "POINT_WINDOW",
        "point_min_count": "POINT_MIN_COUNT",
    }
    overrides = {cfg_key: group[json_key]
                 for json_key, cfg_key in _OVERRIDE_KEYS.items()
                 if json_key in group}
    effective_cfg = replace(cfg, **overrides) if overrides else cfg

    out_dirs = ensure_sensor_dirs(cfg, group_name)
    save_run_config(effective_cfg, out_dirs)
    with open(os.path.join(out_dirs["csv"], "group_definition.json"), "w", encoding="utf-8") as f:
        json.dump(group, f, indent=2, ensure_ascii=False)

    model_path = os.path.join(out_dirs["best_model"], "model.keras")
    if (not cfg.OVERWRITE) and os.path.exists(model_path):
        print(f"[SKIP] group={group_name} (modelo ja existe: {model_path})")
        return {"group": group_name, "sensors": sensors, "skipped": True,
                "reason": "model_exists", "model_path": model_path}

    print("\n==============================")
    print(f"[GROUP]    {group_name}")
    print(f"[SENSORS]  {sensors}")
    print(f"[TIME_STEPS] {effective_cfg.TIME_STEPS}")
    print(f"[OUT]      {out_dirs['root']}")
    print("==============================")

    df_use, long_gap_mask = build_group_dataframe(cfg, df_feat, df_raw, sensors)

    # Exclui sensores travados (std muito baixo) do grupo
    valid_sensors = [s for s in sensors if float(df_use[s].std()) >= cfg.MIN_STD]
    dropped = set(sensors) - set(valid_sensors)
    if dropped:
        print(f"[WARN] group={group_name}: sensores com std baixo removidos: {dropped}")
    if not valid_sensors:
        return {"group": group_name, "sensors": sensors, "skipped": True,
                "reason": "all_sensors_low_std"}
    sensors = valid_sensors
    df_use = df_use[sensors]

    # Sensor alvo: threshold e detecção baseados no MAE deste canal.
    # Se não definido, usa MAE global (média de todos os canais).
    target_sensor = group.get("target_sensor")
    if target_sensor and target_sensor not in sensors:
        print(f"[WARN] group={group_name}: target_sensor={target_sensor!r} removido por low_std — usando MAE global")
        target_sensor = None
    target_idx = sensors.index(target_sensor) if target_sensor else None
    if target_idx is not None:
        print(f"[TARGET] sensor alvo: {target_sensor!r} (canal {target_idx}) — anomalia baseada no MAE deste canal")

    # Máscara de alarmes: união de todos os sensores do grupo
    if "Tag" in df_alarm.columns:
        df_alarm_group = df_alarm.loc[df_alarm["Tag"].isin(sensors)].copy()
    else:
        df_alarm_group = df_alarm.copy()
    if "Data da Ocorrencia" in df_alarm_group.columns:
        df_alarm_group = df_alarm_group.dropna(subset=["Data da Ocorrencia"]).sort_values("Data da Ocorrencia")

    exclude_alarm = build_exclusion_mask(
        df_use.index,
        df_alarm_group["Data da Ocorrencia"] if "Data da Ocorrencia" in df_alarm_group.columns
        else pd.Series(dtype="datetime64[ns]"),
        cfg.EXCLUDE_MINUTES_AROUND_ALARM,
    )

    exclude = exclude_alarm.copy()
    if cfg.EXCLUDE_LONG_GAPS_FROM_TRAIN:
        long_gap_mask = long_gap_mask.reindex(df_use.index).fillna(False)
        exclude = exclude | long_gap_mask

    df_normal = df_use.loc[~exclude].copy()
    df_all = df_use.copy()

    if len(df_normal) < effective_cfg.TIME_STEPS + 10:
        print(f"[SKIP] group={group_name} (poucos dados normais: {len(df_normal)})")
        return {"group": group_name, "sensors": sensors, "skipped": True,
                "reason": "few_normal_points"}

    df_normal = clip_outliers(df_normal, cfg)
    df_all = clip_outliers(df_all, cfg)

    df_normal_z, df_all_z, _, _ = normalize_train_only(cfg, df_normal, df_all)

    values_normal = df_normal_z.values.astype(np.float32)
    values_all = df_all_z.values.astype(np.float32)

    x_train_full = make_sequences(values_normal, effective_cfg.TIME_STEPS, effective_cfg.STRIDE)
    x_train, x_val = train_val_split(
        x_train_full, cfg.VAL_FRAC, cfg.SHUFFLE_TRAIN, cfg.RANDOM_SEED,
        split_mode=cfg.SPLIT_MODE,
    )
    n_features = x_train.shape[-1]

    best_hp, best_model, df_trials = run_tuner(effective_cfg, out_dirs, x_train, x_val, n_features)
    df_trials.to_csv(os.path.join(out_dirs["csv"], "trials_ranking.csv"), index=False)

    with open(os.path.join(out_dirs["best_model"], "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hp.values, f, indent=2, ensure_ascii=False)

    history = refit_best_model(effective_cfg, best_model, x_train, x_val)
    best_model.save(model_path)

    plot_loss(history, os.path.join(out_dirs["figs"], "loss_curve.png"), title=f"grupo={group_name}")

    # Inferência única — deriva MAE global e por canal ao mesmo tempo
    x_train_pred = best_model.predict(x_train_full, batch_size=effective_cfg.BATCH_SIZE, verbose=0)
    train_abs_err = np.abs(x_train_pred - x_train_full)
    train_mae_seq = np.mean(train_abs_err, axis=(1, 2))

    # Se target_sensor definido, usa MAE desse canal para o threshold
    train_mae_thresh = (np.mean(train_abs_err[:, :, target_idx], axis=1)
                        if target_idx is not None else train_mae_seq)
    threshold = compute_threshold(train_mae_thresh, effective_cfg.THRESH_MODE,
                                  target_rate=effective_cfg.TARGET_ANOMALY_RATE)
    plot_hist_mae(train_mae_thresh, threshold, os.path.join(out_dirs["figs"], "train_mae_hist.png"))

    x_all = make_sequences(values_all, effective_cfg.TIME_STEPS, effective_cfg.STRIDE)
    x_all_pred = best_model.predict(x_all, batch_size=effective_cfg.BATCH_SIZE, verbose=0)
    abs_err_all = np.abs(x_all_pred - x_all)
    mae_seq_all = np.mean(abs_err_all, axis=(1, 2))          # (n_seq,) — MAE global
    mae_per_ch = np.mean(abs_err_all, axis=1)                 # (n_seq, n_features) — mean no eixo tempo
    mae_for_anom = mae_per_ch[:, target_idx] if target_idx is not None else mae_seq_all

    anomaly_seq = mae_for_anom > threshold

    state = None
    if cfg.ENABLE_OPERATIONAL_MASK:
        ref_sensor = cfg.OPERATIONAL_REF_SENSOR
        if ref_sensor and ref_sensor not in sensors:
            df_ref, _ = build_sensor_dataframe(cfg, df_feat, df_raw, ref_sensor)
            ref_series = df_ref[ref_sensor]
        else:
            # Usa o sensor de referência se estiver no grupo, senão o primeiro sensor
            ref_col = ref_sensor if (ref_sensor and ref_sensor in sensors) else sensors[0]
            ref_series = df_all[ref_col]
        state = build_operational_state(
            index=df_all_z.index,
            sensor_series=ref_series,
            off_value_quantile=cfg.OFF_VALUE_QUANTILE,
            off_abs_threshold=cfg.OFF_ABS_THRESHOLD,
            off_long_min_hours=cfg.OFF_LONG_MIN_HOURS,
            transient_padding_minutes=cfg.TRANSIENT_PADDING_MINUTES,
            transient_diff_quantile=cfg.TRANSIENT_DIFF_QUANTILE,
        )
        anomaly_seq = mask_anomaly_seq_by_operational_state(
            anomaly_seq=anomaly_seq,
            index=df_all_z.index,
            time_steps=effective_cfg.TIME_STEPS,
            state=state,
        )

    all_index = df_all_z.index
    df_seq_scores = build_sequence_scores_df(all_index, mae_for_anom, anomaly_seq)
    # Colunas de MAE por canal — útil para diagnosticar qual sensor disparou
    for i, s in enumerate(sensors):
        col = np.full(len(df_seq_scores), np.nan)
        n = min(len(mae_per_ch), len(df_seq_scores))
        col[:n] = mae_per_ch[:n, i]
        df_seq_scores[f"mae_{s}"] = col
    df_seq_scores.to_csv(os.path.join(out_dirs["csv"], "sequence_scores_all.csv"), index=False)

    df_point = map_seq_to_point_anomalies(
        anomaly_seq, all_index, effective_cfg.TIME_STEPS,
        cfg.POINT_RULE, effective_cfg.POINT_WINDOW, effective_cfg.POINT_MIN_COUNT,
    )
    if state is not None:
        df_point["operational_state"] = state.reindex(df_point.index).fillna("on")
        df_point.loc[df_point["operational_state"] != "on", "is_anom_point"] = 0
    df_point.to_csv(os.path.join(out_dirs["csv"], "point_anomalies_all.csv"))

    anomalous_times = df_point.index[df_point["is_anom_point"] == 1]

    # A decisão de anomalia vem de UM canal só (target_sensor) ou do MAE global
    # (se não houver alvo). Para não confundir "canal de contexto" com "detecção
    # própria", o alvo ganha o conjunto completo de plots em destaque
    # (prefixo TARGET_, na raiz de figs/); os demais viram apoio leve em
    # figs/contexto/ (mesma anomalia compartilhada, só para auditoria visual).
    contexto_dir = os.path.join(out_dirs["figs"], "contexto")
    os.makedirs(contexto_dir, exist_ok=True)
    failure_times = parse_failure_dates(cfg.FAILURE_DATE)

    for i, s in enumerate(sensors):
        safe_name = s.replace("/", "_").replace("\\", "_")
        is_target = (target_sensor is not None and s == target_sensor)
        if "Tag" in df_alarm.columns and "Data da Ocorrencia" in df_alarm.columns:
            s_alarm_times = df_alarm.loc[df_alarm["Tag"] == s, "Data da Ocorrencia"].dropna()
        elif "Data da Ocorrencia" in df_alarm.columns:
            s_alarm_times = df_alarm["Data da Ocorrencia"].dropna()
        else:
            s_alarm_times = pd.Series(dtype="datetime64[ns]")
        mae_series_s = pd.Series(mae_per_ch[:, i], index=all_index[: len(mae_per_ch)])

        if not is_target:
            # Contexto: só o painel duplo-eixo (mais informativo num único PNG).
            plot_signal_mae_anomaly(
                df_all[s], mae_series_s, float(threshold), anomalous_times,
                os.path.join(contexto_dir, f"{safe_name}.png"),
                title=f"[CONTEXTO] {group_name} | {s} (anomalia decidida pelo alvo: {target_sensor or 'MAE global'})",
                windows=None, failure_times=failure_times,
                alarm_times=s_alarm_times, operational_state=state,
            )
            continue

        # Alvo: conjunto completo de plots, em destaque (prefixo TARGET_).
        prefix = f"TARGET_{safe_name}"
        plot_series_with_anomalies(
            df_all[s],
            anomalous_times,
            os.path.join(out_dirs["figs"], f"{prefix}_series_with_anomalies.png"),
            title=f"[ALVO] Serie + anomalias (CNN1D-AE) | grupo={group_name} | sensor={s}",
            operational_state=state,
        )
        plot_series_alarm_anomaly_subplots(
            df_all[s],
            anomalous_times,
            s_alarm_times,
            os.path.join(out_dirs["figs"], f"{prefix}_series_alarm_anomaly_subplots.png"),
            title=f"[ALVO] {group_name} | {s}",
            operational_state=state,
        )
        plot_signal_mae_anomaly(
            df_all[s], mae_series_s, float(threshold), anomalous_times,
            os.path.join(out_dirs["figs"], f"{prefix}_signal_mae_anomaly.png"),
            title=f"[ALVO] {group_name} | {s}", windows=None, failure_times=failure_times,
            alarm_times=s_alarm_times, operational_state=state,
        )
        if failure_times:
            z = pd.Timedelta(days=float(cfg.FAILURE_ZOOM_DAYS))
            wins = [(f - z, f + z, f"falha {pd.Timestamp(f).strftime('%d/%m/%Y %H:%M')}") for f in failure_times]
            plot_signal_mae_anomaly(
                df_all[s], mae_series_s, float(threshold), anomalous_times,
                os.path.join(out_dirs["figs"], f"{prefix}_signal_mae_anomaly_zoom.png"),
                title=f"[ALVO] {group_name} | {s}", windows=wins, failure_times=failure_times,
                alarm_times=s_alarm_times, operational_state=state,
            )
            plot_series_failure_zoom(
                df_all[s],
                anomalous_times,
                s_alarm_times,
                failure_times,
                os.path.join(out_dirs["figs"], f"{prefix}_series_failure_zoom.png"),
                title=f"[ALVO] {group_name} | {s}",
                zoom_days=cfg.FAILURE_ZOOM_DAYS,
                operational_state=state,
            )

    if target_sensor is None:
        print(f"[WARN] group={group_name}: sem target_sensor definido — todos os sensores "
              f"ficaram em figs/contexto/ (nenhum plot em destaque; anomalia = MAE global do grupo)")

    eval_stats = eval_alarm_hit_rate(df_alarm_group, df_point, cfg.EXCLUDE_MINUTES_AROUND_ALARM)

    calibration_report = {
        "group": group_name,
        "sensors": sensors,
        "target_sensor": target_sensor or "global_mae",
        "n_sensors": len(sensors),
        "threshold": float(threshold),
        "THRESH_MODE": effective_cfg.THRESH_MODE,
        "TIME_STEPS": int(effective_cfg.TIME_STEPS),
        "TARGET_ANOMALY_RATE": float(effective_cfg.TARGET_ANOMALY_RATE),
        "POINT_RULE": cfg.POINT_RULE,
        "POINT_WINDOW": int(effective_cfg.POINT_WINDOW),
        "POINT_MIN_COUNT": int(effective_cfg.POINT_MIN_COUNT),
        "anomaly_rate_points_per_day": compute_anomaly_rate_per_day(df_point),
        "operational_mask_enabled": bool(cfg.ENABLE_OPERATIONAL_MASK),
        **eval_stats,
    }
    if state is not None:
        calibration_report["operational_state_counts"] = {
            str(k): int(v) for k, v in state.value_counts().to_dict().items()
        }
    with open(os.path.join(out_dirs["csv"], "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2, ensure_ascii=False)

    try:
        card_path = write_model_card(
            effective_cfg, out_dirs,
            kind="group", name=group_name, sensors=sensors,
            best_hp=best_hp.values, threshold=float(threshold),
            calibration_report=calibration_report,
            failure_times=parse_failure_dates(cfg.FAILURE_DATE),
            n_train_seq=int(len(x_train_full)), n_features=int(n_features),
            data_period=f"{df_all.index.min()} → {df_all.index.max()}",
        )
        print(f"[CARD] {card_path}")
    except Exception as exc:
        print(f"[WARN] falha ao gerar MODEL_CARD para grupo {group_name}: {exc}")

    report = {
        "group": group_name,
        "sensors": sensors,
        "output_dir": out_dirs["root"],
        "model_path": model_path,
        "card_path": os.path.join(out_dirs["root"], "MODEL_CARD.md"),
        "threshold": float(threshold),
        "THRESH_MODE": effective_cfg.THRESH_MODE,
        "TIME_STEPS": int(effective_cfg.TIME_STEPS),
        "anomaly_rate_points_per_day": calibration_report.get("anomaly_rate_points_per_day"),
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


def _write_models_index(cfg: PipelineConfig, out_root: str, rows: List[Dict]) -> Optional[str]:
    """Gera MODELS_INDEX.md consolidando todos os modelos treinados na execução."""
    trained = [r for r in rows if not r.get("skipped")]
    if not trained:
        return None

    lines = [
        f"# Modelos treinados — {cfg.EQUIPMENT_ID or 'execução'}",
        "",
        f"- **Falha(s):** {cfg.FAILURE_DATE or '—'}",
        f"- **Modo:** {'multivariado (grupo)' if cfg.SENSOR_GROUPS else 'univariado (por sensor)'}",
        f"- **Total de modelos:** {len(trained)}",
        "",
        "| Modelo | Hit rate | Limiar (MAE) | Anom./dia | Model card |",
        "|---|---|---|---|---|",
    ]
    for r in trained:
        name = r.get("sensor") or r.get("group", "—")
        card = r.get("card_path", "")
        rel = os.path.relpath(card, out_root) if card else ""
        # <...> preserva caminhos com espaços (nomes de sensores) em Markdown.
        link = f"[MODEL_CARD.md](<{rel}>)" if rel else "—"
        hit = r.get("hit_rate")
        thr = r.get("threshold")
        rate = r.get("anomaly_rate_points_per_day", "")
        thr_s = f"{thr:.6g}" if isinstance(thr, (int, float)) else "—"
        lines.append(f"| `{name}` | {hit if hit is not None else '—'} | {thr_s} | {rate} | {link} |")

    lines += ["", "_Índice gerado automaticamente pela pipeline CNN1D-AE (Transpetro)._", ""]
    index_path = os.path.join(out_root, "MODELS_INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return index_path


def run(cfg: PipelineConfig, data=None) -> Dict[str, Any]:
    setup_gpu()
    if data is not None:
        df_alarm, df_feat, df_raw, time_report = data
    else:
        df_alarm, df_feat, df_raw, time_report = load_data(cfg)

    summary_out_root = cfg.OUTPUT_ROOT if cfg.OUTPUT_ROOT else "."
    os.makedirs(summary_out_root, exist_ok=True)
    summary_path = os.path.join(summary_out_root, "summary_all_sensors.csv")
    time_report_path = os.path.join(summary_out_root, "time_integrity_report.json")
    with open(time_report_path, "w", encoding="utf-8") as f:
        json.dump(time_report, f, indent=2, ensure_ascii=False)

    rows: List[Dict] = []
    grouped_sensors: Set[str] = set()

    # --- Grupos de sensores ---
    if cfg.SENSOR_GROUPS:
        for g in cfg.SENSOR_GROUPS:
            grouped_sensors.update(g.get("sensors", []))
        print(f"[GROUPS] {len(cfg.SENSOR_GROUPS)} grupo(s) | {len(grouped_sensors)} sensor(es) em grupos")
        for group in cfg.SENSOR_GROUPS:
            try:
                rows.append(run_one_group(cfg, df_alarm, df_feat, df_raw, group))
            except Exception as e:
                print(f"[ERROR] group={group.get('name')} -> {e}")
                rows.append({"group": group.get("name"), "sensors": group.get("sensors", []),
                             "skipped": True, "reason": f"exception: {e}"})

    # --- Sensores individuais (fora de qualquer grupo) ---
    sensors = discover_sensors(cfg, df_feat, df_raw)
    sensors = [s for s in sensors if s not in grouped_sensors]

    if sensors:
        print(f"[MODE] {cfg.MODE} | sensors individuais={len(sensors)} | N_WORKERS={cfg.N_WORKERS}")
        print(f"[SENSORS] primeiros 20: {sensors[:20]}")

        if cfg.N_WORKERS <= 1:
            for s in sensors:
                try:
                    rows.append(run_one_sensor(cfg, df_alarm, df_feat, df_raw, s))
                except Exception as e:
                    print(f"[ERROR] sensor={s} -> {e}")
                    rows.append({"sensor": s, "skipped": True, "reason": f"exception: {e}",
                                 "output_dir": resolve_output_dir(cfg, s)})
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
                        rows.append({"sensor": s, "skipped": True, "reason": f"exception: {e}",
                                     "output_dir": resolve_output_dir(cfg, s)})

    if not rows:
        raise ValueError("Nenhum sensor ou grupo encontrado apos filtros do config.")

    df_summary = pd.DataFrame(rows).sort_values("skipped", ascending=True)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[DONE] Summary salvo em: {summary_path}")

    index_path = _write_models_index(cfg, summary_out_root, rows)
    if index_path:
        print(f"[INDEX] Índice de modelos: {index_path}")

    return {
        "summary_path": summary_path,
        "time_report_path": time_report_path,
        "sensor_outputs": rows,
        "sensors": sensors,
        "groups": [g["name"] for g in (cfg.SENSOR_GROUPS or [])],
    }
