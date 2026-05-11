from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from tensorflow import keras


def reconstruction_mae_per_seq(model: "keras.Model", x: np.ndarray, batch_size: int) -> np.ndarray:
    x_pred = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean(np.abs(x_pred - x), axis=(1, 2))


def compute_threshold(train_mae_seq: np.ndarray, mode: str, target_rate: float = 0.01) -> float:
    mode = mode.lower()
    if mode == "max_train":
        return float(np.max(train_mae_seq))
    if mode == "p95":
        return float(np.percentile(train_mae_seq, 95))
    if mode == "p97":
        return float(np.percentile(train_mae_seq, 97))
    if mode == "p99":
        return float(np.percentile(train_mae_seq, 99))
    if mode == "p99_5":
        return float(np.percentile(train_mae_seq, 99.5))
    if mode == "target_rate":
        rate = float(np.clip(target_rate, 1e-6, 0.5))
        return float(np.quantile(train_mae_seq, 1.0 - rate))
    raise ValueError("THRESH_MODE invalido. Use max_train/p95/p97/p99/p99_5/target_rate.")


def map_seq_to_point_anomalies(
    anomaly_seq: np.ndarray,
    index: pd.DatetimeIndex,
    time_steps: int,
    point_rule: str,
    point_window: int,
    point_min_count: int,
) -> pd.DataFrame:
    seq_series = pd.Series(anomaly_seq.astype(int))
    w = max(1, int(point_window))
    k = max(1, int(point_min_count))

    if point_rule.lower() == "all_of_window":
        k = w
    elif point_rule.lower() != "k_of_window":
        raise ValueError("POINT_RULE invalido. Use 'all_of_window' ou 'k_of_window'.")

    votes = seq_series.rolling(window=w, min_periods=w).sum().fillna(0)
    point_flags = (votes >= k).astype(int)

    df_point = pd.DataFrame(index=index)
    df_point["is_anom_point"] = 0

    end_pos = np.arange(time_steps - 1, time_steps - 1 + len(point_flags))
    valid = end_pos < len(index)
    valid_positions = end_pos[valid]
    valid_flags = point_flags.values[valid]

    if len(valid_positions):
        anom_positions = valid_positions[valid_flags.astype(bool)]
        if len(anom_positions):
            anom_times = index[anom_positions]
            df_point.loc[anom_times, "is_anom_point"] = 1

    return df_point


def build_sequence_scores_df(index: pd.DatetimeIndex, mae_seq: np.ndarray, anomaly_seq: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seq_start_time": index[: len(mae_seq)],
            "mae_seq": mae_seq,
            "is_anom_seq": anomaly_seq.astype(int),
        }
    )


def compute_anomaly_rate_per_day(df_point: pd.DataFrame) -> float:
    if df_point.empty:
        return 0.0
    n_days = max(1e-9, (df_point.index.max() - df_point.index.min()).total_seconds() / 86400.0)
    n_anom = float(df_point["is_anom_point"].sum())
    return float(n_anom / n_days)


def build_alarm_window_mask(index: pd.DatetimeIndex, alarm_times: pd.Series, minutes: int) -> pd.Series:
    mask = pd.Series(False, index=index)
    win = pd.Timedelta(minutes=max(0, int(minutes)))
    alarm_idx = pd.to_datetime(alarm_times, errors="coerce")
    for t in pd.Series(alarm_idx).dropna().drop_duplicates().sort_values():
        mask.loc[(mask.index >= t - win) & (mask.index <= t + win)] = True
    return mask


def _estimate_index_duration_days(index: pd.DatetimeIndex) -> float:
    if len(index) == 0:
        return 0.0
    if len(index) == 1:
        return 1.0 / 86400.0

    deltas = index.to_series().diff().dt.total_seconds().dropna()
    median_dt = float(deltas[deltas > 0].median()) if (deltas > 0).any() else 1.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0
    return float((len(index) * median_dt) / 86400.0)


def _count_anomaly_episodes(df_anom: pd.DataFrame, base_index: pd.DatetimeIndex) -> tuple[int, int]:
    if df_anom.empty:
        return 0, 0

    idx = pd.DatetimeIndex(df_anom.index).sort_values()
    if len(idx) <= 1:
        return 1, int(bool(df_anom["inside_alarm_window"].any()))

    deltas = base_index.to_series().diff().dt.total_seconds().dropna()
    median_dt = float(deltas[deltas > 0].median()) if (deltas > 0).any() else 1.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0

    # Quebras maiores que 3x a granularidade mediana separam episodios independentes.
    episode_id = idx.to_series().diff().dt.total_seconds().fillna(0).ge(3.0 * median_dt).cumsum()
    episode_df = df_anom.copy()
    episode_df["_episode_id"] = episode_id.values
    total = int(episode_df["_episode_id"].nunique())
    matched = int(episode_df.groupby("_episode_id")["inside_alarm_window"].any().sum())
    return total, matched


def evaluate_alarm_detection(
    df_alarm: pd.DataFrame,
    df_point: pd.DataFrame,
    window_minutes: int,
) -> dict[str, object]:
    alarm_times = (
        pd.to_datetime(df_alarm["Data da Ocorrencia"], errors="coerce")
        if "Data da Ocorrencia" in df_alarm.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    alarm_times = pd.Series(alarm_times).dropna().drop_duplicates().sort_values()

    # Filtra alarmes ao range do sensor: alarmes fora do período observado não podem
    # ser detectados e não devem entrar no denominador da hit_rate.
    if not df_point.empty and len(alarm_times):
        t_min = pd.to_datetime(df_point.index.min(), errors="coerce")
        t_max = pd.to_datetime(df_point.index.max(), errors="coerce")
        if pd.notna(t_min) and pd.notna(t_max):
            alarm_times = alarm_times[(alarm_times >= t_min) & (alarm_times <= t_max)]

    n_alarms = int(len(alarm_times))

    if df_point.empty or "is_anom_point" not in df_point.columns:
        return {
            "n_alarms": n_alarms,
            "alarms_with_detected_anomaly_in_window": 0,
            "hit_rate": 0.0 if n_alarms else None,
            "eval_window_minutes": int(window_minutes),
            "precision_event": None,
            "f1_event": None,
            "anomaly_rate_points_per_day_no_alarm_periods": 0.0,
            "median_lead_time_minutes_before_alarm": None,
            "n_predicted_anomaly_episodes": 0,
            "n_false_positive_anomaly_episodes": 0,
        }

    df_eval = df_point.copy()
    df_eval.index = pd.to_datetime(df_eval.index, errors="coerce")
    df_eval = df_eval[~df_eval.index.isna()].sort_index()
    df_eval["is_anom_point"] = pd.to_numeric(df_eval["is_anom_point"], errors="coerce").fillna(0).astype(int)
    df_eval["inside_alarm_window"] = build_alarm_window_mask(df_eval.index, alarm_times, window_minutes)

    hits = 0
    lead_minutes: list[float] = []
    win = pd.Timedelta(minutes=max(0, int(window_minutes)))
    anom_idx = df_eval.index[df_eval["is_anom_point"].eq(1)]
    for t in alarm_times:
        in_window = anom_idx[(anom_idx >= t - win) & (anom_idx <= t + win)]
        if len(in_window) > 0:
            hits += 1
        before_alarm = anom_idx[(anom_idx >= t - win) & (anom_idx <= t)]
        if len(before_alarm) > 0:
            first_detection = before_alarm.min()
            lead_minutes.append(float((t - first_detection).total_seconds() / 60.0))

    recall_event = hits / n_alarms if n_alarms else np.nan

    df_anom = df_eval.loc[df_eval["is_anom_point"].eq(1), ["inside_alarm_window"]]
    n_episodes, matched_episodes = _count_anomaly_episodes(df_anom, pd.DatetimeIndex(df_eval.index))
    false_positive_episodes = max(0, n_episodes - matched_episodes)
    precision_event = matched_episodes / n_episodes if n_episodes else np.nan
    f1_event = (
        2.0 * precision_event * recall_event / (precision_event + recall_event)
        if np.isfinite(precision_event) and np.isfinite(recall_event) and (precision_event + recall_event) > 0
        else np.nan
    )

    no_alarm = ~df_eval["inside_alarm_window"]
    no_alarm_days = _estimate_index_duration_days(pd.DatetimeIndex(df_eval.index[no_alarm]))
    false_positive_points = int(df_eval.loc[no_alarm, "is_anom_point"].sum())
    fp_rate_no_alarm = false_positive_points / max(no_alarm_days, 1e-9)

    return {
        "n_alarms": n_alarms,
        "alarms_with_detected_anomaly_in_window": int(hits),
        "hit_rate": float(recall_event) if np.isfinite(recall_event) else None,
        "eval_window_minutes": int(window_minutes),
        "precision_event": float(precision_event) if np.isfinite(precision_event) else None,
        "f1_event": float(f1_event) if np.isfinite(f1_event) else None,
        "anomaly_rate_points_per_day_no_alarm_periods": float(fp_rate_no_alarm),
        "median_lead_time_minutes_before_alarm": (
            float(np.median(lead_minutes)) if lead_minutes else None
        ),
        "n_predicted_anomaly_episodes": int(n_episodes),
        "n_matched_anomaly_episodes": int(matched_episodes),
        "n_false_positive_anomaly_episodes": int(false_positive_episodes),
        "false_positive_anomaly_points_no_alarm_periods": int(false_positive_points),
    }


def compute_monthly_mae_drift(df_seq_scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df_seq_scores.empty:
        return pd.DataFrame()

    df = df_seq_scores.copy()
    df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], errors="coerce")
    df["mae_seq"] = pd.to_numeric(df["mae_seq"], errors="coerce")
    df["is_anom_seq"] = pd.to_numeric(df["is_anom_seq"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["seq_start_time", "mae_seq"])
    if df.empty:
        return pd.DataFrame()

    df["month"] = df["seq_start_time"].dt.to_period("M").astype(str)
    rows = []
    for month, g in df.groupby("month", sort=True):
        p95 = float(g["mae_seq"].quantile(0.95))
        p99 = float(g["mae_seq"].quantile(0.99))
        rows.append(
            {
                "month": month,
                "n_sequences": int(len(g)),
                "mae_mean": float(g["mae_seq"].mean()),
                "mae_std": float(g["mae_seq"].std(ddof=0)),
                "mae_p50": float(g["mae_seq"].quantile(0.50)),
                "mae_p95": p95,
                "mae_p99": p99,
                "mae_max": float(g["mae_seq"].max()),
                "seq_anomaly_rate": float(g["is_anom_seq"].mean()),
                "p95_threshold_ratio": float(p95 / threshold) if threshold else np.nan,
                "p99_threshold_ratio": float(p99 / threshold) if threshold else np.nan,
                "drift_flag_p99_above_threshold": bool(p99 > threshold),
            }
        )
    return pd.DataFrame(rows)


def build_operational_state(
    index: pd.DatetimeIndex,
    sensor_series: pd.Series,
    off_value_quantile: float = 0.05,
    off_abs_threshold: float | None = None,
    off_long_min_hours: float = 24.0,
    transient_padding_minutes: int = 20,
    transient_diff_quantile: float = 0.99,
) -> pd.Series:
    s = pd.to_numeric(sensor_series.reindex(index), errors="coerce").ffill().bfill()
    state = pd.Series("on", index=index, dtype=object)

    if off_abs_threshold is None:
        off_thr = float(s.quantile(float(np.clip(off_value_quantile, 0.0, 0.5))))
    else:
        off_thr = float(off_abs_threshold)

    is_off = s <= off_thr
    if is_off.any():
        grp = is_off.ne(is_off.shift(fill_value=False)).cumsum()
        run_id = grp.where(is_off)
        run_len = is_off.groupby(grp).transform("sum").where(is_off, 0).fillna(0)

        dt_seconds = index.to_series().diff().dt.total_seconds().median()
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            dt_seconds = 1.0
        min_points_long = int(max(1, np.ceil((off_long_min_hours * 3600.0) / dt_seconds)))

        is_off_long = (run_len >= min_points_long) & is_off
        is_off_short = (run_len < min_points_long) & is_off
        state.loc[is_off_short] = "off_curto"
        state.loc[is_off_long] = "off_longo"

        # Marca transiente nas bordas das corridas off (liga/desliga).
        edge = is_off != is_off.shift(fill_value=is_off.iloc[0])
        edge_times = index[edge]
        if len(edge_times):
            pad = pd.Timedelta(minutes=max(0, int(transient_padding_minutes)))
            for t in edge_times:
                w = (index >= (t - pad)) & (index <= (t + pad))
                state.loc[w & (state == "on")] = "transiente"

    diff_abs = s.diff().abs().fillna(0.0)
    dq = float(np.clip(transient_diff_quantile, 0.5, 0.9999))
    diff_thr = float(diff_abs.quantile(dq))
    if np.isfinite(diff_thr) and diff_thr > 0:
        high_diff = diff_abs >= diff_thr
        state.loc[high_diff & (state == "on")] = "transiente"

    return state


def mask_anomaly_seq_by_operational_state(
    anomaly_seq: np.ndarray,
    index: pd.DatetimeIndex,
    time_steps: int,
    state: pd.Series,
) -> np.ndarray:
    seq_end_pos = np.arange(time_steps - 1, time_steps - 1 + len(anomaly_seq))
    valid = seq_end_pos < len(index)
    out = anomaly_seq.astype(bool).copy()
    if not valid.any():
        return out
    seq_end_idx = index[seq_end_pos[valid]]
    allowed = state.reindex(seq_end_idx).fillna("on").eq("on").values
    out_valid = out[valid] & allowed
    out[valid] = out_valid
    return out
