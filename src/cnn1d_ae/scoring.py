from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from tensorflow import keras


def reconstruction_mae_per_seq(model: "keras.Model", x: np.ndarray, batch_size: int) -> np.ndarray:
    """MAE por sequência, calculado em batches para nunca materializar a predição
    inteira na GPU (evita OOM em séries longas)."""
    n = len(x)
    bs = max(1, int(batch_size))
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, bs):
        xb = x[start:start + bs]
        pb = np.asarray(model.predict_on_batch(xb))
        out[start:start + len(xb)] = np.mean(np.abs(pb - xb), axis=(1, 2))
    return out


def reconstruction_mae_and_per_sensor(
    model: "keras.Model", x: np.ndarray, batch_size: int
):
    """Passada única em batches: retorna (mae_seq (n,), mae_per_sensor (n, n_sensors)).
    Substitui o par predict()+_per_sensor_mae para não segurar a predição completa
    na GPU nem rodar o predict duas vezes."""
    n = len(x)
    n_sensors = x.shape[2]
    bs = max(1, int(batch_size))
    mae_seq = np.empty(n, dtype=np.float32)
    mae_sensor = np.empty((n, n_sensors), dtype=np.float32)
    for start in range(0, n, bs):
        xb = x[start:start + bs]
        pb = np.asarray(model.predict_on_batch(xb))
        adiff = np.abs(pb - xb)
        end = start + len(xb)
        mae_seq[start:end] = np.mean(adiff, axis=(1, 2))
        mae_sensor[start:end] = np.mean(adiff, axis=1)
    return mae_seq, mae_sensor


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
    if mode == "alarm_f2":
        # Handled externally via compute_threshold_alarm_optimized; fallback to p99.
        return float(np.percentile(train_mae_seq, 99))
    raise ValueError("THRESH_MODE invalido. Use max_train/p95/p97/p99/p99_5/target_rate/alarm_f2.")


def _deduplicate_alarm_incidents(alarm_ts: pd.Series, incident_gap_hours: float = 4.0) -> pd.Series:
    """Agrupa alarmes dentro de incident_gap_hours no mesmo incidente; retorna 1 representante por incidente."""
    if alarm_ts.empty:
        return alarm_ts
    sorted_ts = alarm_ts.sort_values().reset_index(drop=True)
    gap = pd.Timedelta(hours=incident_gap_hours)
    keep = [sorted_ts.iloc[0]]
    for t in sorted_ts.iloc[1:]:
        if t - keep[-1] > gap:
            keep.append(t)
    return pd.Series(keep)


def compute_threshold_alarm_optimized(
    mae_seq: np.ndarray,
    index: pd.DatetimeIndex,
    alarm_times: pd.Series,
    time_steps: int,
    eval_window_minutes: int,
    target_recall: float = 0.30,
    max_fp_per_day: float = 15.0,
    fallback_percentile: float = 99.0,
    incident_gap_hours: float = 4.0,
    stride: int = 1,
) -> float:
    """
    Calibração semi-supervisionada: encontra o threshold que atinge target_recall
    nos alarmes históricos dentro de eval_window_minutes, mantendo FP/dia <= max_fp_per_day.

    Usa deduplicação por incidente: alarmes a menos de incident_gap_hours entre si
    são tratados como um único evento para o cálculo de recall. Isso evita que
    muitos alarmes do mesmo incidente distorçam a calibração.

    Estratégia: testa percentis de 99.9 → 70 (do mais estrito ao mais permissivo).
    Retorna o primeiro que satisfaz ambas as condições. Se nenhum satisfizer,
    retorna o que melhor equilibra recall e FP.
    """
    alarm_ts = pd.to_datetime(alarm_times, errors="coerce")
    alarm_ts = pd.Series(alarm_ts).dropna().drop_duplicates().sort_values()
    if len(alarm_ts) and len(index):
        alarm_ts = alarm_ts[(alarm_ts >= index.min()) & (alarm_ts <= index.max())]

    if len(alarm_ts) == 0:
        print("[ALARM-F2] Sem alarmes no range — usando fallback p99.")
        return float(np.percentile(mae_seq, fallback_percentile))

    # Deduplica alarmes individuais em incidentes únicos
    incidents = _deduplicate_alarm_incidents(alarm_ts, incident_gap_hours)
    n_incidents = len(incidents)
    n_alarms_raw = len(alarm_ts)
    if n_alarms_raw != n_incidents:
        print(f"[ALARM-F2] Deduplicação: {n_alarms_raw} alarmes → {n_incidents} incidentes (gap>{incident_gap_hours}h)")

    # Timestamps das sequências (posição final de cada janela)
    n_seq = len(mae_seq)
    end_offset = time_steps - 1
    seq_end_idx = end_offset + np.arange(n_seq) * max(1, int(stride))
    valid = seq_end_idx < len(index)
    seq_end_times = index[seq_end_idx[valid]]
    mae_valid = mae_seq[valid]

    seq_sec = seq_end_times.values.astype("int64") / 1e9
    # Recall medido sobre incidentes (1 representante por grupo)
    incident_sec = incidents.values.astype("int64") / 1e9
    win_sec = float(eval_window_minutes * 60)
    duration_days = float((index.max() - index.min()).total_seconds() / 86400)

    # Para FP: usamos todos os alarmes individuais como "janelas protegidas"
    alarm_sec_all = alarm_ts.values.astype("int64") / 1e9

    # Avalia cada candidato: recall (por incidente) + FP/dia
    candidates_pct = np.linspace(99.9, 70.0, 150)
    best_thresh = float(np.percentile(mae_valid, fallback_percentile))
    best_score = -1.0

    for pct in candidates_pct:
        thresh = float(np.percentile(mae_valid, pct))
        anom_mask = mae_valid > thresh
        if not anom_mask.any():
            continue

        anom_sec = seq_sec[anom_mask]

        # Recall por incidente
        diff_inc = np.abs(incident_sec[:, None] - anom_sec[None, :])
        hits = int((diff_inc <= win_sec).any(axis=1).sum())
        recall = hits / n_incidents

        # FP/dia: sequências anômalas fora de qualquer janela de alarme individual
        diff_all = np.abs(alarm_sec_all[:, None] - anom_sec[None, :])
        in_window = (diff_all <= win_sec).any(axis=0)
        fp_pts = int((~in_window).sum())
        fp_rate = fp_pts / max(duration_days, 1.0)

        fp_penalty = max(0.0, (fp_rate - max_fp_per_day) / max_fp_per_day)
        score = recall - 0.5 * fp_penalty

        if recall >= target_recall and fp_rate <= max_fp_per_day:
            print(
                f"[ALARM-F2] threshold={thresh:.5f} (p{pct:.1f}): "
                f"recall={recall:.1%} ({hits}/{n_incidents} incidentes)  FP/dia={fp_rate:.1f}"
            )
            return thresh

        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(
        f"[ALARM-F2] Nao atingiu target (recall>={target_recall:.0%} incidentes, FP<={max_fp_per_day}/dia). "
        f"Usando melhor compromisso: thresh={best_thresh:.5f}"
    )
    return best_thresh


def apply_adaptive_monthly_threshold(
    mae_seq: np.ndarray,
    index: pd.DatetimeIndex,
    time_steps: int,
    alarm_times: pd.Series,
    eval_window_minutes: int,
    percentile: float = 99.0,
    stride: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Recalibra o threshold mensalmente usando o percentil do MAE em períodos sem alarme.

    Resolve o drift sazonal: em vez de um limiar fixo calibrado no treino,
    cada mês tem seu próprio limiar calculado sobre os dados normais daquele mês.
    Não requer re-treinamento — só recalibra o limiar de decisão.

    Returns:
        (anomaly_seq recalibrado, dict {month_str -> threshold})
    """
    n_seq = len(mae_seq)
    end_offset = time_steps - 1
    seq_end_idx = end_offset + np.arange(n_seq) * max(1, int(stride))
    valid_mask = seq_end_idx < len(index)
    seq_end_times = index[seq_end_idx[valid_mask]]
    mae_valid = mae_seq[valid_mask]

    alarm_win = build_alarm_window_mask(seq_end_times, alarm_times, eval_window_minutes)
    months = pd.DatetimeIndex(seq_end_times).to_period("M")

    global_non_alarm = mae_valid[~alarm_win.values]
    global_thresh = float(np.percentile(global_non_alarm, percentile)) if len(global_non_alarm) >= 10 else float(np.percentile(mae_valid, percentile))

    monthly_thresholds: dict = {}
    thresh_array = np.full(n_seq, global_thresh, dtype=float)

    for month in months.unique():
        m_mask = np.asarray(months == month)
        non_alarm_m = m_mask & (~alarm_win.values)
        if non_alarm_m.sum() >= 30:
            t = float(np.percentile(mae_valid[non_alarm_m], percentile))
        else:
            t = global_thresh
        monthly_thresholds[str(month)] = t
        # aplica ao array de thresholds (mesmo nos inválidos — serão ignorados)
        full_idx = seq_end_idx[valid_mask][m_mask]
        valid_full = full_idx < n_seq
        thresh_array[full_idx[valid_full]] = t

    new_anomaly_seq = mae_seq > thresh_array
    return new_anomaly_seq, monthly_thresholds


def apply_cusum_alarm(
    mae_seq: np.ndarray,
    train_mae_seq: np.ndarray,
    k: float = 0.5,
    h: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Política de alarme CUSUM sobre a série de erro de reconstrução.

    Detector de deriva sustentada (imune a spikes isolados): acumula o desvio
    normalizado do erro e dispara quando o acumulador supera `h`.
        z[t]     = (erro[t] - mu_train) / sigma_train
        cusum[t] = max(0, cusum[t-1] + z[t] - k)
        alarme   = cusum[t] >= h
    mu/sigma são calibrados nos erros de TREINO (sem vazamento). Para os erros de
    reconstrução do AE, FP costumam ser picos de poucos minutos enquanto anomalias
    reais mantêm o erro elevado por horas — o CUSUM exige essa persistência.

    Retorna (anomaly_seq booleano, série cusum) para diagnóstico/plot.
    """
    mu = float(np.mean(train_mae_seq))
    sigma = float(np.std(train_mae_seq))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-9
    z = (np.asarray(mae_seq, dtype=float) - mu) / sigma
    cusum = np.empty(len(z), dtype=float)
    acc = 0.0
    for i, zi in enumerate(z):
        acc = max(0.0, acc + zi - k)
        cusum[i] = acc
    return cusum >= h, cusum


def apply_debounce(anomaly_seq: np.ndarray, n_points: int) -> np.ndarray:
    """Exige `n_points` consecutivos anômalos antes de confirmar o alarme (on-delay
    ISA-18.2). O alarme é atribuído ao último ponto da janela. n<=1 = no-op."""
    n = int(n_points)
    if n <= 1:
        return np.asarray(anomaly_seq, dtype=bool)
    s = pd.Series(np.asarray(anomaly_seq, dtype=int))
    rolling = s.rolling(window=n, min_periods=n).sum()
    return (rolling >= n).fillna(False).to_numpy()


def map_seq_to_point_anomalies(
    anomaly_seq: np.ndarray,
    index: pd.DatetimeIndex,
    time_steps: int,
    point_rule: str,
    point_window: int,
    point_min_count: int,
    stride: int = 1,
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

    end_pos = (time_steps - 1) + np.arange(len(point_flags)) * max(1, int(stride))
    valid = end_pos < len(index)
    valid_positions = end_pos[valid]
    valid_flags = point_flags.values[valid]

    if len(valid_positions):
        anom_positions = valid_positions[valid_flags.astype(bool)]
        if len(anom_positions):
            anom_times = index[anom_positions]
            df_point.loc[anom_times, "is_anom_point"] = 1

    return df_point


def build_sequence_scores_df(index: pd.DatetimeIndex, mae_seq: np.ndarray, anomaly_seq: np.ndarray, stride: int = 1) -> pd.DataFrame:
    start_pos = np.arange(len(mae_seq)) * max(1, int(stride))
    valid = start_pos < len(index)
    return pd.DataFrame(
        {
            "seq_start_time": index[start_pos[valid]],
            "mae_seq": mae_seq[valid],
            "is_anom_seq": anomaly_seq.astype(int)[valid],
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


def build_operational_state_from_running(
    index: pd.DatetimeIndex,
    running: pd.Series,
    buffer_minutes: float = 5.0,
    asymmetric: bool = True,
) -> pd.Series:
    """Estado operacional a partir do flag de operacao (running_a, proxy de NGP_A).

    'on' quando ligado; 'off' quando desligado; 'transiente' nas bordas das transicoes.

    asymmetric=True (recomendado): só mascara o RAMP-UP pos-partida (off->on),
    preservando a janela PRE-desligamento (on->off) — onde pode estar a assinatura
    de uma falha que trippa a maquina. Recupera recall sem reintroduzir FP de partida.

    asymmetric=False: mascara +/- buffer_minutes em TODAS as transicoes (legado).
    """
    run = pd.to_numeric(pd.Series(running).reindex(index), errors="coerce").fillna(0.0)
    state = pd.Series("on", index=index, dtype=object)
    is_off = run <= 0.5
    state.loc[is_off] = "off"
    if bool(is_off.any()):
        pad = pd.Timedelta(minutes=max(0.0, float(buffer_minutes)))
        prev_off = is_off.shift(fill_value=bool(is_off.iloc[0]))
        startup_edge = (~is_off) & prev_off    # off->on (partida)
        if asymmetric:
            # so mascara o ramp-up depois da partida; pre-desligamento fica intacto.
            for t in index[startup_edge.values]:
                w = (index >= t) & (index <= t + pad)
                state.loc[w & (state == "on")] = "transiente"
        else:
            edge = is_off != prev_off
            for t in index[edge.values]:
                w = (index >= (t - pad)) & (index <= (t + pad))
                state.loc[w & (state == "on")] = "transiente"
    return state


def mask_anomaly_seq_by_operational_state(
    anomaly_seq: np.ndarray,
    index: pd.DatetimeIndex,
    time_steps: int,
    state: pd.Series,
    stride: int = 1,
) -> np.ndarray:
    seq_end_pos = (time_steps - 1) + np.arange(len(anomaly_seq)) * max(1, int(stride))
    valid = seq_end_pos < len(index)
    out = anomaly_seq.astype(bool).copy()
    if not valid.any():
        return out
    seq_end_idx = index[seq_end_pos[valid]]
    allowed = state.reindex(seq_end_idx).fillna("on").eq("on").values
    out_valid = out[valid] & allowed
    out[valid] = out_valid
    return out
