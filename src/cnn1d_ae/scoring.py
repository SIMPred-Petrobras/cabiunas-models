from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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


def suppress_short_transient_episodes(
    df_point: pd.DataFrame,
    raw_series: pd.Series,
    state: Optional[pd.Series],
    min_sustained_minutes: float = 30.0,
    glitch_step_mult: float = 8.0,
    regime_shift_mult: float = 4.0,
    shutdown_lookahead_minutes: float = 360.0,
    lookback_minutes: float = 60.0,
    baseline_window_minutes: float = 120.0,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Suprime episódios de anomalia (`is_anom_point` contínuo) classificados
    como "transiente_curto": curtos, fracos, sem assinatura de glitch de
    sensor, mudança de regime operacional, nem parada em seguida.

    Origem: estudo de assinatura de episódios (`analysis/EPISODE_TRIAGE.md`) —
    esse bucket concentra 202 episódios "far" contra só 2 "near" da falha real
    (~1%), o único com risco desprezível de apagar sinal genuíno. Os demais
    buckets (glitch, mudança de regime, precursor de parada, sustentado sem
    causa) exigem tratamento próprio (filtro de qualidade de dado, threshold
    por regime, revisão manual) e NÃO são tocados aqui.

    Retorna o df_point com os episódios suprimidos (is_anom_point -> 0) e a
    lista dos episódios suprimidos (para auditoria/relatório).
    """
    anom = df_point["is_anom_point"].values.astype(bool)
    if not anom.any():
        return df_point, []

    raw = pd.to_numeric(raw_series, errors="coerce").sort_index()
    if state is not None:
        on_mask = state.reindex(raw.index).fillna("on").eq("on")
        raw_on = raw[on_mask]
    else:
        raw_on = raw
    diffs_nz = raw_on.diff().abs()
    diffs_nz = diffs_nz[diffs_nz > 0]
    typical_step = float(diffs_nz.median()) if len(diffs_nz) else np.nan
    step = typical_step if np.isfinite(typical_step) and typical_step > 0 else None

    idx = df_point.index
    grp = (anom != np.r_[False, anom[:-1]]).cumsum()
    keep_mask = np.ones(len(df_point), dtype=bool)
    suppressed: List[Dict] = []

    baseline_win = pd.Timedelta(minutes=baseline_window_minutes)
    lead = pd.Timedelta(minutes=lookback_minutes)
    lookahead = pd.Timedelta(minutes=shutdown_lookahead_minutes)

    for _, pos in pd.Series(grp).groupby(grp).groups.items():
        pos = list(pos)
        if not anom[pos[0]]:
            continue
        t0, t1 = idx[pos[0]], idx[pos[-1]]
        duration_min = (t1 - t0).total_seconds() / 60.0
        if duration_min >= min_sustained_minutes:
            continue  # só episódios curtos são candidatos a "transiente_curto"

        before = raw[(raw.index >= t0 - baseline_win) & (raw.index < t0)]
        after = raw[(raw.index > t1) & (raw.index <= t1 + baseline_win)]
        lead_in = raw[(raw.index >= t0 - lead) & (raw.index <= t1)]
        base_before = float(before.median()) if len(before) else np.nan
        base_after = float(after.median()) if len(after) else np.nan
        level_shift = (abs(base_after - base_before)
                       if np.isfinite(base_before) and np.isfinite(base_after) else np.nan)
        max_step_in_ep = float(lead_in.diff().abs().max()) if len(lead_in) > 1 else 0.0

        frac_off_lead = 0.0
        if state is not None and len(lead_in):
            st_lead = state.reindex(lead_in.index).fillna("on")
            frac_off_lead = float(st_lead.isin(["off_curto", "off_longo", "transiente"]).mean())

        shutdown_after = False
        if state is not None:
            st_after = state[(state.index > t1) & (state.index <= t1 + lookahead)]
            shutdown_after = bool(st_after.isin(["off_curto", "off_longo"]).any())

        is_glitch = step is not None and max_step_in_ep >= glitch_step_mult * step and frac_off_lead >= 0.15
        is_regime = (not is_glitch and step is not None and np.isfinite(level_shift)
                     and level_shift >= regime_shift_mult * step)

        if is_glitch or is_regime or shutdown_after:
            continue  # tratamento próprio (glitch/regime) ou candidato a precursor — não suprime

        keep_mask[pos] = False
        suppressed.append({
            "start": str(t0), "end": str(t1),
            "duration_min": round(duration_min, 1), "n_points": len(pos),
        })

    if suppressed:
        df_point = df_point.copy()
        df_point.loc[~keep_mask, "is_anom_point"] = 0
    return df_point, suppressed
