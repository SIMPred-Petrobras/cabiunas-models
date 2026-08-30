from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from tensorflow import keras


def reconstruction_mae_per_seq(model: "keras.Model", x: np.ndarray, batch_size: int) -> np.ndarray:
    x_pred = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean(np.abs(x_pred - x), axis=(1, 2))


def compute_threshold(
    train_mae_seq: np.ndarray, mode: str, target_rate: float = 0.01, std_k: float = 3.0
) -> float:
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
    if mode == "mean_std":
        return float(np.mean(train_mae_seq) + float(std_k) * np.std(train_mae_seq))
    if mode == "robust_mad":
        # Mediana + k*1.4826*MAD -- 1.4826 e a constante que torna o MAD
        # comparavel ao desvio padrao sob normalidade. Mean/std (mean_std)
        # sao sensiveis a cauda longa/outliers na distribuicao de erro de
        # treino -- num grupo com muitas features derivadas (multiescala+
        # textura), poucos pontos de treino com erro alto ja inflam
        # mean+k*std a um patamar que a serie inteira nunca cruza (EXP13:
        # 0 deteccoes em toda a serie com mean_std/std_k=4). Mediana/MAD
        # ignoram esses outliers, dando um limiar mais fiel ao "typical"
        # comportamento de erro do treino.
        median = np.median(train_mae_seq)
        mad = np.median(np.abs(train_mae_seq - median))
        return float(median + float(std_k) * 1.4826 * mad)
    raise ValueError("THRESH_MODE invalido. Use max_train/p95/p97/p99/p99_5/target_rate/mean_std/robust_mad.")


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

    # Sequencia i comeca na posicao original i*stride (make_sequences),
    # entao termina em i*stride + time_steps - 1 -- para stride=1 isso
    # colapsa na conta antiga (time_steps - 1 + i). Sem essa correcao,
    # STRIDE>1 desalinha silenciosamente os timestamps de deteccao.
    end_pos = np.arange(len(point_flags)) * int(stride) + (time_steps - 1)
    valid = end_pos < len(index)
    valid_positions = end_pos[valid]
    valid_flags = point_flags.values[valid]

    if len(valid_positions):
        anom_positions = valid_positions[valid_flags.astype(bool)]
        if len(anom_positions):
            anom_times = index[anom_positions]
            df_point.loc[anom_times, "is_anom_point"] = 1

    return df_point


def build_sequence_scores_df(
    index: pd.DatetimeIndex, mae_seq: np.ndarray, anomaly_seq: np.ndarray, stride: int = 1
) -> pd.DataFrame:
    # Sequencia i comeca na posicao original i*stride -- para stride=1
    # colapsa em index[:len(mae_seq)] (comportamento antigo).
    start_pos = np.arange(len(mae_seq)) * int(stride)
    valid = start_pos < len(index)
    return pd.DataFrame(
        {
            "seq_start_time": index[start_pos[valid]],
            "mae_seq": mae_seq[valid],
            "is_anom_seq": anomaly_seq[valid].astype(int),
        }
    )


def compute_anomaly_rate_per_day(df_point: pd.DataFrame) -> float:
    if df_point.empty:
        return 0.0
    n_days = max(1e-9, (df_point.index.max() - df_point.index.min()).total_seconds() / 86400.0)
    n_anom = float(df_point["is_anom_point"].sum())
    return float(n_anom / n_days)


def eval_alarm_hit_rate(df_alarm: pd.DataFrame, df_point: pd.DataFrame, minutes: int) -> dict:
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


def compute_normal_alert_rate(df_point: pd.DataFrame, near_alarm_mask: pd.Series) -> float:
    """Fracao de pontos operacionais ('on'), fora de qualquer janela de alarme, marcados como anomalia.

    Equivalente ao 'normal_alert_rate' da pipeline de AutoML, mas calculado
    sobre a nossa propria mascara de alarmes/estado operacional.
    """
    normal_mask = ~near_alarm_mask.reindex(df_point.index).fillna(False)
    if "operational_state" in df_point.columns:
        normal_mask &= df_point["operational_state"] == "on"
    normal = df_point.loc[normal_mask]
    if normal.empty:
        return 0.0
    return float(normal["is_anom_point"].mean())


def compute_composite_score(
    detection_rate: float,
    normal_alert_rate: float,
    fp_penalty: float = 2.0,
    min_detection_rate: float = 0.0,
) -> dict:
    """Score que balanceia deteccao de alarme x falso positivo (mesma forma da
    pipeline de AutoML da Lara, mas alimentado com detection_rate/normal_alert_rate
    calculados na nossa propria regua de avaliacao — ver analise_automl_lara.md).
    """
    detection_rate = float(detection_rate)
    normal_alert_rate = float(normal_alert_rate)
    balanced_score = detection_rate - float(fp_penalty) * (normal_alert_rate ** 2)
    if detection_rate < min_detection_rate:
        balanced_score -= (min_detection_rate - detection_rate) * 2.0
    composite_score = float(np.clip((balanced_score + fp_penalty) / (1.0 + fp_penalty), 0.0, 1.0))
    return {
        "composite_score": composite_score,
        "balanced_score": float(balanced_score),
        "detection_rate": detection_rate,
        "normal_alert_rate": normal_alert_rate,
    }


def build_operational_state(
    index: pd.DatetimeIndex,
    sensor_series: pd.Series,
    off_value_quantile: float = 0.05,
    off_abs_threshold: float | None = None,
    off_long_min_hours: float = 24.0,
    transient_padding_minutes: int = 20,
    transient_diff_quantile: float = 0.99,
    secondary_series: pd.Series | None = None,
    secondary_off_abs_threshold: float | None = None,
) -> pd.Series:
    """`sensor_series` (tipicamente OPERATIONAL_REF_SENSOR, ex: RUNNING_A) e a
    referencia primaria de liga/desliga. `secondary_series`/
    `secondary_off_abs_threshold` (opcional, ex: o proprio sensor-alvo do
    grupo) adicionam um segundo criterio de "off" independente -- OR'd com o
    primeiro antes da classificacao off_curto/off_longo/transiente. Motivado
    por um desligamento real (2025-08-19 a 2025-08-23) em que RUNNING_A ficou
    ~0.96-1.0 o periodo todo (nao caiu abaixo do OFF_ABS_THRESHOLD) enquanto
    TC382_03_A caiu para ~28-32C (nivel ambiente, fisicamente incompativel
    com operacao) -- a mascara baseada so em RUNNING_A nao detectava esse
    periodo como off, respondendo por 65% do normal_alert_rate do EXP7
    item1+2. Ver docs/analise_automl_exp9_planejamento.md."""
    s = pd.to_numeric(sensor_series.reindex(index), errors="coerce").ffill().bfill()
    state = pd.Series("on", index=index, dtype=object)

    if off_abs_threshold is None:
        off_thr = float(s.quantile(float(np.clip(off_value_quantile, 0.0, 0.5))))
    else:
        off_thr = float(off_abs_threshold)

    is_off = s <= off_thr
    if secondary_series is not None and secondary_off_abs_threshold is not None:
        s2 = pd.to_numeric(secondary_series.reindex(index), errors="coerce").ffill().bfill()
        is_off = is_off | (s2 <= float(secondary_off_abs_threshold))
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


def compute_load_ramp_gate(
    load_series: pd.Series,
    ramp_halflife_minutes: float = 120.0,
    window_minutes: float = 360.0,
) -> tuple[pd.Series, pd.Series]:
    """Rampa causal de um sensor-proxy de carga e seu maximo trailing.

    smoothed = EWMA(load_series, half-life=ramp_halflife_minutes)
    ramp     = |d(smoothed)/dt| em unidades/hora
    gate     = ramp.rolling(window_minutes, trailing).max()

    Retorna (gate, smoothed) alinhados ao index de load_series.
    """
    s = pd.to_numeric(load_series, errors="coerce").sort_index()
    dt_seconds = s.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    halflife_periods = max(1.0, (float(ramp_halflife_minutes) * 60.0) / dt_seconds)
    smoothed = s.ewm(halflife=halflife_periods).mean()

    dt_hours = s.index.to_series().diff().dt.total_seconds() / 3600.0
    ramp = (smoothed.diff() / dt_hours).abs()
    gate = ramp.rolling(f"{int(window_minutes)}min", min_periods=1).max()
    return gate, smoothed


def apply_load_gate(
    df_point: pd.DataFrame,
    load_series: pd.Series,
    ramp_max: float,
    level_min: float = 0.0,
    ramp_halflife_minutes: float = 120.0,
    window_minutes: float = 360.0,
) -> pd.DataFrame:
    """Suprime is_anom_point durante manobra de carga (rampa alta) do proxy informado.

    Causal: usa reindex(method='ffill') para nunca olhar o futuro. Regra:
    bloqueia quando ramp >= ramp_max, ou (se level_min > 0) quando o nivel
    suavizado do proxy fica abaixo de level_min.
    """
    gate, smoothed = compute_load_ramp_gate(load_series, ramp_halflife_minutes, window_minutes)
    gate_at_point = gate.reindex(df_point.index, method="ffill")
    level_at_point = smoothed.reindex(df_point.index, method="ffill")

    blocked = gate_at_point >= ramp_max
    if level_min > 0:
        blocked = blocked | (level_at_point < level_min)
    blocked = blocked.fillna(False)

    df_point = df_point.copy()
    df_point["load_gate_blocked"] = blocked.values
    df_point.loc[blocked.values, "is_anom_point"] = 0
    return df_point


def compute_volatility_index(df_sensors: pd.DataFrame, window_minutes: float) -> pd.Series:
    """Indice de volatilidade multivariado: desvio-padrao movel causal
    (janela trailing, nunca olha o futuro) de cada coluna de `df_sensors`,
    reduzido pela media entre colunas a cada instante. Pensado para um
    grupo de sensores fisicamente correlacionados (ex: os 10 canais de
    vibracao de mancal) onde uma manobra real (rampa de carga) eleva a
    variabilidade de varios canais ao mesmo tempo, mesmo sem uma rampa de
    *nivel* clara no sensor-alvo -- complementar ao portao de rampa
    (`apply_load_gate`, que reage a taxa de variacao do nivel, nao a
    variabilidade local). Ver docs/analise_automl_exp9_planejamento.md."""
    dt_seconds = df_sensors.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    window_samples = max(2, int(round((float(window_minutes) * 60.0) / dt_seconds)))
    roll_std = df_sensors.rolling(window_samples, min_periods=max(2, window_samples // 2)).std()
    return roll_std.mean(axis=1)


def apply_volatility_gate(df_point: pd.DataFrame, volatility_index: pd.Series, threshold: float) -> pd.DataFrame:
    """Suprime is_anom_point quando o indice de volatilidade (ver
    `compute_volatility_index`) ultrapassa `threshold` -- causal por
    construcao (a serie de entrada ja e uma janela trailing)."""
    idx_at_point = volatility_index.reindex(df_point.index, method="ffill")
    blocked = (idx_at_point > float(threshold)).fillna(False)

    df_point = df_point.copy()
    df_point["volatility_gate_blocked"] = blocked.values
    df_point.loc[blocked.values, "is_anom_point"] = 0
    return df_point


def compute_frozen_sensor_mask(df_sensors: pd.DataFrame, window_minutes: float) -> pd.Series:
    """True quando QUALQUER coluna de `df_sensors` fica com leitura
    literalmente constante (`diff()==0`) por uma janela sustentada de
    `window_minutes` -- indica falha de instrumento/comunicacao (sensor
    travado), nao sinal real. Causal por construcao (rolling trailing).
    Validado no EXP10c com W=5min -- ver docs/analise_automl_exp10.md,
    secao "Veto de sensor congelado"."""
    dt_seconds = df_sensors.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    w_samples = max(1, int(round((float(window_minutes) * 60.0) / dt_seconds)))
    diff_zero = df_sensors.diff() == 0
    frozen_any = pd.Series(False, index=df_sensors.index)
    for col in df_sensors.columns:
        sustained = diff_zero[col].rolling(w_samples, min_periods=w_samples).sum() >= w_samples
        frozen_any = frozen_any | sustained.fillna(False)
    return frozen_any


def apply_frozen_sensor_veto(df_point: pd.DataFrame, frozen_mask: pd.Series) -> pd.DataFrame:
    """Suprime is_anom_point quando `compute_frozen_sensor_mask` indica
    algum sensor do grupo travado."""
    mask_at_point = frozen_mask.reindex(df_point.index).fillna(False)
    df_point = df_point.copy()
    df_point["frozen_sensor_blocked"] = mask_at_point.values
    df_point.loc[mask_at_point.values, "is_anom_point"] = 0
    return df_point


def apply_min_duration_filter(df_point: pd.DataFrame, min_duration_minutes: float) -> pd.DataFrame:
    """Suprime episodios CONTINUOS de `is_anom_point` mais curtos que
    `min_duration_minutes` -- aplicado por ultimo no laco de portoes
    (depois de mascara/rampa/volatilidade/veto de congelamento), mede a
    duracao do que sobra apos todos os outros filtros. Precursores reais
    tendem a persistir muito mais tempo que ruido residual (mediana
    49,5min vs 2,5min no EXP10c congelado) -- ver
    docs/analise_automl_exp10.md, secao "Duracao do score: TP vs FP
    residual". So valido contra um modelo UNICO/congelado -- 3 tentativas
    mostraram que nao sobrevive a ENABLE_WALKFORWARD_RETRAIN (ver secoes
    seguintes do mesmo doc); `run_automl_group` impede a combinacao."""
    flags = df_point["is_anom_point"].values.astype(bool)
    dt_seconds = df_point.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    min_samples = max(1, int(round((float(min_duration_minutes) * 60.0) / dt_seconds)))

    m = flags.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]

    filtered = flags.copy()
    for s, e in zip(starts, ends):
        if (e - s) < min_samples:
            filtered[s:e] = False

    df_point = df_point.copy()
    df_point["duration_filter_blocked"] = flags & ~filtered
    df_point["is_anom_point"] = filtered.astype(int)
    return df_point


def mask_anomaly_seq_by_operational_state(
    anomaly_seq: np.ndarray,
    index: pd.DatetimeIndex,
    time_steps: int,
    state: pd.Series,
    stride: int = 1,
) -> np.ndarray:
    # Mesma correcao de map_seq_to_point_anomalies: sequencia i comeca na
    # posicao original i*stride, termina em i*stride + time_steps - 1. Sem
    # isso (stride>1), o lookup de estado operacional cai em timestamps
    # errados -- no EXP13 (stride=15), isso mascarava 100% das deteccoes
    # (2603 sequencias cruzavam o threshold, 0 sobravam pos-mascara) porque
    # o calculo antigo so cobria os primeiros ~44 dias da serie (a mesma
    # contagem de indices, sem multiplicar pelo stride), region que calha
    # de ser majoritariamente off/transiente.
    seq_end_pos = np.arange(len(anomaly_seq)) * int(stride) + (time_steps - 1)
    valid = seq_end_pos < len(index)
    out = anomaly_seq.astype(bool).copy()
    if not valid.any():
        return out
    seq_end_idx = index[seq_end_pos[valid]]
    allowed = state.reindex(seq_end_idx).fillna("on").eq("on").values
    out_valid = out[valid] & allowed
    out[valid] = out_valid
    return out
