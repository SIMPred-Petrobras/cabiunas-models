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


def group_alerts_into_episodes(is_anom: pd.Series, merge_gap_minutes: float) -> pd.DataFrame:
    """Agrupa is_anom_point em episodios, FUNDINDO runs separados por
    menos de `merge_gap_minutes` -- diferente de contar cada run
    contiguo como um episodio (o que trata um alarme que "pisca"
    liga-desliga-liga em poucos minutos como varios eventos). Regua
    proposta pelo usuario: gap < 2h conta como o mesmo episodio.

    Retorna DataFrame com colunas start/end (ambas inclusive,
    timestamps do index de `is_anom`).
    """
    idx = is_anom.index
    flags = is_anom.values.astype(bool)
    if not flags.any():
        return pd.DataFrame(columns=["start", "end"])

    m = flags.astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1  # inclusive

    gap = pd.Timedelta(minutes=float(merge_gap_minutes))
    merged_starts = [idx[starts[0]]]
    merged_ends = [idx[ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        run_start = idx[s]
        if run_start - merged_ends[-1] < gap:
            merged_ends[-1] = idx[e]
        else:
            merged_starts.append(run_start)
            merged_ends.append(idx[e])

    return pd.DataFrame({"start": merged_starts, "end": merged_ends})


def classify_episodes_regua(
    episodes: pd.DataFrame,
    failure_times: pd.Series,
    operational_state: pd.Series,
    pre_window_hours: float = 48.0,
    post_window_hours: float = 48.0,
    min_stoppage_hours: float = 2.0,
) -> pd.DataFrame:
    """Classifica cada episodio (de `group_alerts_into_episodes`) em 3
    categorias -- regua proposta pelo usuario, equivalente ao protocolo
    oficial do Francisco (falha = parada real, janela de 48h):

      - "deteccao": o episodio COMECA dentro de ate `pre_window_hours`
        ANTES de alguma falha catalogada (`failure_times`).
      - "inconclusivo": nao e deteccao, mas a maquina teve uma PARADA
        REAL (operational_state != "on" por >= `min_stoppage_hours`
        continuas) comecando ate `post_window_hours` DEPOIS do fim do
        episodio -- pode ser um evento fisico real nao catalogado (ver
        docs/analise_automl_exp10.md, secao "Cruzamento com catalogo
        completo"); nao conta a favor nem contra o modelo.
      - "falso_positivo": nenhum dos dois -- nem antecede uma falha
        catalogada, nem antecede uma parada real qualquer.

    Retorna `episodes` com colunas extras `classe` e `falha_associada`
    (timestamp da falha detectada, quando `classe == "deteccao"`).
    """
    episodes = episodes.copy()
    fail_arr = np.sort(pd.DatetimeIndex(failure_times).values.astype("datetime64[ns]"))
    pre = pd.Timedelta(hours=float(pre_window_hours))
    post = pd.Timedelta(hours=float(post_window_hours))
    min_stop = pd.Timedelta(hours=float(min_stoppage_hours))

    def matching_failure(start: pd.Timestamp):
        start64 = np.datetime64(start)
        pos = np.searchsorted(fail_arr, start64)
        if pos < len(fail_arr):
            dt = fail_arr[pos] - start64
            if np.timedelta64(0) <= dt <= np.timedelta64(pre):
                return fail_arr[pos]
        return None

    off = operational_state != "on"
    off_diff = off.astype(int).diff().fillna(int(off.iloc[0]) if len(off) else 0)
    off_starts = off.index[off_diff == 1]
    off_ends = off.index[off_diff == -1]
    if len(off) and off.iloc[0]:
        off_starts = off_starts.insert(0, off.index[0])
    if len(off_starts) > len(off_ends):
        off_ends = off_ends.append(pd.DatetimeIndex([off.index[-1]]))
    stoppage_starts = np.array(
        [s for s, e in zip(off_starts, off_ends) if (e - s) >= min_stop],
        dtype="datetime64[ns]",
    )
    stoppage_starts.sort()

    def has_real_stoppage_after(end: pd.Timestamp) -> bool:
        end64 = np.datetime64(end)
        pos = np.searchsorted(stoppage_starts, end64)
        if pos < len(stoppage_starts):
            dt = stoppage_starts[pos] - end64
            return np.timedelta64(0) <= dt <= np.timedelta64(post)
        return False

    classes, falha_assoc = [], []
    for _, row in episodes.iterrows():
        matched = matching_failure(row["start"])
        if matched is not None:
            classes.append("deteccao")
            falha_assoc.append(pd.Timestamp(matched))
        elif has_real_stoppage_after(row["end"]):
            classes.append("inconclusivo")
            falha_assoc.append(pd.NaT)
        else:
            classes.append("falso_positivo")
            falha_assoc.append(pd.NaT)

    episodes["classe"] = classes
    episodes["falha_associada"] = falha_assoc
    return episodes


def compute_persistence_gate(score: pd.Series, threshold: float, persistence_minutes: float) -> pd.Series:
    """Canal ativo (True) quando `score` fica ACIMA de `threshold` por
    pelo menos `persistence_minutes` seguidos -- causal (so olha o
    passado, via `.rolling()` trailing). Equivalente ao "degrau: acima
    do limiar por 30 min seguidos" do detector de 4 sinais
    (DOC_EQUIPE/ROTEIRO_APRESENTACAO_TC33003A.pdf). Diferente do nosso
    debounce (`map_seq_to_point_anomalies`/`point_window`), que conta
    AMOSTRAS de sequencia (nao necessariamente minutos reais) -- aqui
    usa o grid real dos dados (mediana do `diff()` do indice)."""
    dt_seconds = score.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    n_samples = max(1, int(round((float(persistence_minutes) * 60.0) / dt_seconds)))
    above = (score > float(threshold)).astype(int)
    sustained = above.rolling(n_samples, min_periods=n_samples).sum() >= n_samples
    return sustained.fillna(False)


def compute_cusum_gate(score: pd.Series, baseline_mean: float, slack: float, h: float) -> pd.Series:
    """Gatilho CUSUM (soma cumulativa unidirecional, positiva): acumula
    `max(0, score - baseline_mean - slack)`, resetando quando o
    acumulado cairia abaixo de 0; ativa quando o acumulado ultrapassa
    `h`. Pega deriva FRACA e PERSISTENTE que nunca cruzaria um limiar
    fixo sozinha -- mesmo mecanismo do canal CUSUM do detector de 4
    sinais (H=80 no exemplo deles). Complementar a
    `compute_persistence_gate` (que pega excursao FRANCA); os dois
    juntos, em OR, formam "os dois gatilhos por canal" da arquitetura
    deles."""
    x = score.to_numpy(dtype=np.float64)
    out = np.empty(len(x), dtype=np.float64)
    cusum = 0.0
    k = float(slack)
    mu0 = float(baseline_mean)
    for i in range(len(x)):
        cusum = max(0.0, cusum + (x[i] - mu0 - k))
        out[i] = cusum
    return pd.Series(out > float(h), index=score.index)


def combine_channels_vote(
    channels: dict[str, pd.Series], min_votes: int, required_any: list[str] | None = None,
) -> pd.Series:
    """Combina canais booleanos (mesmo indice) por votacao: True se
    pelo menos `min_votes` canais estiverem ativos ao mesmo tempo, e
    (se `required_any` for dado) pelo menos um dos canais NOMEADOS ali
    estiver entre os ativos -- mesma regra do detector de 4 sinais
    ("voto >=2 canais, exigindo que um deles seja mancal ou vibracao").
    Substitui o OR/AND ingenuo testado antes (EXP33 OU EXP34: 8/8 mas
    FP explode; EXP33 E EXP34: FP melhora mas volta a perder
    deteccoes) -- a votacao com >=3 canais disponiveis (temperatura,
    vibracao, alarme de processo) tem mais graus de liberdade que um
    OR/AND binario entre so 2."""
    df = pd.DataFrame({k: v.astype(bool) for k, v in channels.items()})
    n_active = df.sum(axis=1)
    vote = n_active >= int(min_votes)
    if required_any:
        vote = vote & df[list(required_any)].any(axis=1)
    return vote


def apply_refractory(signal: pd.Series, refractory_minutes: float) -> pd.Series:
    """Suprime re-ativacoes de `signal` (bool) que comecem dentro de
    `refractory_minutes` depois do INICIO da ativacao anterior --
    "repetir nao e noticia, resolve a deriva de custo" (detector de 4
    sinais, refratario de 48h). Um episodio contiguo de True conta
    como uma unica ativacao; qualquer novo episodio que comece dentro
    da janela refrataria da ativacao anterior e suprimido por
    inteiro."""
    dt_seconds = signal.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    refractory_td = pd.Timedelta(minutes=float(refractory_minutes))

    vals = signal.to_numpy(dtype=bool)
    idx = signal.index
    out = np.zeros(len(vals), dtype=bool)
    last_trigger = None
    i = 0
    n = len(vals)
    while i < n:
        if vals[i] and (last_trigger is None or (idx[i] - last_trigger) >= refractory_td):
            last_trigger = idx[i]
            j = i
            while j < n and vals[j]:
                out[j] = True
                j += 1
            i = j
        else:
            i += 1
    return pd.Series(out, index=idx)


def compute_operational_period_days(df_point: pd.DataFrame) -> float:
    """Dias de OPERACAO VIGIADA (denominador correto da regua) --
    conta so `operational_state == "on"` (maquina rodando, quente,
    fora do blackout pos-partida), NAO o span de calendario min-max do
    indice. Usar o span de calendario infla o denominador (maquina
    tambem fica parada/em transiente boa parte do tempo) e SUBESTIMA
    falso_positivo_por_mes.

    Confirmado por revisao externa da equipe
    (DOC_EQUIPE/ROTEIRO_APRESENTACAO_TC33003A.pdf, secao "Validacao -- a
    regua"): a regua do time reduz 485 dias de calendario para 353 dias
    julgaveis (73%); no nosso EXP30 a fracao julgavel medida foi 67,3%
    (443,2 de 658,0 dias) -- o uso indevido do span de calendario nos
    fez SUBESTIMAR falso_positivo_por_mes em ~48% em resultados
    anteriores desta sessao. Ver docs/analise_automl_exp10.md, secao
    "Denominador correto da regua".
    """
    dt_seconds = df_point.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    n_on = int((df_point["operational_state"] == "on").sum())
    return float(n_on) * dt_seconds / 86400.0


def compute_regua_metrics(
    classified_episodes: pd.DataFrame,
    failure_times: pd.Series,
    period_days: float,
) -> dict:
    """Resume `classify_episodes_regua` em metricas agregadas: hit_rate
    (fracao de falhas catalogadas com >=1 episodio de deteccao) e taxa
    de falso positivo (episodios/mes), separando explicitamente os
    "inconclusivo" (nao contam nem a favor nem contra).

    IMPORTANTE: `period_days` deve ser dias de OPERACAO VIGIADA
    (`compute_operational_period_days`), NAO o span de calendario
    (`index.max() - index.min()`) -- ver docstring de
    `compute_operational_period_days` para o motivo e o erro ja
    cometido nesta sessao com o denominador errado."""
    n_falhas = len(pd.DatetimeIndex(failure_times).unique())
    falhas_detectadas = classified_episodes.loc[
        classified_episodes["classe"] == "deteccao", "falha_associada"
    ].nunique()
    n_deteccao = int((classified_episodes["classe"] == "deteccao").sum())
    n_inconclusivo = int((classified_episodes["classe"] == "inconclusivo").sum())
    n_fp = int((classified_episodes["classe"] == "falso_positivo").sum())
    months = max(1e-9, float(period_days) / 30.4375)

    return {
        "n_falhas_catalogadas": int(n_falhas),
        "falhas_detectadas": int(falhas_detectadas),
        "hit_rate": float(falhas_detectadas / n_falhas) if n_falhas > 0 else None,
        "n_episodios_deteccao": n_deteccao,
        "n_episodios_inconclusivo": n_inconclusivo,
        "n_episodios_falso_positivo": n_fp,
        "falso_positivo_por_mes": float(n_fp / months),
    }


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
    `min_duration_minutes` -- aplicado logo apos a mascara operacional,
    ANTES dos portoes de rampa/volatilidade/veto de congelamento
    (automl_pipeline.py). Ordem importa: aplicar depois desses portoes
    mediria a duracao de episodios ja fragmentados por eles (um
    precursor real longo pode virar varios pedacos curtos), derrubando
    hit_rate sem motivo -- ver docs/analise_automl_exp10.md, secao
    "EXP20" (bug de ordem encontrado e corrigido: hit_rate 47,5% com a
    ordem errada vs 90% validado com a ordem certa). Precursores reais
    tendem a persistir muito mais tempo que ruido residual (mediana
    49,5min vs 2,5min no EXP10c congelado) -- ver secao "Duracao do
    score: TP vs FP residual". So valido contra um modelo UNICO/
    congelado -- 3 tentativas mostraram que nao sobrevive a
    ENABLE_WALKFORWARD_RETRAIN; `run_automl_group` impede a combinacao."""
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


def compute_step_change_index(load_series: pd.Series, short_window_minutes: float, long_window_minutes: float) -> pd.Series:
    """Indice de mudanca de NIVEL (degrau), causal: |media movel curta -
    media movel longa| / (desvio-padrao movel longo + eps) -- mesma
    matematica do "localz" usado como feature em
    preprocess.py:_build_changepoint_features, aqui aplicado como
    indice de portao (nao alimenta o modelo). Complementa
    `compute_load_ramp_gate` (que reage a TAXA de variacao suavizada,
    nao ao nivel em si) -- pensado pra degraus quase instantaneos que a
    suavizacao por EWMA do portao de rampa dilui. Ver
    docs/analise_automl_exp10.md, secao "Portao de mudanca de nivel
    (EXP22)"."""
    s = pd.to_numeric(load_series, errors="coerce")
    dt_seconds = s.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 30.0
    sw = max(2, int(round((float(short_window_minutes) * 60.0) / dt_seconds)))
    lw = max(sw + 1, int(round((float(long_window_minutes) * 60.0) / dt_seconds)))

    short_mean = s.rolling(sw, min_periods=1).mean()
    long_mean = s.rolling(lw, min_periods=1).mean()
    long_std = s.rolling(lw, min_periods=1).std().fillna(0.0)
    return ((short_mean - long_mean).abs() / (long_std + 1e-6)).fillna(0.0)


def _nearest_within_window(t_arr: np.ndarray, ref_sorted: np.ndarray, win: pd.Timedelta) -> np.ndarray:
    """Para cada instante em t_arr, indice em ref_sorted (ja ordenado)
    do mais proximo dentro de `win`, ou -1 se nao houver nenhum.
    Mesma logica do `nearest_alarm` usado nas investigacoes ad-hoc de
    cruzamento com catalogo (dataset_francisco_lara/gen_figura_*.py)."""
    out = np.full(len(t_arr), -1, dtype=np.int64)
    if len(ref_sorted) == 0 or len(t_arr) == 0:
        return out
    pos = np.searchsorted(ref_sorted, t_arr)
    for i in range(len(t_arr)):
        p = pos[i]
        cands = [c for c in (p - 1, p) if 0 <= c < len(ref_sorted)]
        if not cands:
            continue
        best = min(cands, key=lambda c: abs(t_arr[i] - ref_sorted[c]))
        if abs(t_arr[i] - ref_sorted[best]) <= win:
            out[i] = best
    return out


def annotate_alert_catalog_context(
    df_point: pd.DataFrame,
    catalog_df: pd.DataFrame,
    window_hours: float,
    time_col: str = "Data da Ocorrencia",
    tag_col: str = "Tag",
) -> pd.DataFrame:
    """Anota cada is_anom_point==1 com o alarme mais proximo de um
    catalogo AMPLO (ex: 47 tags) dentro de +-window_hours -- PURAMENTE
    INFORMATIVO, nunca altera is_anom_point/hit_rate/normal_alert_rate.

    Implementa a estrategia "classificar por confianca e corroborar em
    tempo de alerta" (docs/analise_automl_exp10.md, secao "Cruzamento
    com catalogo completo"): 88,1% dos episodios residuais "amarelos"
    contra os TRIPs curados eram, na verdade, sinal real de outro
    alarme do processo dentro de 24h. Em vez de suprimir esses pontos
    dentro da pipeline (a tentativa mais direcionada disso -- supressao
    cirurgica baseada em mecanismo -- foi testada offline e REJEITADA
    por custar 2 dos 8 TRIPs reais, mesma secao do doc), essa funcao so
    adiciona contexto para o consumidor do alerta (dashboard/operador)
    decidir a prioridade.

    Colunas adicionadas: alert_catalog_tag, alert_catalog_time,
    alert_catalog_distance_h, alert_confidence (
    "explicado_catalogo" | "isolado" | None se nao for anomalia).
    """
    df_point = df_point.copy()
    df_point["alert_catalog_tag"] = pd.array([None] * len(df_point), dtype=object)
    df_point["alert_catalog_time"] = pd.NaT
    df_point["alert_catalog_distance_h"] = np.nan

    is_anom = df_point["is_anom_point"].values == 1
    confidence = np.where(is_anom, "isolado", None).astype(object)
    df_point["alert_confidence"] = confidence

    anom_idx = np.where(is_anom)[0]
    if len(anom_idx) == 0 or catalog_df.empty or time_col not in catalog_df.columns:
        return df_point

    ref = catalog_df[[c for c in (time_col, tag_col) if c in catalog_df.columns]].dropna(subset=[time_col])
    ref = ref.sort_values(time_col)
    ref_arr = ref[time_col].values.astype("datetime64[ns]")
    ref_tags = ref[tag_col].values if tag_col in ref.columns else None

    t_arr = df_point.index.values[anom_idx].astype("datetime64[ns]")
    win = pd.Timedelta(hours=float(window_hours))
    match = _nearest_within_window(t_arr, ref_arr, win)

    hit = match >= 0
    rows = anom_idx[hit]
    matched_ref = match[hit]
    if len(rows) == 0:
        return df_point

    dist_h = np.abs(t_arr[hit] - ref_arr[matched_ref]).astype("timedelta64[s]").astype(np.float64) / 3600.0
    col_time = df_point.columns.get_loc("alert_catalog_time")
    col_dist = df_point.columns.get_loc("alert_catalog_distance_h")
    col_conf = df_point.columns.get_loc("alert_confidence")
    df_point.iloc[rows, col_time] = ref_arr[matched_ref]
    df_point.iloc[rows, col_dist] = dist_h
    df_point.iloc[rows, col_conf] = "explicado_catalogo"
    if ref_tags is not None:
        col_tag = df_point.columns.get_loc("alert_catalog_tag")
        df_point.iloc[rows, col_tag] = ref_tags[matched_ref]

    return df_point


def compute_catalog_enrichment_control(
    df_point: pd.DataFrame,
    catalog_df: pd.DataFrame,
    window_hours: float,
    n_samples: int = 5000,
    random_seed: int = 42,
    time_col: str = "Data da Ocorrencia",
) -> dict:
    """Controle negativo obrigatorio para `annotate_alert_catalog_context`:
    mede que fracao de instantes ALEATORIOS (nao anomalos, so
    operational_state=="on") cai perto de algum alarme do catalogo
    dentro de +-window_hours, e compara com a fracao medida entre os
    pontos anomalos. Sem isso, uma taxa alta de "explicado" pode ser
    so reflexo de um catalogo denso (muitos alarmes, janela larga),
    nao evidencia real de corroboracao.

    Motivado por revisao externa da equipe (docs/analise_automl_exp10.md,
    secao "Controle negativo do enriquecimento por catalogo"): a janela
    de +-24h usada originalmente (EXP30/31) dava 71,7% de "explicado" ja
    em instantes aleatorios, contra 97,6% nos pontos anomalos --
    enriquecimento de so 1,36x, bem mais fraco do que o "88,1%/97,6%"
    sozinho fazia parecer. Retorna as duas taxas e o fator de
    enriquecimento (`anomaly_rate / baseline_rate`); um enriquecimento
    proximo de 1,0 significa que a janela esta larga demais pra ter
    poder discriminativo.
    """
    on_idx = df_point.index[df_point["operational_state"] == "on"]
    rng = np.random.default_rng(random_seed)
    n = min(int(n_samples), len(on_idx))
    if n == 0 or catalog_df.empty or time_col not in catalog_df.columns:
        return {
            "window_hours": float(window_hours), "n_random_samples": 0,
            "baseline_explained_rate": float("nan"), "anomaly_explained_rate": float("nan"),
            "enrichment_factor": float("nan"),
        }
    sample_pos = rng.choice(len(on_idx), size=n, replace=False)
    random_times = on_idx[sample_pos].values.astype("datetime64[ns]")

    ref = catalog_df[[time_col]].dropna(subset=[time_col]).sort_values(time_col)
    ref_arr = ref[time_col].values.astype("datetime64[ns]")
    win = pd.Timedelta(hours=float(window_hours))
    baseline_match = _nearest_within_window(random_times, ref_arr, win)
    baseline_rate = float((baseline_match >= 0).mean())

    anom_mask = df_point["is_anom_point"] == 1
    if anom_mask.any() and "alert_confidence" in df_point.columns:
        anomaly_rate = float((df_point.loc[anom_mask, "alert_confidence"] == "explicado_catalogo").mean())
    else:
        anomaly_rate = float("nan")

    enrichment = (anomaly_rate / baseline_rate) if baseline_rate > 0 else float("nan")
    return {
        "window_hours": float(window_hours),
        "n_random_samples": int(n),
        "baseline_explained_rate": baseline_rate,
        "anomaly_explained_rate": anomaly_rate,
        "enrichment_factor": enrichment,
    }


def apply_step_change_gate(df_point: pd.DataFrame, step_index: pd.Series, threshold: float) -> pd.DataFrame:
    """Suprime is_anom_point quando `compute_step_change_index` ultrapassa
    `threshold` -- causal por construcao (a serie de entrada ja e uma
    janela trailing)."""
    idx_at_point = step_index.reindex(df_point.index, method="ffill")
    blocked = (idx_at_point > float(threshold)).fillna(False)

    df_point = df_point.copy()
    df_point["step_change_gate_blocked"] = blocked.values
    df_point.loc[blocked.values, "is_anom_point"] = 0
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
