from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .feature_engineering import generate_features


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _long_gap_mask(series: pd.Series, interpolate_limit: int) -> pd.Series:
    missing = series.isna()
    grp = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_len = missing.groupby(grp).transform("sum")
    return missing & (run_len > int(interpolate_limit))


# ---------------------------------------------------------------------------
# 1. Detecção de valores sentinela
# ---------------------------------------------------------------------------

def detect_and_mask_sentinels(
    series: pd.Series,
    low: Optional[float],
    high: Optional[float],
) -> Tuple[pd.Series, int]:
    """
    Substitui por NaN valores fora dos limites físicos do sensor antes da interpolação.

    Valores como -40.5 °C indicam falha de instrumento (thermocouple aberto) e criam
    gradientes artificiais de 600+ °C/30 s que disparam falsos positivos no autoencoder.
    Marcá-los como NaN permite que a interpolação (limitada a INTERPOLATE_LIMIT pontos)
    ou o long_gap_mask os trate corretamente.

    Returns:
        (série corrigida, número de pontos mascarados)
    """
    result = series.copy()
    mask = pd.Series(False, index=series.index)
    if low is not None:
        mask |= result < float(low)
    if high is not None:
        mask |= result > float(high)
    result[mask] = np.nan
    n_masked = int(mask.sum())
    return result, n_masked


# ---------------------------------------------------------------------------
# 2. Filtro de Hampel para spikes isolados
# ---------------------------------------------------------------------------

def hampel_filter(
    series: pd.Series,
    window: int = 5,
    sigma: float = 3.0,
) -> Tuple[pd.Series, int]:
    """
    Filtro de Hampel: detecta e substitui outliers isolados usando MAD em janela deslizante.

    Diferente do clipping global, o Hampel usa contexto local, preservando rampas
    sustentadas (startups) e removendo apenas picos de medição isolados.

    Algoritmo:
        mediana_local = mediana(janela de 2k+1 pontos centrada no ponto)
        MAD_local = mediana(|x - mediana_local|) na mesma janela
        spike se |x - mediana_local| > sigma * 1.4826 * MAD_local
        substituição: mediana_local (não NaN, série contínua)

    Returns:
        (série filtrada, número de spikes substituídos)
    """
    k = max(1, int(window))
    w = 2 * k + 1
    result = series.copy().astype(float)

    roll_median = series.rolling(window=w, center=True, min_periods=1).median()
    roll_mad = (series - roll_median).abs().rolling(window=w, center=True, min_periods=1).median()
    threshold = float(sigma) * 1.4826 * roll_mad

    # Quando MAD=0 (sinal perfeitamente plano) qualquer desvio da mediana é spike;
    # quando MAD>0 usa o critério sigma*1.4826*MAD como limiar.
    spike_mask = (series - roll_median).abs() > threshold

    result[spike_mask] = roll_median[spike_mask]
    n_spikes = int(spike_mask.sum())
    return result, n_spikes


# ---------------------------------------------------------------------------
# 3. Máscara de gradiente estável para normalização robusta
# ---------------------------------------------------------------------------

def build_stable_gradient_mask(
    df: pd.DataFrame,
    sensor: str,
    quantile: float = 0.95,
) -> pd.Series:
    """
    Identifica pontos de operação estável: gradiente do sensor abaixo do quantile dado.

    Usado para calcular center/scale da normalização somente sobre dados estáveis,
    evitando que a distribuição bimodal ON/OFF desloque o z-score para o "vácuo"
    entre os dois regimes e comprima o sinal de temperatura de operação normal.
    """
    s = pd.to_numeric(df[sensor], errors="coerce")
    grad = s.diff().abs().fillna(0.0)
    thr = float(grad.quantile(float(np.clip(quantile, 0.0, 1.0))))
    return grad <= thr


# ---------------------------------------------------------------------------
# 4. Máscara de runs constantes (forward-fill upstream)
# ---------------------------------------------------------------------------

def build_constant_run_mask(
    series: pd.Series,
    min_length: int = 3,
) -> pd.Series:
    """
    Detecta runs de ≥ min_length valores idênticos consecutivos.

    Dado pré-interpolado com last-value-carried-forward cria plateaus artificiais:
        [570.2, 570.2, 570.2, 570.2]  ← gap de 4 pontos preenchido com o último valor

    Esses segmentos não são dados reais — o sensor simplesmente não mediu nada novo.
    Excluí-los do treino impede que o AE aprenda a reconstruir plateaus como "normal"
    e elimina falsos positivos quando esses plateaus aparecem durante a inferência.

    Returns:
        Máscara booleana True onde o ponto faz parte de um run suspeito.
    """
    s = pd.to_numeric(series, errors="coerce")
    is_equal = (s == s.shift(1)) & s.notna() & s.shift(1).notna()
    grp = (~is_equal).cumsum()
    run_len = is_equal.groupby(grp).transform("sum")
    # run_len conta quantos pontos consecutivos são iguais ao anterior.
    # Um run de min_length pontos iguais tem run_len >= min_length-1 nos últimos pontos.
    suspect = run_len >= (min_length - 1)
    # Propagar para incluir o ponto que iniciou o run (tem is_equal=False mas mesmo valor).
    return (suspect | suspect.shift(-1).fillna(False))


# ---------------------------------------------------------------------------
# 5. Máscara de exclusão pós-startup
# ---------------------------------------------------------------------------

def build_startup_exclusion_mask(
    index: pd.DatetimeIndex,
    sensor_series: pd.Series,
    off_threshold: float,
    minutes_after_startup: int,
) -> pd.Series:
    """
    Exclui do treino X minutos após cada evento de startup (sensor cruza OFF→ON).

    O modelo não deve aprender a rampa de temperatura (32→670 °C em ~3 min) como
    padrão normal — ela cria janelas com gradiente extremo que o autoencoder não
    consegue reconstruir, gerando falsos positivos no início de cada partida.

    Startup detectado quando: sensor_series cruza de ≤ off_threshold para > off_threshold.
    """
    mask = pd.Series(False, index=index)
    if minutes_after_startup <= 0:
        return mask

    s = pd.to_numeric(sensor_series.reindex(index), errors="coerce").ffill().bfill()
    is_off = s <= float(off_threshold)
    # Startup: ponto atual ON, ponto anterior OFF
    startup = (~is_off) & is_off.shift(1, fill_value=True)
    startup_times = index[startup.values]

    delta = pd.Timedelta(minutes=int(minutes_after_startup))
    for t in startup_times:
        mask.loc[(mask.index >= t) & (mask.index <= t + delta)] = True
    return mask


# ---------------------------------------------------------------------------
# 5b. Máscara de spike de gradiente para limpeza do treino
# ---------------------------------------------------------------------------

def build_gradient_spike_mask(
    df: pd.DataFrame,
    sensor: str,
    running_series: Optional[pd.Series],
    std_mult: float = 8.0,
    suppress_minutes: int = 60,
    transition_minutes: int = 0,
) -> pd.Series:
    """
    Exclui ±suppress_minutes ao redor de spikes de gradiente local.

    Calibração: μ/σ calculados sobre steady state (ON + fora de janelas de transição).
    - running_series > 0.5 → pontos ON
    - transition_minutes > 0 → exclui adicionalmente ±N min ao redor de cada
      mudança de estado (ON↔OFF), refinando o steady state de calibração.

    Pode ser usado tanto para exclusão de treino quanto para supressão no scoring.
    """
    s = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill()
    grad = s.diff().abs().fillna(0.0)

    if running_series is not None:
        run = running_series.reindex(df.index).fillna(0.0)
        steady_mask = run > 0.5
    else:
        steady_mask = pd.Series(True, index=df.index)

    if transition_minutes > 0 and running_series is not None:
        run_aligned = running_series.reindex(df.index).fillna(0.0)
        state_changes = (run_aligned > 0.5).astype(int).diff().abs().fillna(0) > 0
        change_times = df.index[state_changes].tolist()
        delta_trans = pd.Timedelta(minutes=int(transition_minutes))
        trans_mask = pd.Series(False, index=df.index)
        for t in change_times:
            trans_mask.loc[(trans_mask.index >= t - delta_trans) &
                           (trans_mask.index <= t + delta_trans)] = True
        steady_mask = steady_mask & ~trans_mask

    grad_steady = grad[steady_mask]
    if len(grad_steady) < 10:
        return pd.Series(False, index=df.index)

    mu = float(grad_steady.mean())
    sigma = float(grad_steady.std())
    if sigma == 0.0:
        return pd.Series(False, index=df.index)

    spike_locs = grad > (mu + float(std_mult) * sigma)
    spike_times = df.index[spike_locs]

    out = pd.Series(False, index=df.index)
    if len(spike_times) == 0:
        return out

    delta = pd.Timedelta(minutes=int(suppress_minutes))
    for t in spike_times:
        out.loc[(out.index >= t - delta) & (out.index <= t + delta)] = True
    return out


# ---------------------------------------------------------------------------
# 5c. Construção do DataFrame do sensor (com sentinel + interpolação)
# ---------------------------------------------------------------------------

def build_sensor_dataframe(
    cfg: PipelineConfig, df_feat: pd.DataFrame, df_raw: pd.DataFrame, sensor: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Retorna DF indexado pelo tempo com coluna(s) do sensor + máscara de gaps longos.

    Pipeline de limpeza:
        1. Selecionar colunas e deduplicar timestamps
        2. Converter para numérico
        3. Detectar e mascarar valores sentinela (se SENTINEL_MODE != "none")
        4. Calcular long_gap_mask ANTES da interpolação (inclui buracos criados por sentinelas)
        5. Interpolar (limit=INTERPOLATE_LIMIT), fallback temporal, fallback borda
    """
    source = cfg.TRAIN_SOURCE.lower()

    if source == "raw":
        if sensor not in df_raw.columns:
            raise ValueError(f"Sensor '{sensor}' nao existe em RAW.")
        source_df = df_raw
    else:
        if sensor not in df_feat.columns:
            raise ValueError(f"Sensor '{sensor}' nao existe em FEATURES.")
        source_df = df_feat

    context_cols = [
        c
        for c in (cfg.CONTEXT_COLS or [])
        if cfg.ENABLE_CONTEXT_FEATURES and c in source_df.columns and c not in {cfg.TIME_COL, sensor}
    ]
    selected_cols = [cfg.TIME_COL, sensor, *context_cols]
    df_use = source_df[selected_cols].copy()

    # Remoção de common-mode: substitui o sensor pelo resíduo contra a média
    # leave-one-out dos irmãos do grupo (ex: os 6 TC382). Feito aqui porque
    # source_df ainda tem TODOS os sensores; df_use já está reduzido. Mantém
    # univariado e expõe a divergência local (anomalia) neutralizando deriva/carga
    # compartilhada. No-op se o sensor não está no grupo ou faltam irmãos.
    if cfg.ENABLE_COMMON_MODE_REMOVAL:
        group = [c for c in (cfg.COMMON_MODE_GROUP or []) if c in source_df.columns]
        siblings = [c for c in group if c != sensor]
        if sensor in group and siblings:
            s = pd.to_numeric(df_use[sensor], errors="coerce")
            sib_mean = source_df[siblings].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            df_use[sensor] = (s - sib_mean).astype(float)
            print(f"[COMMON-MODE] sensor={sensor} = valor − média({len(siblings)} irmãos do grupo)")

    before_dupes = int(df_use.duplicated(subset=[cfg.TIME_COL]).sum())
    if before_dupes:
        print(f"[DATA-CLEAN] sensor={sensor}: removendo {before_dupes} timestamps duplicados.")
    df_use = df_use.sort_values(cfg.TIME_COL).drop_duplicates(subset=[cfg.TIME_COL], keep="first")

    for col in selected_cols:
        if col != cfg.TIME_COL:
            df_use[col] = pd.to_numeric(df_use[col], errors="coerce")

    # --- Detecção de sentinelas (antes da interpolação para não propagar artefatos) ---
    if cfg.SENTINEL_MODE.lower() != "none":
        df_use[sensor], n_sentinel = detect_and_mask_sentinels(
            df_use[sensor], cfg.SENTINEL_LOW, cfg.SENTINEL_HIGH
        )
        if n_sentinel:
            print(
                f"[SENTINEL] sensor={sensor}: {n_sentinel} valores sentinela mascarados "
                f"(low={cfg.SENTINEL_LOW}, high={cfg.SENTINEL_HIGH})."
            )

    # long_gap_mask calculado sobre os dados já com sentinelas como NaN
    long_gap_raw = _long_gap_mask(df_use[sensor], cfg.INTERPOLATE_LIMIT)

    df_use = df_use.set_index(cfg.TIME_COL).sort_index()
    long_gap_raw.index = df_use.index

    df_use[sensor] = df_use[sensor].interpolate(limit=int(cfg.INTERPOLATE_LIMIT), limit_direction="both")
    n_missing_after_interp = int(df_use[sensor].isna().sum())
    if n_missing_after_interp:
        print(
            f"[DATA-CLEAN] sensor={sensor}: {n_missing_after_interp} NaNs restantes apos "
            f"interpolacao limitada (INTERPOLATE_LIMIT={cfg.INTERPOLATE_LIMIT}). "
            "Aplicando fallback de interpolacao temporal."
        )
        df_use[sensor] = df_use[sensor].interpolate(method="time", limit_direction="both")

    n_missing_after_fallback = int(df_use[sensor].isna().sum())
    if n_missing_after_fallback:
        print(
            f"[DATA-CLEAN] sensor={sensor}: {n_missing_after_fallback} NaNs restantes "
            "apos interpolacao temporal; preenchendo bordas."
        )
        df_use[sensor] = df_use[sensor].ffill().bfill()

    assert not df_use[sensor].isna().any(), "Falha interna: NaN remanescente apos interpolacao."

    # Preenche NaN nas colunas de contexto (não interpoladas acima)
    for ctx_col in context_cols:
        if ctx_col in df_use.columns and df_use[ctx_col].isna().any():
            df_use[ctx_col] = (
                df_use[ctx_col]
                .interpolate(limit=int(cfg.INTERPOLATE_LIMIT), limit_direction="both")
                .ffill()
                .bfill()
                .fillna(0.0)
            )

    return df_use, long_gap_raw


# ---------------------------------------------------------------------------
# 6. Aplicação do filtro de Hampel pós-interpolação
# ---------------------------------------------------------------------------

def apply_hampel_filter(
    df: pd.DataFrame,
    sensor: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Aplica o filtro de Hampel ao sensor principal se ENABLE_HAMPEL_FILTER=True.
    Chamado após interpolação e antes da divisão df_normal/df_all.
    """
    if not cfg.ENABLE_HAMPEL_FILTER:
        return df

    out = df.copy()
    cleaned, n_spikes = hampel_filter(out[sensor], window=cfg.HAMPEL_WINDOW, sigma=cfg.HAMPEL_SIGMA)
    if n_spikes:
        print(
            f"[HAMPEL] sensor={sensor}: {n_spikes} spikes isolados substituidos "
            f"(window={cfg.HAMPEL_WINDOW}, sigma={cfg.HAMPEL_SIGMA})."
        )
    out[sensor] = cleaned
    return out


# ---------------------------------------------------------------------------
# 7. Feature engineering wrapper
# ---------------------------------------------------------------------------

def apply_feature_engineering(
    df_normal: pd.DataFrame,
    df_all: pd.DataFrame,
    sensor: str,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (cfg.ENABLE_ROLLING_FEATURES or cfg.ENABLE_SPECTRAL_FEATURES or cfg.ENABLE_CONTEXT_FEATURES or cfg.ENABLE_TREND_FEATURES):
        return df_normal, df_all

    return (
        generate_features(df_normal, sensor, cfg),
        generate_features(df_all, sensor, cfg),
    )


# ---------------------------------------------------------------------------
# 8. Máscara de exclusão baseada em alarmes
# ---------------------------------------------------------------------------

def build_exclusion_mask(
    index: pd.DatetimeIndex,
    alarm_times: pd.Series,
    minutes: int,
    minutes_before: int | None = None,
    minutes_after: int | None = None,
) -> pd.Series:
    """Marca janelas de exclusão de treino em torno dos alarmes.

    Quando minutes_before/minutes_after são None, usa `minutes` simétrico (legado).
    A assimetria permite excluir mais depois do alarme (evento + recuperação) e
    menos antes, preservando o regime normal pré-alarme no treino.
    """
    exclude = pd.Series(False, index=index)
    before = pd.Timedelta(minutes=minutes_before if minutes_before is not None else minutes)
    after = pd.Timedelta(minutes=minutes_after if minutes_after is not None else minutes)
    for t in alarm_times.values:
        t0 = pd.Timestamp(t) - before
        t1 = pd.Timestamp(t) + after
        exclude.loc[(exclude.index >= t0) & (exclude.index <= t1)] = True
    return exclude


# ---------------------------------------------------------------------------
# 9. Clipping de outliers globais
# ---------------------------------------------------------------------------

def clip_outliers(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    mode = cfg.OUTLIER_MODE.lower()
    if mode == "none":
        return df

    out = df.copy()
    if mode == "quantile":
        q_low = out.quantile(cfg.OUTLIER_Q_LOW)
        q_high = out.quantile(cfg.OUTLIER_Q_HIGH)
        return out.clip(lower=q_low, upper=q_high, axis=1)

    if mode == "mad":
        med = out.median(axis=0)
        mad = (out - med).abs().median(axis=0).replace(0, 1e-9)
        low = med - cfg.OUTLIER_MAD_K * 1.4826 * mad
        high = med + cfg.OUTLIER_MAD_K * 1.4826 * mad
        return out.clip(lower=low, upper=high, axis=1)

    raise ValueError("OUTLIER_MODE invalido. Use 'none', 'quantile' ou 'mad'.")


# ---------------------------------------------------------------------------
# 10. Normalização com suporte a subconjunto estável
# ---------------------------------------------------------------------------

def normalize_train_only(
    cfg: PipelineConfig,
    df_normal: pd.DataFrame,
    df_all: pd.DataFrame,
    stable_mask: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Calcula center/scale a partir de df_normal (dados sem alarmes).

    Se NORMALIZE_ON_STABLE_ONLY=True e stable_mask fornecida, restringe o cálculo
    de estatísticas aos pontos estáveis (baixo gradiente), corrigindo a distorção
    causada pela distribuição bimodal ON/OFF dos sensores de temperatura.

    Retorna: (df_normal_z, df_all_z, center, scale)
    """
    mode = cfg.NORMALIZE_MODE.lower()

    df_ref = df_normal
    if cfg.NORMALIZE_ON_STABLE_ONLY and stable_mask is not None:
        stable_in_normal = stable_mask.reindex(df_normal.index).fillna(False)
        df_stable = df_normal[stable_in_normal]
        min_pts = max(cfg.TIME_STEPS * 2, 100)
        if len(df_stable) >= min_pts:
            df_ref = df_stable
            print(
                f"[NORMALIZE] Usando {len(df_ref):,} pontos estaveis de {len(df_normal):,} "
                f"normais ({100*len(df_ref)/max(len(df_normal),1):.1f}%) para center/scale."
            )
        else:
            print(
                f"[NORMALIZE] Pontos estaveis insuficientes ({len(df_stable)}); "
                "usando conjunto normal completo."
            )

    if mode == "zscore":
        center = df_ref.mean(axis=0)
        scale = df_ref.std(axis=0).replace(0, 1.0)
    elif mode == "robust":
        center = df_ref.median(axis=0)
        q1 = df_ref.quantile(0.25)
        q3 = df_ref.quantile(0.75)
        scale = (q3 - q1).replace(0, 1.0)
    else:
        raise ValueError("NORMALIZE_MODE invalido. Use 'zscore' ou 'robust'.")

    df_normal_z = (df_normal - center) / scale
    df_all_z = (df_all - center) / scale

    return df_normal_z, df_all_z, center, scale
