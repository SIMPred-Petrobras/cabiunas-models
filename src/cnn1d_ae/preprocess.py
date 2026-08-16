from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from .config import PipelineConfig


# Janelas menores que isso (amostras) nao tem tamanho suficiente para
# kurtosis/skewness/crest factor serem estimadores minimamente estaveis --
# essas features de "textura" so sao calculadas para janelas >= este valor.
TEXTURE_MIN_WINDOW = 60


def _build_derived_features(df: pd.DataFrame, sensor: str, windows: List[int]) -> pd.DataFrame:
    """Features derivadas em multiplas escalas de tempo (uma passada por
    janela em `windows`), alem do delta instantaneo (1 passo, independente
    de janela). Para cada janela w: media/desvio movel (nivel/variabilidade
    recente) e `trend_w` = valor atual menos o valor de w passos atras
    (tendencia/inclinacao ao longo daquela janela especifica) -- pensado
    para capturar precursores lentos que uma unica janela curta nao pega.

    Para janelas >= TEXTURE_MIN_WINDOW, adiciona tambem features de
    "textura" do sinal (mudanca de forma da distribuicao, nao so
    nivel/variancia): kurtosis, skewness e crest factor (pico/RMS) --
    comuns em condition monitoring de vibracao para pegar sinais ficando
    mais "impulsivos" antes de uma falha de mancal.
    Ver docs/analise_automl_exp7_planejamento.md."""
    out = df.copy()
    out[f"{sensor}__delta_1"] = out[sensor].diff().fillna(0.0)
    for window in windows:
        w = max(2, int(window))
        out[f"{sensor}__roll_med_{w}"] = out[sensor].rolling(w, min_periods=1).median()
        out[f"{sensor}__roll_std_{w}"] = out[sensor].rolling(w, min_periods=1).std().fillna(0.0)
        out[f"{sensor}__trend_{w}"] = (out[sensor] - out[sensor].shift(w)).fillna(0.0)
        if w >= TEXTURE_MIN_WINDOW:
            roll = out[sensor].rolling(w, min_periods=max(10, w // 4))
            out[f"{sensor}__roll_kurt_{w}"] = roll.kurt().fillna(0.0)
            out[f"{sensor}__roll_skew_{w}"] = roll.skew().fillna(0.0)
            rms = np.sqrt((out[sensor] ** 2).rolling(w, min_periods=1).mean())
            peak = out[sensor].abs().rolling(w, min_periods=1).max()
            out[f"{sensor}__crest_{w}"] = (peak / rms.replace(0, np.nan)).fillna(0.0)
    return out


def _derived_windows(cfg: PipelineConfig) -> List[int]:
    return list(cfg.DERIVED_ROLLING_WINDOWS) if cfg.DERIVED_ROLLING_WINDOWS else [cfg.DERIVED_ROLLING_WINDOW]


def _build_changepoint_features(
    df: pd.DataFrame, sensor: str, short_window: int, long_window: int, cusum_k: float
) -> pd.DataFrame:
    """Features causais de deteccao de mudanca de regime, complementares ao
    threshold por percentil global (que so dispara quando o valor cruza um
    limiar historico fixo). Aqui o alvo e detectar quando o sinal ja diverge
    da sua propria linha de base *local* recente, mesmo sem cruzar esse
    limiar -- pensado para os casos "sem deteccao" do EXP7 item 1+2, que
    mostram inflexao de tendencia sutil mas dentro do ruido normal do sinal.

    - `localz_{sw}_{lw}`: z-score da media de curto prazo (sw) em relacao a
      media/desvio de longo prazo (lw) -- "quantos desvios-padrao locais a
      media recente ja esta longe da linha de base".
    - `cusum_pos_{lw}` / `cusum_neg_{lw}`: CUSUM causal (Page's CUSUM) do
      desvio em relacao a media movel de longo prazo, com folga (slack)
      proporcional ao desvio padrao local -- acumula evidencia de um desvio
      sustentado (mesmo pequeno) e reseta quando o sinal volta a linha de
      base. Precisa de loop sequencial: o reset em max(0, ...)/min(0, ...)
      nao vetoriza em pandas.rolling.

    Ver docs/analise_automl_exp7_planejamento.md (item 3)."""
    out = df.copy()
    x = out[sensor]
    sw = max(2, int(short_window))
    lw = max(sw + 1, int(long_window))

    short_mean = x.rolling(sw, min_periods=1).mean()
    long_mean = x.rolling(lw, min_periods=1).mean()
    long_std = x.rolling(lw, min_periods=1).std().fillna(0.0)
    eps = 1e-6
    out[f"{sensor}__localz_{sw}_{lw}"] = ((short_mean - long_mean) / (long_std + eps)).fillna(0.0)

    vals = x.to_numpy(dtype=np.float64)
    mean_arr = long_mean.to_numpy(dtype=np.float64)
    k_arr = float(cusum_k) * long_std.to_numpy(dtype=np.float64)
    n = len(vals)
    pos = np.empty(n, dtype=np.float64)
    neg = np.empty(n, dtype=np.float64)
    p = 0.0
    ng = 0.0
    for i in range(n):
        dev = vals[i] - mean_arr[i]
        p = max(0.0, p + dev - k_arr[i])
        ng = max(0.0, ng - dev - k_arr[i])
        pos[i] = p
        neg[i] = ng
    out[f"{sensor}__cusum_pos_{lw}"] = pos
    out[f"{sensor}__cusum_neg_{lw}"] = neg
    return out


THERMAL_ARRAY_SPREAD_COL = "thermal_array_spread"


def _build_thermal_array_spread(
    cfg: PipelineConfig, df_use: pd.DataFrame, source_df: pd.DataFrame, array_sensors: List[str]
) -> pd.DataFrame:
    """Pseudo-sensor de desbalanceamento termico: desvio-padrao, a cada
    instante, entre as leituras de um array de sondas fisicamente
    redundantes (ex: os 6 termopares TC382_0X_A do mesmo anel de exaustao,
    corr >=0.99 par-a-par -- ver docs/analise_automl_exp9_planejamento.md).
    Individualmente cada sonda e quase identica ao alvo (adiciona-las cruas
    seria quase circular); mas um array que comeca a divergir entre sondas
    pode ser um precursor real de degradacao localizada antes da sonda
    monitorada isoladamente cruzar o limiar de alarme. Reusa
    `_build_derived_features` para dar ao pseudo-sensor o mesmo tratamento
    multi-escala/textura de qualquer sensor real."""
    missing = [s for s in array_sensors if s not in source_df.columns]
    if missing:
        raise ValueError(f"Sensores do array termico nao encontrados na fonte: {missing}")

    array_df = source_df[[cfg.TIME_COL] + list(array_sensors)].copy()
    for s in array_sensors:
        array_df[s] = pd.to_numeric(array_df[s], errors="coerce")
    array_df = array_df.set_index(cfg.TIME_COL).sort_index()
    for s in array_sensors:
        array_df[s] = array_df[s].interpolate(limit=int(cfg.INTERPOLATE_LIMIT), limit_direction="both")
        array_df[s] = array_df[s].ffill().bfill()
    array_df = array_df.reindex(df_use.index)

    out = df_use.copy()
    out[THERMAL_ARRAY_SPREAD_COL] = array_df[array_sensors].std(axis=1, skipna=True).fillna(0.0)
    if cfg.ENABLE_DERIVED_FEATURES:
        out = _build_derived_features(out, sensor=THERMAL_ARRAY_SPREAD_COL, windows=_derived_windows(cfg))
    return out


def _long_gap_mask(series: pd.Series, interpolate_limit: int) -> pd.Series:
    missing = series.isna()
    grp = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_len = missing.groupby(grp).transform("sum")
    return missing & (run_len > int(interpolate_limit))


def build_sensor_dataframe(
    cfg: PipelineConfig, df_feat: pd.DataFrame, df_raw: pd.DataFrame, sensor: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Retorna DF indexado pelo tempo com coluna(s) do sensor + mascara de pontos em gaps longos.
    """
    source = cfg.TRAIN_SOURCE.lower()

    if source == "raw":
        if sensor not in df_raw.columns:
            raise ValueError(f"Sensor '{sensor}' nao existe em RAW.")
        df_use = df_raw[[cfg.TIME_COL, sensor]].copy()
    else:
        if sensor not in df_feat.columns:
            raise ValueError(f"Sensor '{sensor}' nao existe em FEATURES.")
        df_use = df_feat[[cfg.TIME_COL, sensor]].copy()

    df_use[sensor] = pd.to_numeric(df_use[sensor], errors="coerce")

    long_gap_raw = _long_gap_mask(df_use[sensor], cfg.INTERPOLATE_LIMIT)

    df_use = df_use.set_index(cfg.TIME_COL).sort_index()
    long_gap_raw.index = df_use.index

    df_use[sensor] = df_use[sensor].interpolate(limit=int(cfg.INTERPOLATE_LIMIT), limit_direction="both")
    df_use[sensor] = df_use[sensor].ffill().bfill()

    if cfg.ENABLE_DERIVED_FEATURES:
        df_use = _build_derived_features(df_use, sensor=sensor, windows=_derived_windows(cfg))
    if cfg.ENABLE_CHANGEPOINT_FEATURES:
        df_use = _build_changepoint_features(
            df_use, sensor=sensor, short_window=cfg.CHANGEPOINT_SHORT_WINDOW,
            long_window=cfg.CHANGEPOINT_LONG_WINDOW, cusum_k=cfg.CHANGEPOINT_CUSUM_K,
        )

    return df_use, long_gap_raw


def build_group_dataframe(
    cfg: PipelineConfig,
    df_feat: pd.DataFrame,
    df_raw: pd.DataFrame,
    sensors: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega múltiplos sensores de uma só vez, alinhados no mesmo índice temporal.
    A máscara de gaps longos é a união (OR) de todos os canais — conservadora,
    exclui o ponto se QUALQUER sensor estava ausente por muito tempo.
    """
    source = cfg.TRAIN_SOURCE.lower()
    source_df = df_raw if source == "raw" else df_feat

    missing = [s for s in sensors if s not in source_df.columns]
    if missing:
        raise ValueError(f"Sensores nao encontrados na fonte '{source}': {missing}")

    df_use = source_df[[cfg.TIME_COL] + list(sensors)].copy()

    long_gap_union: pd.Series | None = None
    for s in sensors:
        df_use[s] = pd.to_numeric(df_use[s], errors="coerce")
        lgm = _long_gap_mask(df_use[s], cfg.INTERPOLATE_LIMIT)
        long_gap_union = lgm if long_gap_union is None else (long_gap_union | lgm)

    df_use = df_use.set_index(cfg.TIME_COL).sort_index()
    assert long_gap_union is not None
    long_gap_union.index = df_use.index

    for s in sensors:
        df_use[s] = df_use[s].interpolate(limit=int(cfg.INTERPOLATE_LIMIT), limit_direction="both")
        df_use[s] = df_use[s].ffill().bfill()

    if cfg.ENABLE_DERIVED_FEATURES:
        windows = _derived_windows(cfg)
        for s in sensors:
            df_use = _build_derived_features(df_use, sensor=s, windows=windows)
    if cfg.ENABLE_CHANGEPOINT_FEATURES:
        for s in sensors:
            df_use = _build_changepoint_features(
                df_use, sensor=s, short_window=cfg.CHANGEPOINT_SHORT_WINDOW,
                long_window=cfg.CHANGEPOINT_LONG_WINDOW, cusum_k=cfg.CHANGEPOINT_CUSUM_K,
            )
    if cfg.ENABLE_THERMAL_ARRAY_SPREAD and cfg.THERMAL_ARRAY_SENSORS:
        df_use = _build_thermal_array_spread(cfg, df_use, source_df, cfg.THERMAL_ARRAY_SENSORS)

    return df_use, long_gap_union


def select_feature_columns(cfg: PipelineConfig, df_use: pd.DataFrame, sensors: List[str]) -> List[str]:
    """Nomes das colunas de feature (sensor bruto + derivadas habilitadas)
    presentes em `df_use` para os `sensors` dados -- mesma logica usada
    tanto pelo AutoML nao-supervisionado quanto pelo classificador
    supervisionado (EXP7/EXP8), para as duas abordagens usarem exatamente
    o mesmo espaco de features e serem comparaveis."""
    feature_cols = list(sensors)
    if cfg.ENABLE_DERIVED_FEATURES:
        windows = _derived_windows(cfg)
        suffixes = ["__delta_1"]
        for w in windows:
            w = max(2, int(w))
            suffixes += [f"__roll_med_{w}", f"__roll_std_{w}", f"__trend_{w}"]
            if w >= TEXTURE_MIN_WINDOW:
                suffixes += [f"__roll_kurt_{w}", f"__roll_skew_{w}", f"__crest_{w}"]
        for s in sensors:
            for suffix in suffixes:
                col = f"{s}{suffix}"
                if col in df_use.columns:
                    feature_cols.append(col)
    if cfg.ENABLE_CHANGEPOINT_FEATURES:
        sw = max(2, int(cfg.CHANGEPOINT_SHORT_WINDOW))
        lw = max(sw + 1, int(cfg.CHANGEPOINT_LONG_WINDOW))
        cp_suffixes = [f"__localz_{sw}_{lw}", f"__cusum_pos_{lw}", f"__cusum_neg_{lw}"]
        for s in sensors:
            for suffix in cp_suffixes:
                col = f"{s}{suffix}"
                if col in df_use.columns:
                    feature_cols.append(col)
    return feature_cols


def build_exclusion_mask(index: pd.DatetimeIndex, alarm_times: pd.Series, minutes: int) -> pd.Series:
    exclude = pd.Series(False, index=index)
    delta = pd.Timedelta(minutes=minutes)
    for t in alarm_times.values:
        t0 = pd.Timestamp(t) - delta
        t1 = pd.Timestamp(t) + delta
        exclude.loc[(exclude.index >= t0) & (exclude.index <= t1)] = True
    return exclude


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


def normalize_train_only(
    cfg: PipelineConfig, df_normal: pd.DataFrame, df_all: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mode = cfg.NORMALIZE_MODE.lower()

    if mode == "zscore":
        center = df_normal.mean(axis=0)
        scale = df_normal.std(axis=0).replace(0, 1.0)
    elif mode == "robust":
        center = df_normal.median(axis=0)
        q1 = df_normal.quantile(0.25)
        q3 = df_normal.quantile(0.75)
        scale = (q3 - q1).replace(0, 1.0)
    else:
        raise ValueError("NORMALIZE_MODE invalido. Use 'zscore' ou 'robust'.")

    df_normal_z = (df_normal - center) / scale
    df_all_z = (df_all - center) / scale

    return df_normal_z, df_all_z, center, scale
