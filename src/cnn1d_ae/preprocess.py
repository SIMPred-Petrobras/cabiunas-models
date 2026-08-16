from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from .config import PipelineConfig


def _build_derived_features(df: pd.DataFrame, sensor: str, windows: List[int]) -> pd.DataFrame:
    """Features derivadas em multiplas escalas de tempo (uma passada por
    janela em `windows`), alem do delta instantaneo (1 passo, independente
    de janela). Para cada janela w: media/desvio movel (nivel/variabilidade
    recente) e `trend_w` = valor atual menos o valor de w passos atras
    (tendencia/inclinacao ao longo daquela janela especifica) -- pensado
    para capturar precursores lentos que uma unica janela curta nao pega.
    Ver docs/analise_automl_exp7_planejamento.md."""
    out = df.copy()
    out[f"{sensor}__delta_1"] = out[sensor].diff().fillna(0.0)
    for window in windows:
        w = max(2, int(window))
        out[f"{sensor}__roll_med_{w}"] = out[sensor].rolling(w, min_periods=1).median()
        out[f"{sensor}__roll_std_{w}"] = out[sensor].rolling(w, min_periods=1).std().fillna(0.0)
        out[f"{sensor}__trend_{w}"] = (out[sensor] - out[sensor].shift(w)).fillna(0.0)
    return out


def _derived_windows(cfg: PipelineConfig) -> List[int]:
    return list(cfg.DERIVED_ROLLING_WINDOWS) if cfg.DERIVED_ROLLING_WINDOWS else [cfg.DERIVED_ROLLING_WINDOW]


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

    return df_use, long_gap_union


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
