from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from .config import PipelineConfig
from .feature_engineering import generate_features


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

    before_dupes = int(df_use.duplicated(subset=[cfg.TIME_COL]).sum())
    if before_dupes:
        print(f"[DATA-CLEAN] sensor={sensor}: removendo {before_dupes} timestamps duplicados antes das sequencias.")
    df_use = df_use.sort_values(cfg.TIME_COL).drop_duplicates(subset=[cfg.TIME_COL], keep="first")

    for col in selected_cols:
        if col != cfg.TIME_COL:
            df_use[col] = pd.to_numeric(df_use[col], errors="coerce")

    long_gap_raw = _long_gap_mask(df_use[sensor], cfg.INTERPOLATE_LIMIT)

    df_use = df_use.set_index(cfg.TIME_COL).sort_index()
    long_gap_raw.index = df_use.index

    df_use[sensor] = df_use[sensor].interpolate(limit=int(cfg.INTERPOLATE_LIMIT), limit_direction="both")
    n_missing_after_interp = int(df_use[sensor].isna().sum())
    if n_missing_after_interp:
        print(
            f"[DATA-CLEAN] sensor={sensor}: {n_missing_after_interp} NaNs restantes apos "
            f"interpolacao limitada (INTERPOLATE_LIMIT={cfg.INTERPOLATE_LIMIT}). "
            "Aplicando fallback explicito de interpolacao temporal; gaps longos seguem "
            "marcados para exclusao do treino via long_gap_mask."
        )
        df_use[sensor] = df_use[sensor].interpolate(method="time", limit_direction="both")

    n_missing_after_fallback = int(df_use[sensor].isna().sum())
    if n_missing_after_fallback:
        print(
            f"[DATA-CLEAN] sensor={sensor}: {n_missing_after_fallback} NaNs ainda restantes "
            "apos interpolacao temporal; aplicando preenchimento de borda como ultimo recurso."
        )
        df_use[sensor] = df_use[sensor].ffill().bfill()

    assert not df_use[sensor].isna().any(), "Falha interna: NaN remanescente apos interpolacao."

    return df_use, long_gap_raw


def apply_feature_engineering(
    df_normal: pd.DataFrame,
    df_all: pd.DataFrame,
    sensor: str,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (cfg.ENABLE_ROLLING_FEATURES or cfg.ENABLE_SPECTRAL_FEATURES or cfg.ENABLE_CONTEXT_FEATURES):
        return df_normal, df_all

    return (
        generate_features(df_normal, sensor, cfg),
        generate_features(df_all, sensor, cfg),
    )


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
