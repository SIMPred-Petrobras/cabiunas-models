from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def _window(cfg_value: int | None, fallback: int) -> int:
    return max(1, int(cfg_value if cfg_value is not None else fallback))


def _safe_denominator(series: pd.Series) -> pd.Series:
    denom = series.replace(0, np.nan).ffill().bfill().fillna(1.0)
    return denom.replace(0, 1.0)


def _infer_sampling_seconds(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 1.0
    deltas = index.to_series().diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 1.0
    median_dt = float(deltas.median())
    return median_dt if np.isfinite(median_dt) and median_dt > 0 else 1.0


def _add_rolling_features(out: pd.DataFrame, sensor: str, cfg: PipelineConfig) -> pd.DataFrame:
    window = _window(cfg.ROLLING_WINDOW, cfg.TIME_STEPS)
    s = pd.to_numeric(out[sensor], errors="coerce")
    roll = s.rolling(window=window, min_periods=1)

    rms = np.sqrt((s.pow(2)).rolling(window=window, min_periods=1).mean())
    max_abs = s.abs().rolling(window=window, min_periods=1).max()
    crest_denom = _safe_denominator(rms)

    out[f"{sensor}_roll_rms"] = rms.fillna(0.0)
    out[f"{sensor}_roll_kurtosis"] = roll.kurt().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[f"{sensor}_roll_std"] = roll.std().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[f"{sensor}_roll_crest"] = (max_abs / crest_denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[f"{sensor}_delta_1"] = s.diff(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[f"{sensor}_delta_2"] = s.diff(2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _spectral_features_for_window(values: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    centered = values - float(np.mean(values))
    mag = np.abs(np.fft.rfft(centered))
    energy = mag.pow(2) if hasattr(mag, "pow") else mag**2

    if len(energy) == 0:
        return {
            "fft_energy_low": 0.0,
            "fft_energy_mid": 0.0,
            "fft_energy_high": 0.0,
            "fft_dominant_freq": 0.0,
        }

    chunks = np.array_split(energy, 3)
    dominant = int(np.argmax(mag))
    return {
        "fft_energy_low": float(chunks[0].sum()) if len(chunks) > 0 else 0.0,
        "fft_energy_mid": float(chunks[1].sum()) if len(chunks) > 1 else 0.0,
        "fft_energy_high": float(chunks[2].sum()) if len(chunks) > 2 else 0.0,
        "fft_dominant_freq": float(freqs[dominant]) if dominant < len(freqs) else 0.0,
    }


def _add_spectral_features(out: pd.DataFrame, sensor: str, cfg: PipelineConfig) -> pd.DataFrame:
    if cfg.SENSOR_TYPE.lower() != "vibration":
        return out

    window = _window(cfg.SPECTRAL_WINDOW, cfg.TIME_STEPS)
    stride = _window(cfg.SPECTRAL_STRIDE, cfg.STRIDE)
    s = pd.to_numeric(out[sensor], errors="coerce")
    values = s.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return out

    effective_window = min(window, n)
    sample_seconds = _infer_sampling_seconds(out.index)
    freqs = np.fft.rfftfreq(effective_window, d=sample_seconds)

    rows = []
    row_index = []
    last_start = max(0, n - effective_window)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        stop = start + effective_window
        feats = _spectral_features_for_window(values[start:stop], freqs)
        row_index.append(out.index[stop - 1])
        rows.append(feats)

    spectral = pd.DataFrame(rows, index=pd.Index(row_index))
    spectral = spectral[~spectral.index.duplicated(keep="last")]
    spectral = spectral.reindex(out.index).interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)

    for col in spectral.columns:
        out[f"{sensor}_{col}"] = spectral[col].astype(float)
    return out


def _rolling_linear_slope(s: pd.Series, window: int) -> pd.Series:
    """Slope of linear fit (units/sample) over a sliding window, vectorized via stride tricks."""
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    if window < 2 or n < window:
        return pd.Series(np.zeros(n), index=s.index)

    from numpy.lib.stride_tricks import sliding_window_view
    wins = sliding_window_view(vals, window_shape=window)   # (n-window+1, window)
    x = np.arange(window, dtype=float) - (window - 1) / 2.0  # centred
    x_sq = float(np.dot(x, x))
    if x_sq == 0:
        return pd.Series(np.zeros(n), index=s.index)
    y_mean = wins.mean(axis=1, keepdims=True)
    slopes_valid = np.dot(wins - y_mean, x) / x_sq        # (n-window+1,)
    slopes = np.zeros(n)
    slopes[window - 1:] = slopes_valid
    return pd.Series(slopes, index=s.index)


def _add_trend_features(out: pd.DataFrame, sensor: str, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Adds two drift-detection features:
      {sensor}_trend_slope     — linear fit slope over TREND_SLOPE_WINDOW (captures direction of change).
      {sensor}_dev_baseline    — deviation from a long rolling mean (captures slow level shift).

    These are intentionally NOT present in rolling features because:
    - slope gives the *direction* of change independent of absolute level;
    - dev_baseline exposes multi-hour drift that a 30-min window cannot see.
    """
    s = pd.to_numeric(out[sensor], errors="coerce")
    slope_window = _window(cfg.TREND_SLOPE_WINDOW, cfg.TIME_STEPS)
    out[f"{sensor}_trend_slope"] = _rolling_linear_slope(s, slope_window).fillna(0.0)

    # Long-baseline deviation: value minus rolling mean over BASELINE_HOURS
    sample_dt = _infer_sampling_seconds(out.index)
    pts_per_hour = max(1, int(round(3600.0 / sample_dt)))
    baseline_window = max(slope_window, int(cfg.BASELINE_HOURS * pts_per_hour))
    baseline = s.rolling(window=baseline_window, min_periods=1).mean()
    out[f"{sensor}_dev_baseline"] = (s - baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _add_context_features(out: pd.DataFrame, sensor: str, cfg: PipelineConfig) -> pd.DataFrame:
    context_cols = [c for c in (cfg.CONTEXT_COLS or []) if c in out.columns and c != sensor]
    if not context_cols:
        return out

    sensor_series = pd.to_numeric(out[sensor], errors="coerce")
    sensor_mean = float(sensor_series.mean())
    if not np.isfinite(sensor_mean) or sensor_mean == 0:
        sensor_mean = 1.0

    for ctx in context_cols:
        ctx_series = pd.to_numeric(out[ctx], errors="coerce")
        denom = _safe_denominator(ctx_series)
        ctx_mean = float(ctx_series.mean())
        if not np.isfinite(ctx_mean) or ctx_mean == 0:
            ctx_mean = 1.0

        ctx_normalized_by_mean = (ctx_series / ctx_mean) * sensor_mean
        out[f"{sensor}_ratio_{ctx}"] = (sensor_series / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"{sensor}_residual_{ctx}"] = (
            sensor_series - ctx_normalized_by_mean
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def generate_features(
    df: pd.DataFrame,
    sensor: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    if sensor not in df.columns:
        raise ValueError(f"Sensor '{sensor}' nao encontrado para gerar features.")

    out = df.copy()

    if cfg.ENABLE_ROLLING_FEATURES:
        out = _add_rolling_features(out, sensor, cfg)

    if cfg.ENABLE_TREND_FEATURES:
        out = _add_trend_features(out, sensor, cfg)

    if cfg.ENABLE_SPECTRAL_FEATURES:
        out = _add_spectral_features(out, sensor, cfg)

    if cfg.ENABLE_CONTEXT_FEATURES:
        out = _add_context_features(out, sensor, cfg)

    if out.isna().any().any():
        missing_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Feature engineering gerou NaN nas colunas: {missing_cols}")

    return out
