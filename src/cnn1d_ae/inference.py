"""Inferência de produção a partir do bundle salvo no treino.

O bundle (`best_model/inference_bundle.json`) carrega o scaler (center/scale),
clip bounds, threshold e parâmetros fixados no TREINO. Aplicar essas estatísticas
de treino — em vez de refitar na nova distribuição — é o que mantém o threshold
calibrado válido em dados novos.

Uso típico:
    from tensorflow import keras
    from src.cnn1d_ae.inference import load_bundle, score_dataframe
    bundle = load_bundle("…/best_model/inference_bundle.json")
    model  = keras.models.load_model("…/best_model/model.keras")
    scores = score_dataframe(model, bundle, df_novo)   # df indexado por tempo
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .sequences import make_sequences
from .scoring import reconstruction_mae_per_seq


def load_bundle(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transform_features(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Replica clip(bounds de treino) → (x − center)/scale com as estatísticas do bundle."""
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para inferência: {missing}")
    out = df[cols].apply(pd.to_numeric, errors="coerce").copy()

    clip_bounds = bundle.get("clip_bounds") or {}
    for c in cols:
        if c in clip_bounds:
            low, high = clip_bounds[c]
            out[c] = out[c].clip(lower=low, upper=high)

    center = bundle.get("center", {})
    scale = bundle.get("scale", {})
    for c in cols:
        sc = float(scale.get(c, 1.0))
        if sc == 0:
            sc = 1.0
        out[c] = (out[c] - float(center.get(c, 0.0))) / sc

    return out.to_numpy(dtype=np.float32)


def score_dataframe(model, bundle: dict, df_sensor: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """Reproduz o scoring por sequência em dados novos.

    df_sensor: DataFrame indexado por tempo com `feature_columns` (e, opcionalmente,
    `running_col` para a máscara de operação). Retorna um DataFrame por sequência
    com seq_end_time, mae_seq, is_anom_seq e operational_state (se aplicável).
    """
    time_steps = int(bundle["time_steps"])
    stride = max(1, int(bundle["stride"]))
    values = transform_features(df_sensor, bundle)
    x = make_sequences(values, time_steps, stride)
    mae = reconstruction_mae_per_seq(model, x, batch_size)
    anom = mae > float(bundle["threshold"])

    idx = df_sensor.index
    end_pos = (time_steps - 1) + np.arange(len(mae)) * stride
    end_pos = np.clip(end_pos, 0, len(idx) - 1)

    out = pd.DataFrame(
        {
            "seq_end_time": idx[end_pos],
            "mae_seq": mae,
            "is_anom_seq": anom.astype(int),
        }
    )

    running_col = bundle.get("running_col")
    if running_col and running_col in df_sensor.columns:
        rthr = float(bundle.get("running_threshold", 0.5))
        run = pd.to_numeric(df_sensor[running_col], errors="coerce").to_numpy()[end_pos]
        on = run > rthr
        out["operational_state"] = np.where(on, "on", "off")
        out.loc[~on, "is_anom_seq"] = 0  # suprime anomalia fora de operação

    return out


def score_production(model, bundle: dict, df_sensor: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """Scoring de produção usando o bloco `production_alerting` do bundle finalizado
    (finalize_bundle.py): erro de reconstrução → EWMA(half_life) → comparação com o
    threshold ABSOLUTO calibrado → máscara de operação → debounce. É o ponto de
    operação que minimiza FP sem perder recall, pronto para streaming.
    """
    pa = bundle.get("production_alerting")
    if pa is None:
        raise ValueError("bundle sem 'production_alerting' — rode scripts/finalize_bundle.py antes")

    base = score_dataframe(model, bundle, df_sensor, batch_size)
    s = pd.Series(base["mae_seq"].to_numpy(), index=pd.DatetimeIndex(base["seq_end_time"]))

    dt = pd.Series(s.index).diff().dt.total_seconds().median()
    dt = dt if (dt and dt > 0) else 300.0
    hl_pts = max(1, int(round(float(pa["half_life_hours"]) * 3600.0 / dt)))
    ewma = s.ewm(halflife=hl_pts).mean()

    base = base.copy()
    base["health_ewma"] = ewma.to_numpy()
    alert = (ewma.to_numpy() >= float(pa["ewma_abs_threshold"])).astype(int)
    if "operational_state" in base.columns:
        alert = np.where(base["operational_state"].to_numpy() == "off", 0, alert)

    debounce_h = float(pa.get("debounce_hours", 0.0) or 0.0)
    if debounce_h > 0:
        # exige alerta sustentado por >= debounce_h (runs contíguos mais curtos são zerados)
        need = max(1, int(round(debounce_h * 3600.0 / dt)))
        a = alert.copy()
        i = 0
        while i < len(a):
            if a[i]:
                j = i
                while j + 1 < len(a) and a[j + 1]:
                    j += 1
                if (j - i + 1) < need:
                    a[i:j + 1] = 0
                i = j + 1
            else:
                i += 1
        alert = a
    base["alert"] = alert
    return base
