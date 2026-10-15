"""Retreino mensal causal do AE com HISTÓRICO EXPANDIDO e warm-start (o v2 usa só as últimas 20.000 amostras estáveis)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .modelos.ae_torch import ModeloAE
from .walk import inicios_de_mes


def walk_ae(g: pd.DataFrame, tags: list[str], estavel: np.ndarray, C: np.ndarray, ini: pd.Timestamp, fim: pd.Timestamp, semente: int = 0,
            n_max: int = 250000, epocas_inicial: int = 30, epocas_mes: int = 6) -> pd.Series:
    X = g[tags].to_numpy("float64"); idx = g.index; out = np.full(len(g), np.nan); meses = inicios_de_mes(ini, fim); anterior = None
    for k, m0 in enumerate(meses):
        m1 = meses[k + 1] if k + 1 < len(meses) else fim + pd.Timedelta("2min")
        i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        cand = np.flatnonzero(estavel[:i0] & np.isfinite(X[:i0]).all(axis=1) & np.isfinite(C[:i0]).all(axis=1))
        if len(cand) < 5000:
            continue
        tr = cand[-n_max:]
        mod = ModeloAE(len(tags), C.shape[1], semente=semente + 1000 * k).fit(X[tr], C[tr], epocas_inicial if anterior is None else epocas_mes, anterior)
        out[i0:i1] = mod.score(X[i0:i1], C[i0:i1]); anterior = mod
    return pd.Series(out, index=idx)
