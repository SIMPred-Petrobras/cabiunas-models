"""Retreino mensal CAUSAL (o protocolo de produção): cada mês é pontuado por um modelo ajustado só com as últimas `n_treino` amostras ESTÁVEIS anteriores ao dia 1 do mês.

O escore de cada mês é dividido pelo p99 dos escores do PRÓPRIO treino (a convenção do v2: o escore fica em múltiplos do p99 do baseline).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def inicios_de_mes(ini: pd.Timestamp, fim: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(ini.normalize().replace(day=1), fim, freq="MS", tz="UTC"))


def walk_forward(fabrica, g: pd.DataFrame, tags: list[str], estavel: np.ndarray, ini: pd.Timestamp, fim: pd.Timestamp, n_treino: int = 20000, semente: int = 0) -> pd.Series:
    X = g[tags].to_numpy("float64"); idx = g.index; out = np.full(len(g), np.nan)
    meses = inicios_de_mes(ini, fim)
    for k, m0 in enumerate(meses):
        m1 = meses[k + 1] if k + 1 < len(meses) else fim + pd.Timedelta("2min")
        i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        cand = np.flatnonzero(estavel[:i0] & np.isfinite(X[:i0]).all(axis=1))
        if len(cand) < n_treino // 4:
            continue
        tr = X[cand[-n_treino:]]
        mod = fabrica(semente=semente).fit(tr)
        p99 = float(np.nanquantile(mod.score(tr), 0.99))
        out[i0:i1] = mod.score(X[i0:i1]) / max(p99, 1e-12)
    return pd.Series(out, index=idx)
