"""Contexto causal de regime para os modelos condicionais.

  tau_horas     horas desde a última partida (ligada suave: lacunas < 10 min fundidas); NaN fora da operação
  nivel_lento   mediana das últimas 7 d de amostras ESTÁVEIS, ATRASADA em 3 d (a janela é (t-10 d, t-3 d]); o atraso impede que a degradação em
                desenvolvimento nos últimos dias entre no contexto e seja "explicada" pelo modelo (o mesmo princípio da guarda do vb)
Ambas passam pela auditoria do futuro (`nv.causal.auditoria_do_futuro`).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def ligada_suave(g: pd.DataFrame) -> np.ndarray:
    lig = (g["RUNNING_A"] > 0.5).fillna(False).to_numpy(bool); out = lig.copy(); n = len(lig); i = 0
    while i < n:
        if not lig[i]:
            j = i
            while j < n and not lig[j]:
                j += 1
            if 0 < i and j < n and (j - i) < 5:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def tau_horas(g: pd.DataFrame) -> pd.Series:
    lig = ligada_suave(g); n = len(g); tau = np.full(n, np.nan); inicio = -1
    for i in range(n):
        if lig[i] and (i == 0 or not lig[i - 1]):
            inicio = i
        if lig[i] and inicio >= 0:
            tau[i] = (i - inicio) / 30.0
    return pd.Series(tau, index=g.index)


def features_tau(tau: pd.Series) -> np.ndarray:
    t = tau.fillna(72.0).clip(upper=72.0).to_numpy()
    return np.column_stack([t / 72.0, (t < 6).astype(float), (t < 24).astype(float)])


def nivel_lento(g: pd.DataFrame, estavel: np.ndarray, tags: list[str], janela_h: int = 168, atraso_h: int = 72) -> pd.DataFrame:
    X = g[tags].astype("float64").where(pd.Series(estavel, index=g.index), np.nan)
    h = X.resample("1h").median()
    r = h.rolling(janela_h, min_periods=min(24, janela_h)).median().shift(atraso_h)
    return r.reindex(g.index, method="ffill")
