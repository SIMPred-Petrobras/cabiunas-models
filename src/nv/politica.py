"""Política de alarme FIXA aplicada a qualquer escore (para comparar escores sem comparar políticas).

  1. EWMA de meia-vida 1 h SÓ sobre instantes vigiados (o EWMA nunca engole parada nem blackout);
  2. episódio = corrida de EWMA > limiar sobre instantes vigiados contíguos; vale se dura >= 2 h (dur_min);
  3. refratário: depois de um episódio, os que começam em até 72 h do seu fim são descartados.

O LIMIAR não se escolhe com eventos: `calibra_limiar` o acha por bisseção para um ORÇAMENTO de horas de alarme fora das janelas de trip (o mesmo orçamento para todos os escores: comparação a custo igual).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .avalia.metricas import avalia, PASSO_H

DUR_MIN = pd.Timedelta(hours=2)
REFRATARIO = pd.Timedelta(hours=72)


def ewma_vigiado(score: pd.Series, vig: pd.Series, meia_vida: str = "1h") -> pd.Series:
    v = vig.reindex(score.index).fillna(False).to_numpy(bool)
    s = score.where(v)
    e = s.ewm(halflife=pd.Timedelta(meia_vida), times=s.index, ignore_na=True).mean()
    return e.where(v)


def episodios(e: pd.Series, limiar: float, dur_min=DUR_MIN, refratario=REFRATARIO) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    acima = (e > limiar).fillna(False).to_numpy(bool)
    if not acima.any():
        return []
    d = np.diff(np.concatenate(([0], acima.astype(np.int8), [0])))
    ini = np.flatnonzero(d == 1); fim = np.flatnonzero(d == -1) - 1; idx = e.index
    out, bloq = [], None
    for i, j in zip(ini, fim):
        a, b = idx[i], idx[j]
        if (b - a) + pd.Timedelta("2min") < dur_min:
            continue
        if bloq is not None and a <= bloq:
            continue
        out.append((a, b)); bloq = b + refratario
    return out


def alarme_de(eps, indice: pd.DatetimeIndex) -> pd.Series:
    a = np.zeros(len(indice), bool)
    for x, y in eps:
        a[indice.searchsorted(x):indice.searchsorted(y, side="right")] = True
    return pd.Series(a, index=indice)


def calibra_limiar(e: pd.Series, vig: pd.Series, trips, alvo_h_fora: float, inicio=None, n_iter: int = 28) -> float:
    """Bisseção em log do limiar para que as horas de alarme FORA das janelas de trip igualem `alvo_h_fora` (monótono decrescente no limiar)."""
    v = e.dropna()
    lo, hi = float(np.nanquantile(v, 0.50)), float(v.max())
    f = lambda t: avalia(alarme_de(episodios(e, t), e.index), vig, trips, inicio_periodo=inicio).horas_alarme_fora
    for _ in range(n_iter):
        mid = float(np.sqrt(max(lo, 1e-12) * hi))
        if f(mid) > alvo_h_fora:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(max(lo, 1e-12) * hi))
