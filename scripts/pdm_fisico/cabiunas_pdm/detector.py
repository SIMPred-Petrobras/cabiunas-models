"""Constantes e auxiliares do detector -- copia fiel de
src/cabiunas_pdm/detector.py (branch origin/feat/pdm-deteccao-4sinais)."""
from __future__ import annotations

import pandas as pd

SUSTAIN = 15          # 15 x 2 min = 30 min
THR_FAM = 2.0
THR_SPREAD = 3.0
BLACKOUT = "6h"
FIT_POINTS = 20_000   # ~28 dias estaveis


def _spread_mancal(X: pd.DataFrame) -> pd.Series:
    """Divergencia do mancal alvo contra a mediana dos tres irmaos.

    Devolve o spread COM SINAL -- o valor absoluto e aplicado depois, sobre o
    z-score, nao sobre o spread. Trocar a ordem muda o resultado."""
    irm = X[["954005_624_TI_0301", "954005_624_TI_0303",
             "954005_624_TI_0307"]].median(axis=1)
    return X["954005_624_TI_0305"] - irm


def _sustained(s: pd.Series, thr: float) -> pd.Series:
    return ((s > thr).astype(int)
            .rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
