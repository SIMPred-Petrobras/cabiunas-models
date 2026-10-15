"""Normalização LOCAL do escore (LRN): tira da série os desvios lentos e persistentes do normal com uma referência atrasada.

  z(t) = max(0, (e(t) - q50_ref) / sigma_ref)
  q50_ref, sigma_ref   mediana e dispersão robusta ((q75 - q25) / 1,349) das MEDIANAS HORÁRIAS de e sobre os instantes vigiados em [t - janela - guarda, t - guarda)
  guarda               o atraso entre o fim da referência e o instante (7 d): a degradação que se desenvolve nos últimos dias não entra na referência e não é "explicada" por ela (o princípio do vb do v2)

A referência só usa horas ANTERIORES a t - guarda (mais de 1 hora de folga sobre a hora corrente), então é causal; `auditoria_do_futuro` a confere nos testes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def lrn(e: pd.Series, vig: pd.Series, janela_h: int = 672, guarda_h: int = 168) -> pd.Series:
    v = vig.reindex(e.index).fillna(False)
    h = e.where(v).resample("1h").median()
    mp = max(janela_h // 4, 24)
    q50 = h.rolling(janela_h, min_periods=mp).median().shift(guarda_h)
    q25 = h.rolling(janela_h, min_periods=mp).quantile(0.25).shift(guarda_h)
    q75 = h.rolling(janela_h, min_periods=mp).quantile(0.75).shift(guarda_h)
    sig = ((q75 - q25) / 1.349).clip(lower=1e-6)
    r50 = q50.reindex(e.index, method="ffill"); rs = sig.reindex(e.index, method="ffill")
    z = ((e - r50) / np.maximum(rs, 0.05 * np.abs(r50) + 1e-6)).clip(lower=0)
    return z.where(v)
