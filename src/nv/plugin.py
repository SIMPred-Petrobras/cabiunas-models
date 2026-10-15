"""Plug-in de canais: troca um dos quatro sinais do v2 por um sinal candidato e roda a DECISÃO DE PRODUÇÃO (`ci.prever`) como caixa-preta.

Por quê: a decisão do v2 (CUSUM com memória, níveis A/B, portão sp|vb, escalada, refratário) vale muitos eventos (EXP 05b: 4 a 9 FP episódios). Um candidato de SINAL só mostra o que vale se for medido DENTRO dessa decisão, não só sob uma política simples.

Escala: a decisão tem limiares em unidades do sinal do v2. O candidato é levado para a MESMA escala por MAPEAMENTO DE QUANTIS monótono sobre os instantes vigiados (preserva a ordem; iguala a distribuição marginal no normal), sem usar rótulos.
LIMITE: o mapa é global no período (usa a marginal inteira, inclusive a futura): é um ajuste de ESCALA sem rótulo, e fica declarado em todo resultado que o usa.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .controle_sinais import _ci

CANAIS = ("t", "p", "sp", "vb")


def mapeia_quantis(cand: pd.Series, ref: pd.Series, vig: pd.Series, n: int = 4001) -> pd.Series:
    """Função monótona F tal que F(cand) tem a distribuição de `ref` sobre os instantes vigiados em que ambos existem."""
    v = vig.reindex(cand.index).fillna(False).to_numpy(bool)
    a = cand.to_numpy(float); b = ref.reindex(cand.index).to_numpy(float)
    ok = v & np.isfinite(a) & np.isfinite(b)
    q = np.linspace(0, 1, n)
    xa = np.quantile(a[ok], q); xb = np.quantile(b[ok], q)
    xa = np.maximum.accumulate(xa + np.arange(n) * 1e-12)                      # estritamente crescente para o interp
    out = np.full(len(a), np.nan); f = np.isfinite(a)
    out[f] = np.interp(a[f], xa, xb)
    return pd.Series(out, index=cand.index).where(vig.reindex(cand.index).fillna(False))


def decide_v2(proc: pd.DataFrame, det: dict) -> pd.DataFrame:
    """A decisão de produção sobre um `proc` (colunas t, p, sp, vb, mask, reset)."""
    ci = _ci()
    return ci.prever({"detector": det}, proc)


def substitui(proc: pd.DataFrame, canal: str, novo: pd.Series) -> pd.DataFrame:
    out = proc.copy()
    out[canal] = novo.reindex(out.index)
    return out
