"""Consenso de vizinhança (H-CONSENSO, EXP 25c): a maioria de configurações vizinhas dos k de nível do v2 como alarme.

Especificação FIXADA em 15/10/2026, antes de qualquer dado novo: 60 configurações cujos oito k (nível A e B de t, p, sp, vb, nessa ordem) são multiplicados por fatores independentes uniformes em [0,9; 1,1] (`default_rng(20261016)`, na ordem de configuração e, dentro dela, (a,t), (a,p), (a,sp), (a,vb), (b,t), (b,p), (b,sp), (b,vb)); o alarme de consenso é a fração de configurações em alarme >= q (0,7). Não alterar os números sem registrar uma nova hipótese.
"""
from __future__ import annotations
import copy
from typing import Callable
import numpy as np
import pandas as pd

CANAIS = ("t", "p", "sp", "vb")
CHAVES = [(nv, c) for nv in ("a", "b") for c in CANAIS]
SEMENTE = 20261016
N_CONFIG = 60
AMPLITUDE = 0.10
QUORUM = 0.7


def sorteio(n: int = N_CONFIG, amp: float = AMPLITUDE, semente: int = SEMENTE) -> list[dict]:
    rng = np.random.default_rng(semente)
    return [{k: float(rng.uniform(1 - amp, 1 + amp)) for k in CHAVES} for _ in range(n)]


def configura(det: dict, mults: dict) -> dict:
    d = copy.deepcopy(det)
    for (nv, c), f in mults.items():
        d["k_nivel_" + nv][c] = d["k_nivel_" + nv][c] * f
    return d


def consenso(proc: pd.DataFrame, det: dict, decide: Callable[[pd.DataFrame, dict], pd.DataFrame], q: float = QUORUM, n: int = N_CONFIG) -> pd.Series:
    """Série booleana de consenso. `decide(proc, det)` devolve o DataFrame de `prever` (coluna `is_anomaly`)."""
    votos = np.mean([decide(proc, configura(det, m))["is_anomaly"].astype(bool).to_numpy() for m in sorteio(n)], axis=0)
    return pd.Series(votos >= q, index=proc.index)
