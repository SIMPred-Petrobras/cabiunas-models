#!/usr/bin/env python3
"""Limiar por calibracao conformal em vez de percentil do proprio baseline.

O detector anterior fixava o corte no percentil do conjunto em que a escala foi
estimada. Isso so vale se o mes seguinte tiver a mesma distribuicao -- e a
medicao mostrou que nao tem: a mediana de |z| em operacao da 1,25 onde deveria
dar 0,67, ou seja o espalhamento real e ~1,9x o das 400 h de baseline. O corte
sai baixo demais e o alarme fica ligado 62% do tempo.

Aqui o corte para o mes M vem do percentil dos scores dos ultimos HORAS_CAL de
operacao JA PONTUADA antes de M -- dado fora da amostra, com a mesma deriva.
E calibracao split-conformal: a taxa de cruzamento passa a valer o que se pediu,
sem supor estacionariedade.
"""
from __future__ import annotations
import numpy as np, pandas as pd

PAS = pd.Timedelta("2min")


def limiar_conformal(Z: pd.DataFrame, quente: pd.Series, falhas: pd.Series,
                     alpha: float, horas_cal=700.0, excl_dias=7.0):
    """Sidak sobre canais + percentil tomado na janela de calibracao."""
    M = Z.shape[1]
    p = (1.0 - alpha) ** (1.0 / M)
    idx = Z.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n_cal = int(horas_cal * 60 / 2)
    razao = pd.Series(np.nan, index=idx, dtype="float32")
    culpado = pd.Series(pd.NA, index=idx, dtype="object")
    Zv = Z.to_numpy()
    cols = np.asarray(Z.columns)
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        elig = quente & (idx < m0) & Z.notna().any(axis=1)
        for f in falhas[falhas < m0]:
            elig &= ~((idx >= f - pd.Timedelta(days=excl_dias)) & (idx <= f + pd.Timedelta(days=2)))
        pos = np.flatnonzero(elig.to_numpy())[-n_cal:]
        if pos.size < n_cal // 4:
            continue
        q = np.nanquantile(Zv[pos], p, axis=0)
        q[~np.isfinite(q) | (q <= 0)] = np.nan
        sel = np.flatnonzero((idx >= m0) & (idx < m1))
        if sel.size == 0:
            continue
        r = Zv[sel] / q
        todo_na = np.isnan(r).all(axis=1)
        rr = np.where(np.isnan(r), -np.inf, r)
        j = np.where(todo_na, 0, np.argmax(rr, axis=1))
        razao.iloc[sel] = np.where(todo_na, np.nan, rr[np.arange(len(sel)), j]).astype("float32")
        culpado.iloc[sel] = np.where(todo_na, None, cols[j])
    return razao, culpado
