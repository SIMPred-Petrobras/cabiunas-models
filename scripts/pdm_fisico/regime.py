#!/usr/bin/env python3
"""Residuo condicionado ao regime.

O modelo do baseline nao e mais 'a mediana das ultimas 400 h' e sim
'o valor esperado deste canal NESTE ponto de operacao', ajustado nas ultimas
400 h. Quando a maquina muda de ponto no mes seguinte, a mediana erra e a
regressao extrapola -- e essa a diferenca entre um z inflado 3x e um z util.

A base de regime e deliberadamente do LADO DE PROCESSO. T5 fica de fora dela
porque temperatura de exaustao a carga constante e o indicador de saude
classico da turbina a gas: se entrasse como regressor, a degradacao termica
seria absorvida como se fosse regime.
"""
from __future__ import annotations
import numpy as np, pandas as pd

BASES = {
    "nenhuma": [],
    "P":       ["954005_624_PI_0339"],
    "P_oleo":  ["954005_624_PI_0339", "954005_624_TI_0325"],
    "P_oleo_T5": ["954005_624_PI_0339", "954005_624_TI_0325", "T5_AVG_A"],
}


def desenho(g: pd.DataFrame, base: list[str], quad=True) -> pd.DataFrame:
    if not base:
        return pd.DataFrame(index=g.index)
    L = g[base].astype("float64")
    L.columns = [f"L{i}" for i in range(len(base))]
    if quad:
        for i, c in enumerate(list(L.columns)):
            L[f"{c}q"] = L[c] ** 2
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                L[f"L{i}x{j}"] = L[f"L{i}"] * L[f"L{j}"]
    return L


def _ajusta(y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """OLS com uma repescagem: reajusta sem os pontos a mais de 3 MAD, para
    que um transiente dentro do baseline nao puxe a reta."""
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    s = np.median(np.abs(r - np.median(r))) * 1.4826
    if s > 0:
        m = np.abs(r - np.median(r)) <= 3 * s
        if m.sum() > A.shape[1] * 20:
            b, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
    return b


def z_condicionado(X: pd.DataFrame, L: pd.DataFrame, quente: pd.Series,
                   falhas: pd.Series, horas_base=400.0, excl_dias=7.0,
                   passo="2min") -> pd.DataFrame:
    """z robusto do residuo, com modelo e escala refeitos a cada mes usando
    somente o passado."""
    idx = X.index
    PAS = pd.Timedelta(passo)
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n_base = int(horas_base * 60 / (PAS / pd.Timedelta("1min")))
    Z = pd.DataFrame(np.nan, index=idx, columns=X.columns, dtype="float32")
    tem_L = L.shape[1] > 0

    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        elig = quente & (idx < m0)
        for f in falhas[falhas < m0]:
            elig &= ~((idx >= f - pd.Timedelta(days=excl_dias)) & (idx <= f + pd.Timedelta(days=2)))
        pos = np.flatnonzero(elig.to_numpy())[-n_base:]
        if pos.size < n_base // 4:
            continue
        sel = np.flatnonzero((idx >= m0) & (idx < m1))
        if sel.size == 0:
            continue

        if tem_L:
            Lb = L.to_numpy()[pos]; Ls = L.to_numpy()[sel]
            okb = np.isfinite(Lb).all(axis=1)
        else:
            Lb = Ls = None; okb = np.ones(pos.size, bool)

        for c in X.columns:
            yb = X[c].to_numpy()[pos].astype("float64")
            ys = X[c].to_numpy()[sel].astype("float64")
            mb = okb & np.isfinite(yb)
            if mb.sum() < 200:
                continue
            if tem_L:
                A = np.c_[np.ones(mb.sum()), Lb[mb]]
                b = _ajusta(yb[mb], A)
                rb = yb[mb] - A @ b
                As = np.c_[np.ones(len(ys)), Ls]
                rs = ys - As @ b
            else:
                rb = yb[mb] - np.median(yb[mb])
                rs = ys - np.median(yb[mb])
            s = np.median(np.abs(rb - np.median(rb))) * 1.4826
            if not np.isfinite(s) or s <= 0:
                continue
            Z.iloc[sel, Z.columns.get_loc(c)] = np.abs(rs / s).astype("float32")
    return Z
