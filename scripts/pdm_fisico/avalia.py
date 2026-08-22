#!/usr/bin/env python3
"""Regua unica, identica a do relatorio do Francisco: janela de deteccao de
48 h, episodios agrupados em 2 h, falso positivo por MES DE OPERACAO (nao de
calendario). Acrescenta o que faltava la: teste de permutacao."""
from __future__ import annotations
import numpy as np, pandas as pd

PAS = pd.Timedelta("2min")
JANELA_H = 48.0
GAP_EP_H = 2.0


def episodios(alerta: pd.Series, gap_h=GAP_EP_H) -> list[tuple]:
    a = alerta.fillna(False).to_numpy()
    idx = alerta.index
    if not a.any():
        return []
    corte = np.flatnonzero(a[1:] != a[:-1]) + 1
    ini = np.concatenate(([0], corte)); fim = np.concatenate((corte, [len(a)]))
    br = [(idx[i], idx[j - 1]) for i, j in zip(ini, fim) if a[i]]
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h):
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def sustenta(acima: pd.Series, minutos: float) -> pd.Series:
    """So conta como alerta o que ficou acima do limite por 'minutos' seguidos.
    E o filtro que separa ruido de instrumento (nao persiste) de degradacao."""
    n = max(1, int(minutos / 2))
    return acima.fillna(False).rolling(n, min_periods=n).min().fillna(0).astype(bool)


def avalia(alerta: pd.Series, eventos: pd.Series, quente: pd.Series,
           janela_h=JANELA_H) -> dict:
    eps = episodios(alerta)
    horas_op = float(quente.sum()) * 2 / 60.0
    meses_op = horas_op / 730.0
    jan = [(t - pd.Timedelta(hours=janela_h), t) for t in eventos]

    det, leads = [], []
    for t0, t1 in jan:
        dentro = alerta.loc[(alerta.index >= t0) & (alerta.index < t1)]
        dentro = dentro[dentro.fillna(False)]
        if len(dentro):
            det.append(t1)
            leads.append((t1 - dentro.index[0]).total_seconds() / 3600.0)

    fp, h_fp = 0, 0.0
    for a, b in eps:
        if not any((a <= t1) and (b >= t0) for t0, t1 in jan):
            fp += 1
            h_fp += (b - a).total_seconds() / 3600.0 + 2 / 60.0
    return dict(det=len(det), n_ev=len(eventos), episodios=len(eps), fp=fp,
                fp_mes=fp / max(meses_op, 1e-9), h_fp_mes=h_fp / max(meses_op, 1e-9),
                lead_med=float(np.mean(leads)) if leads else np.nan,
                lead_min=float(np.min(leads)) if leads else np.nan,
                duty=float(alerta.fillna(False).mean()), horas_op=horas_op,
                detectados=[t.strftime("%Y-%m-%d") for t in det])


def permuta(alerta: pd.Series, quente: pd.Series, obs: int, n_ev: int, n=20000,
            janela_h=JANELA_H, seed=0) -> dict:
    """Nulo: n_ev instantes sorteados entre os instantes de operacao quente.
    Responde 'quantas falhas um detector com esta cobertura acerta por acaso'."""
    rng = np.random.default_rng(seed)
    a = alerta.fillna(False).to_numpy()
    elig = np.flatnonzero(quente.to_numpy())
    if elig.size == 0 or n_ev == 0:
        return dict(nulo=np.nan, p=np.nan)
    w = int(janela_h * 60 / 2)
    cs = np.concatenate(([0], np.cumsum(a)))
    lo = np.maximum(0, elig - w)
    cov = (cs[elig] - cs[lo]) > 0                 # houve alerta em [t-48h, t)?
    draws = rng.random((n, n_ev)) < cov.mean()
    tot = draws.sum(axis=1)
    return dict(nulo=float(tot.mean()), p=float((tot >= obs).mean()),
                cobertura=float(cov.mean()))
