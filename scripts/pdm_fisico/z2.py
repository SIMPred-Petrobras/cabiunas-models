#!/usr/bin/env python3
"""z robusto com PISO DE ESCALA.

Diagnostico que motiva: a mediana de |z| de vib_351Y em 2024Q1 deu 51,9. Nao e
degradacao -- e o MAD das 400 h de baseline colapsando. Um sensor que passou o
mes parado num patamar recebe escala ~0,05; no mes seguinte ele muda 2 unidades
e o z vai a 40. Num detector que toma o MAXIMO sobre 26 canais, um canal quieto
domina o alarme o tempo inteiro.

O piso e a ideia do portao do Diego movida um nivel abaixo: em vez de vetar o
score quando a grandeza esta fora de faixa fisica, veta-se a ESCALA quando ela
fica menor do que aquele canal costuma ter. Formalmente:

    s_c(M) = max( MAD_c(ultimas 400 h),  phi * MAD_c(todo o passado) )

phi=0 e o comportamento anterior; phi=1 abre mao da adaptacao local e usa so o
espalhamento de longo prazo. E um parametro so, e ele e varrido, nao escolhido.
"""
from __future__ import annotations
import numpy as np, pandas as pd

PAS = pd.Timedelta("2min")


def z_piso(X: pd.DataFrame, quente: pd.Series, falhas: pd.Series,
           phi: float, horas_base=400.0, excl_dias=7.0) -> pd.DataFrame:
    idx = X.index
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n_base = int(horas_base * 30)
    Z = pd.DataFrame(np.nan, index=idx, columns=X.columns, dtype="float32")
    Xv = X.to_numpy().astype("float64")
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        elig = quente & (idx < m0)
        for f in falhas[falhas < m0]:
            elig &= ~((idx >= f - pd.Timedelta(days=excl_dias)) & (idx <= f + pd.Timedelta(days=2)))
        todo = np.flatnonzero(elig.to_numpy())
        if todo.size < n_base // 4:
            continue
        loc = todo[-n_base:]
        sel = np.flatnonzero((idx >= m0) & (idx < m1))
        if sel.size == 0:
            continue
        med = np.nanmedian(Xv[loc], axis=0)
        s_loc = np.nanmedian(np.abs(Xv[loc] - med), axis=0) * 1.4826
        if phi > 0:
            med_g = np.nanmedian(Xv[todo], axis=0)
            s_glob = np.nanmedian(np.abs(Xv[todo] - med_g), axis=0) * 1.4826
            s = np.maximum(s_loc, phi * s_glob)
        else:
            s = s_loc
        s[~np.isfinite(s) | (s <= 0)] = np.nan
        Z[sel[0]:sel[-1] + 1] = np.abs((Xv[sel] - med) / s).astype("float32")
    return Z
