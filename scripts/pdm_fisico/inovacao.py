#!/usr/bin/env python3
"""IDEIA B -- canal de INOVACAO: trocar teste de ESTADO por teste de TRANSICAO.

DIAGNOSTICO QUE MOTIVA (onde_morre.py, 11/09/2026). O voto combinado fica aceso
44,9% do tempo e por isso os episodios se fundem e os inicios ficam 52 a 670 h no
passado. A banda ja e 3/8 no `voto>=2`, antes de qualquer pos-processamento.

A CAUSA da densidade: o canal e `EWMA(sinal) > limiar`, um teste de NIVEL. Ele
pergunta "a maquina esta num lugar incomum?" -- e ela passa muito tempo em
lugares incomuns e ESTAVEIS. Um teste de INOVACAO pergunta "a maquina esta se
movendo de forma inesperada?", que e naturalmente esparso.

    nivel    : E = EWMA_h(x)                    -> aceso quando x esta alto
    inovacao : I = |x - EWMA_h(x)| / escala     -> aceso quando x SURPREENDE

A escala e a mediana robusta da propria inovacao numa janela longa, causal.
Compara canal a canal antes de montar detector.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import mask, idx, alvo, op
from publica_clearml import SIN, HL

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
RNG = np.random.default_rng(20260911)
n = lambda h: int(pd.Timedelta(hours=h)/pd.Timedelta("2min"))

z = np.load("piso_fisico_cache.npz")
spv = np.abs((z["b_all"] - z["med_sp"])/z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Z = np.abs((z["Xh"] - z["MED"])/z["S"])
vbv = np.full(len(idx), np.nan)
vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
vbv[~np.isfinite(vbv)] = np.nan
CRU = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)

ti = np.asarray(idx.astype("int64"))
lo_us = int(pd.Timedelta(hours=TMAX).value)//1000
hi_us = int(pd.Timedelta(hours=TMIN).value)//1000
eleg = idx[(idx >= idx[0]+JAN) & op.to_numpy() & mask.to_numpy()]
SORT = RNG.choice(np.asarray(eleg.astype("int64")), size=(4000, len(alvo)))
T_OBS = np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo])

def cobre(A, T):
    a = np.searchsorted(ti, T-lo_us, "left"); b = np.searchsorted(ti, T-hi_us, "right")
    return np.array([bool(y > x and A[x:y].any()) for x, y in zip(a, b)])

def valida(A, rot):
    A = np.asarray(A); o = int(cobre(A, T_OBS).sum())
    nul = np.array([cobre(A, SORT[k]).sum() for k in range(1500)])
    p = float((nul >= o).mean())
    duty = 100*float((A & mask.to_numpy()).sum())/float(mask.sum())
    print(f"  {rot:<34} duty {duty:5.2f}% | banda {o}/8 | nulo {nul.mean():.2f}"
          f" | {o/max(nul.mean(),1e-9):5.2f}x | p={p:.4f}" + ("  ***" if p < 0.05 else ""))
    return o, p, duty

print("CANAL DE INOVACAO -- |x - EWMA(x)| normalizado")
print("=" * 100)
INOV = {}
for c in SIN:
    x = CRU[c].where(mask)
    for hl in ("30min", "2h", "8h"):
        e = x.ewm(halflife=pd.Timedelta(hl), times=idx).mean()
        i = (x - e).abs()
        esc = i.rolling(n(400), min_periods=n(100)).median()
        INOV[(c, hl)] = (i/esc.replace(0, np.nan))
    print(f"\n  canal {c}  (nivel atual: aceso "
          f"{100*float((x > x.quantile(0.7)).sum())/float(mask.sum()):.0f}% acima do p70)")
    for hl in ("30min", "2h", "8h"):
        I = INOV[(c, hl)]
        for k in (4, 8, 15):
            valida(((I >= k) & mask).fillna(False).to_numpy(), f"inov hl={hl}, k={k}")
