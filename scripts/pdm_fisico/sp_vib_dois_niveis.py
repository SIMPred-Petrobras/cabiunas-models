#!/usr/bin/env python3
"""`sp_vib` dentro da MELHOR maquina (dois niveis por canal), nao na publicada."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import paradas_reais_2h
from sp_vibracao import constroi

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()
FRAC, IDADE, ABS, REFRAT, DUR_ESC = 0.03, 96, 20, 72, 60
HI, KVHI = 1.7, 2.2
LO = {"t": 1.1, "p": 0.7, "sp": 0.9, "vb": 1.8}
SV = constroi(2400)
CACHE = {}

def canal(c, k):
    if (c, k) in CACHE: return CACHE[(c, k)]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    CACHE[(c, k)] = (deg | cu) & mask
    return CACHE[(c, k)]

A0 = {c: canal(c, LO[c]) for c in SIN}
B0 = {c: canal(c, HI if c != "vb" else KVHI) for c in SIN}
KH = {"t": HI, "p": HI, "sp": HI, "vb": KVHI}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()

def mede(al):
    eps = AV.episodios(al); banda, leads, ini = 0, [], 0
    for t in alvo:
        c = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if any(t - JAN <= a <= t for a, _ in eps): ini += 1
    jw = [(t - JAN, t) for t in alvo]
    fp = h = 0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw): continue
        if len(paradas[(paradas.ini >= a) & (paradas.ini <= b + JAN)]): continue
        fp += 1; h += (b-a).total_seconds()/3600
    return banda, ini, AV.avalia(al, alvo, mask)["det"], fp/meses, h/meses, \
           (np.mean(leads) if leads else np.nan)

def roda(A, B, portaB):
    vA = pd.Series(sum(A[c].astype(int) for c in A) >= 3, index=idx) & mask
    vB = pd.Series(sum(B[c].astype(int) for c in B) >= 2, index=idx) & mask & portaB
    vv = quebra_idade(corta_rearma((vA | vB).to_numpy(), F, FRAC), F, ABS, IDADE)
    return mede(pos_dur_esc(pd.Series(vv, index=idx), REFRAT, F, ABS, IDADE, DUR_ESC))

print(f"{'configuracao':>46} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 110)
r0 = roda(A0, B0, B0["sp"] | B0["vb"])
print(f"{'dois niveis por canal (melhor medido)':>46} | {r0[0]:5d}/8{r0[1]:7d}/8{r0[2]:6d}/8"
      f"{r0[3]:10.3f}{r0[4]:9.1f}{r0[5]:8.1f}h")
print("-" * 110)
melhor = None
for kA, kB, conj in itertools.product((4, 5, 6), (5, 6, 8),
                                      ("5o canal", "substitui sp", "substitui sp e p")):
    svA = ((SV >= kA) & mask).fillna(False); svB = ((SV >= kB) & mask).fillna(False)
    if conj == "5o canal":
        A = {**A0, "sv": svA}; B = {**B0, "sv": svB}; pb = B0["sp"] | B0["vb"] | svB
    elif conj == "substitui sp":
        A = {"t": A0["t"], "p": A0["p"], "sv": svA, "vb": A0["vb"]}
        B = {"t": B0["t"], "p": B0["p"], "sv": svB, "vb": B0["vb"]}; pb = svB | B0["vb"]
    else:
        A = {"t": A0["t"], "sv": svA, "vb": A0["vb"]}
        B = {"t": B0["t"], "sv": svB, "vb": B0["vb"]}; pb = svB | B0["vb"]
    r = roda(A, B, pb)
    if r[2] == 8 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
        melhor = ((r[0], -r[3], -r[4]), kA, kB, conj, r)
    if r[0] >= r0[0] and r[2] >= 7:
        print(f"{f'sp_vib A>={kA} B>={kB}, {conj}':>46} | {r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8"
              f"{r[3]:10.3f}{r[4]:9.1f}{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 110)
if melhor:
    _, kA, kB, conj, r = melhor
    print(f"  MELHOR com det=8/8: sp_vib A>={kA} B>={kB}, {conj}")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
