#!/usr/bin/env python3
"""Otimizar para ESPARSIDADE DO VOTO -- a causa raiz.

DIAGNOSTICO (onde_morre.py, 11/09/2026). A banda ja e 3/8 no `voto>=2`, ANTES de
portao, refratario e duracao -- os tres nao mudam nada. Os episodios do voto
comecam 144h, 195h, 67h, 52h e 670h antes dos trips, porque o voto fica aceso
**44,9% do tempo**: com esse duty os episodios se fundem e o inicio esta sempre
no passado distante.

Todos os experimentos anteriores otimizaram CONTAGEM DE DETECCAO com o voto
saturado. Aqui a busca e outra: achar a configuracao de limiares que deixa o voto
ESPARSO (alvo: duty < 10%) e medir o que sobra na banda.

Limiar livre por canal, sem amarrar t/p/sp ao mesmo kb.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy(); tot = float(mask.sum())
CACHE = {}

def canal(c, k, cus=True):
    if (c, k, cus) in CACHE: return CACHE[(c, k, cus)]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    r = deg & mask
    if cus:
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        r = (deg | cu) & mask
    CACHE[(c, k, cus)] = r
    return r

def pos(v, rf, dm):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(v):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=rf)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b-a).total_seconds()/60 + 2 >= dm: fin.loc[a:b] = True
    return fin & sel

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

GR = {"t": [1.7, 2.5, 3.5, 5.0], "p": [1.7, 2.5, 3.5, 5.0],
      "sp": [1.7, 2.5, 3.5, 5.0], "vb": [2.2, 3.5, 5.0, 7.0]}
print("BUSCA POR VOTO ESPARSO -- limiar livre por canal, com e sem CUSUM")
print("=" * 110)
print(f"{'k t/p/sp/vb':>22}{'cusum':>7}{'duty voto':>11}{'banda':>8}{'inicio':>8}"
      f"{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 110)
res = []
for cus in (True, False):
    for kt, kp, ks, kv in itertools.product(GR["t"], GR["p"], GR["sp"], GR["vb"]):
        ON = {"t": canal("t", kt, cus), "p": canal("p", kp, cus),
              "sp": canal("sp", ks, cus), "vb": canal("vb", kv, cus)}
        v = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
        duty = 100*float(v.sum())/tot
        r = mede(pos(v & (ON["sp"] | ON["vb"]), REFRAT_H, DUR_MIN))
        res.append((r[0], -duty, kt, kp, ks, kv, cus, duty, r))
res.sort(reverse=True)
vistos = set()
for b, _, kt, kp, ks, kv, cus, duty, r in res[:40]:
    key = (b, round(duty))
    if key in vistos or r[2] < 7: continue
    vistos.add(key)
    print(f"{f'{kt}/{kp}/{ks}/{kv}':>22}{('sim' if cus else 'nao'):>7}{duty:10.1f}%"
          f"{r[0]:6d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
          f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 110)
print("  publicado: k=1,7/1,7/1,7/2,2 com cusum -> duty 44,9%, banda 3/8, 0,517 FP/mes")
oito = [x for x in res if x[8][2] == 8]
if oito:
    b, _, kt, kp, ks, kv, cus, duty, r = oito[0]
    print(f"\n  MELHOR com det=8/8: k={kt}/{kp}/{ks}/{kv}, cusum={cus}, duty {duty:.1f}%")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
