#!/usr/bin/env python3
"""Detector construido com canais de INOVACAO -- ataca a causa raiz.

Os canais de inovacao sao 10 a 30x mais esparsos que os de nivel com cobertura
comparavel (vb: 1,62% contra 52,0% de duty, banda 6/8 contra 7/8, razao 3,26x
contra 1,48x). Se a densidade do voto era a causa dos inicios distantes, um voto
montado sobre eles deve nascer perto dos eventos.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h
from inovacao import CRU, INOV

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0

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
        c = [a for a, _ in eps if t-JAN <= a <= t-pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t-max(c)).total_seconds()/3600)
        if any(t-JAN <= a <= t for a, _ in eps): ini += 1
    jw = [(t-JAN, t) for t in alvo]; fp = h = 0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw): continue
        if len(paradas[(paradas.ini >= a) & (paradas.ini <= b+JAN)]): continue
        fp += 1; h += (b-a).total_seconds()/3600
    return banda, ini, AV.avalia(al, alvo, mask)["det"], fp/meses, h/meses, \
           (np.mean(leads) if leads else np.nan), len(eps)

print(f"{'configuracao':>46} | {'duty voto':>10}{'banda':>7}{'inicio':>8}{'det':>7}"
      f"{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 116)
print(f"{'publicado (canais de nivel)':>46} | {44.9:9.1f}%{3:5d}/8{4:7d}/8{8:6d}/8"
      f"{0.517:10.3f}{7.1:9.1f}{12.5:8.1f}h")
print(f"{'melhor atual (dois niveis)':>46} | {'--':>10}{5:5d}/8{6:7d}/8{8:6d}/8"
      f"{0.344:10.3f}{6.6:9.1f}{19.7:8.1f}h")
print("-" * 116)
melhor = None
HLS = ("30min", "2h", "8h")
for hl, k, nv, rf in itertools.product(HLS, (6, 8, 10, 15), (1, 2), (48, 72)):
    ON = {c: ((INOV[(c, hl)] >= k) & mask).fillna(False) for c in SIN}
    ns = sum(ON[c].astype(int) for c in SIN)
    for porta in ("sem", "sp|vb"):
        v = pd.Series(ns >= nv, index=idx) & mask
        if porta == "sp|vb": v = v & (ON["sp"] | ON["vb"])
        duty = 100*float(v.sum())/float(mask.sum())
        r = mede(pos(v, rf, DUR_MIN))
        if r[2] >= 7 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
            melhor = ((r[0], -r[3], -r[4]), hl, k, nv, rf, porta, duty, r)
        if r[0] >= 5 and r[2] >= 7:
            print(f"{f'inov hl={hl} k={k} voto>={nv} refr={rf}h portao={porta}':>46} | "
                  f"{duty:9.2f}%{r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
                  f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 116)
if melhor:
    _, hl, k, nv, rf, porta, duty, r = melhor
    print(f"  MELHOR: inovacao hl={hl}, k={k}, voto>={nv}, refrat={rf}h, portao={porta}")
    print(f"     duty do voto {duty:.2f}% -> banda {r[0]}/8, inicio {r[1]}/8, det {r[2]}/8, "
          f"{r[3]:.3f} FP/mes, {r[4]:.1f} h/mes, lead {r[5]:.1f}h, {r[6]} episodios")
