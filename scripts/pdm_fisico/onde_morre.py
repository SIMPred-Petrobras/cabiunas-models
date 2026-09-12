#!/usr/bin/env python3
"""Onde exatamente o detector perde, se o sinal e o voto estao la?

Medido: em 8/8 eventos ha 24 a 44 h com >=2 canais simultaneos na banda. Entao a
corroboracao existe. Rastreia o voto por cada etapa do pos-processamento e
mostra em qual delas cada evento e perdido.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)

et = {}
et["1. voto>=2"] = pd.Series(ns >= 2, index=idx) & mask
et["2. + portao sp|vb"] = et["1. voto>=2"] & (ON["sp"] | ON["vb"])
al = pd.Series(False, index=idx); bloq = None
for a, b in AV.episodios(et["2. + portao sp|vb"]):
    if bloq is not None and a <= bloq: continue
    al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
et["3. + refratario 48h"] = al
fin = pd.Series(False, index=idx)
for a, b in AV.episodios(al):
    if (b-a).total_seconds()/60 + 2 >= DUR_MIN: fin.loc[a:b] = True
et["4. + duracao 120min"] = fin & sel

print("ONDE CADA EVENTO E PERDIDO  (nasce na banda [4h,48h]?)")
print("=" * 104)
print(f"{'evento':>12} | " + "".join(f"{k:>22}" for k in et))
print("-" * 104)
for t in alvo:
    lo, hi = t - JAN, t - pd.Timedelta(hours=TMIN)
    linha = ""
    for k, s in et.items():
        eps = AV.episodios(s)
        nasce = [a for a, _ in eps if lo <= a <= hi]
        if nasce:
            linha += f"{f'NASCE {(t-max(nasce)).total_seconds()/3600:.0f}h':>22}"
        else:
            dep = [(a, b) for a, b in eps if a <= t and b >= lo]
            linha += f"{(f'de pe (ini -{(t-dep[0][0]).total_seconds()/3600:.0f}h)' if dep else 'nada'):>22}"
    print(f"{t:%d/%m/%Y} | {linha}")
print("-" * 104)
for k, s in et.items():
    eps = AV.episodios(s)
    nb = sum(1 for t in alvo
             if any(t - JAN <= a <= t - pd.Timedelta(hours=TMIN) for a, _ in eps))
    print(f"  {k:<24} banda {nb}/8   ({len(eps)} episodios, "
          f"{100*float(s.sum())/float(mask.sum()):.1f}% do tempo)")
