#!/usr/bin/env python3
"""Checagem final do ponto combinado: vizinhanca de kb_lo e tabela por evento."""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from combina_final import constroi, mede

print("VIZINHANCA DE kb_lo  (kb_hi=1,7  kv=2,2  idade=96h  ABS=20)")
print("=" * 92)
print(f"{'kb_lo':>7} | {'banda':>7} {'inicio':>7} {'det':>6} {'FP/mes':>9} {'h/mes':>8} {'lead':>8}")
print("-" * 92)
for lo in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.7]:
    b, i, d, fp, h, lm = mede(constroi(lo, 1.7, 2.2, 96, 20))
    m = "  <<< escolhido" if lo == 1.0 else ""
    print(f"{lo:7.1f} | {b:5d}/8 {i:5d}/8 {d:5d}/8 {fp:9.3f} {h:8.1f} {lm:7.1f}h{m}")

print("\n\nPOR EVENTO -- ponto final contra o publicado")
print("=" * 100)
al = constroi(1.0, 1.7, 2.2, 96, 20)
eps = AV.episodios(al)
JAN = pd.Timedelta(hours=48)
PUB = {"27/02/2025": "de pe 144h", "17/03/2025": "de pe 195h", "07/04/2025": "nasce 5,0h",
       "11/04/2025": "nasce 2,8h", "29/04/2025": "de pe 17h", "04/11/2025": "nasce 8,8h",
       "09/12/2025": "nasce 23,6h", "26/02/2026": "de pe 670h"}
print(f"{'evento':>12} | {'publicado':>16} | {'ponto final':>28} | acionavel?")
print("-" * 100)
n_band = 0
for t in alvo:
    k = t.strftime("%d/%m/%Y"); t0 = t - JAN
    nasc = [(a, b) for a, b in eps if t0 <= a <= t]
    if nasc:
        a, b = max(nasc, key=lambda ab: ab[0])
        lead = (t - a).total_seconds()/3600
        dur = (b - a).total_seconds()/60
        s = f"nasce {lead:5.1f}h  dura {dur:5.0f}min"
        ok = "SIM" if lead >= 4.0 else f"nao (lead < 4h)"
        n_band += lead >= 4.0
    else:
        dep = [(a, b) for a, b in eps if a <= t and b >= t0]
        s = (f"de pe (ep de {(dep[0][1]-dep[0][0]).total_seconds()/3600:.0f}h)"
             if dep else "nao detecta")
        ok = "nao"
    print(f"{k:>12} | {PUB[k]:>16} | {s:>28} | {ok}")
print("-" * 100)
print(f"  na banda acionavel [4h, 48h]: {n_band}/8")
