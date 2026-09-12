#!/usr/bin/env python3
"""Confirma o ponto de halflife 30min/10min: plato, por evento, e o porque."""
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import idx, alvo
from blackout_e_halflife import monta, mede, HL0, CRU
from publica_clearml import SIN

JAN = pd.Timedelta("48h")
GR = ["5min", "10min", "15min", "20min", "30min", "45min", "1h", "90min", "2h"]
print("PLATO FINO EM HALFLIFE (blackout 6 h)")
print("=" * 100)
print(f"{'t/p':>8} | " + "".join(f"{b:>16}" for b in ["10min", "20min", "30min", "45min"]))
print(f"{'':>8} | " + "".join(f"{'banda/ini  FP':>16}" for _ in range(4)))
print("-" * 100)
tab = {}
for a_ in GR:
    linha = ""
    for b_ in ["10min", "20min", "30min", "45min"]:
        al, mk = monta(6, {"t": a_, "p": a_, "sp": b_, "vb": b_})
        bd, i, fp, h, lm, ne = mede(al, mk)
        tab[(a_, b_)] = (bd, i, fp, h, lm)
        linha += f"{bd}/{i}  {fp:8.3f}".rjust(16)
    print(f"{a_:>8} | {linha}")

print("\n\nHORAS/MES na mesma grade")
print("=" * 100)
print(f"{'t/p':>8} | " + "".join(f"{b:>16}" for b in ["10min", "20min", "30min", "45min"]))
print("-" * 100)
for a_ in GR:
    print(f"{a_:>8} | " + "".join(f"{tab[(a_,b)][3]:16.1f}" for b in ["10min","20min","30min","45min"]))

print("\n\nPOR EVENTO -- publicado (1h/30min) contra 30min/10min")
print("=" * 96)
al0, mk0 = monta(6, HL0)
al1, mk1 = monta(6, {"t": "30min", "p": "30min", "sp": "10min", "vb": "10min"})
e0, e1 = AV.episodios(al0), AV.episodios(al1)
print(f"{'evento':>12} | {'publicado':>26} | {'30min/10min':>26}")
print("-" * 96)
for t in alvo:
    def q(eps):
        i = [a for a, _ in eps if t - JAN <= a <= t]
        if i:
            a = max(i); b = [y for x, y in eps if x == a][0]
            l = (t-a).total_seconds()/3600
            return (f"nasce {l:5.1f}h dura {(b-a).total_seconds()/60:5.0f}min"
                    + ("  [banda]" if 4 <= l <= 48 else ""))
        return "de pe" if any(x <= t and y >= t - JAN for x, y in eps) else "--"
    c0, c1 = q(e0), q(e1)
    print(f"{t:%d/%m/%Y} | {c0:>26} | {c1:>26}" + ("  <<<" if c0 != c1 else ""))

print("\n\nPOR QUE MELHORA -- duracao dos episodios")
print("=" * 70)
for rot, eps in (("publicado (1h/30min)", e0), ("30min/10min", e1)):
    d = np.array([(b-a).total_seconds()/3600 for a, b in eps])
    print(f"  {rot:<22} {len(d):3d} eps | mediana {np.median(d):6.2f} h | "
          f"media {d.mean():6.2f} h | max {d.max():7.2f} h | total {d.sum():7.0f} h")
