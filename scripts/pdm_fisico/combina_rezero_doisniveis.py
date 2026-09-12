#!/usr/bin/env python3
"""Ultimo teste: rezero do CUSUM + gatilho de dois niveis.

Os dois melhoram a regua de inicio por caminhos diferentes -- o rezero encurta
episodio, o dois niveis antecipa o nascimento. Se nao forem redundantes, somam.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, CARGA, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h
from cusum_rezera import cusum_rz, mede

reset = (~mask).to_numpy()
CACHE = {}
def canal(c, k, modo, par):
    key = (c, k, modo, par)
    if key in CACHE: return CACHE[key]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    x = ((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
    cu = pd.Series(cusum_rz(x, reset, H_CUSUM, modo, par) > H_CUSUM, index=idx)
    CACHE[key] = (deg | cu) & mask
    return CACHE[key]

def pos(v, refrat_h, dur_min):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(v):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b-a).total_seconds()/60 + 2 >= dur_min: fin.loc[a:b] = True
    return fin & sel

print("REZERO + DOIS NIVEIS")
print("=" * 104)
print(f"{'modo CUSUM':>20}{'k_lo':>6}{'k_hi':>6}{'refr':>6} | {'banda':>7}{'inicio':>8}"
      f"{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 104)
melhor = None
for modo, par in (("nunca", 0), ("head", 0.5)):
    for lo, hi, rf in itertools.product([0.8, 1.0, 1.2], [1.7, 2.0], [48, 72, 96]):
        A = {c: canal(c, lo if c != "vb" else 1.8, modo, par) for c in SIN}
        B = {c: canal(c, hi if c != "vb" else 2.2, modo, par) for c in SIN}
        vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
        vB = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask & (B["sp"] | B["vb"])
        b, i, d, fp, h, lm, ne = mede(pos(vA | vB, rf, DUR_MIN))
        if d == 8 and (melhor is None or (b, -fp, -h) > melhor[0]):
            melhor = ((b, -fp, -h), modo, par, lo, hi, rf, b, i, d, fp, h, lm)
        if b >= 4 and d >= 7:
            print(f"{(modo + (f' {par}' if par else '')):>20}{lo:6.1f}{hi:6.1f}{rf:6d} | "
                  f"{b:5d}/8{i:7d}/8{d:6d}/8{fp:10.3f}{h:9.1f}"
                  f"{(lm if np.isfinite(lm) else 0):8.1f}h")
print("-" * 104)
if melhor:
    _, modo, par, lo, hi, rf, b, i, d, fp, h, lm = melhor
    print(f"  MELHOR com det=8/8: cusum={modo}{par or ''} k={lo}/{hi} refrat={rf}h")
    print(f"     -> banda {b}/8, inicio {i}/8, {fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
else:
    print("  nenhuma configuracao mantem det = 8/8")
print("\n  publicado         : banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes, 7,1 h/mes")
print("  melhor ja medido  : banda 5/8, inicio 6/8, det 8/8, 0,517 FP/mes, 7,8 h/mes (dois niveis por canal)")
