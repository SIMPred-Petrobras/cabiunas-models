#!/usr/bin/env python3
"""Combina as duas ideias que passaram: gatilho de dois niveis + escalada por idade.

  dois niveis   (dois_niveis.py)        -- lead 14,4 -> 19,3 h, FP 0,775 -> 0,689
  escalada/idade(escalada_por_idade.py) -- inicio 5/8 -> 6/8 a custo zero

Sao ortogonais: uma mexe em QUANDO o alarme nasce, a outra em QUANDO ele pode
renascer. Testa a uniao e mapeia o plato.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, DUR_MIN
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

reset = (~mask).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
TMIN, TMAX = 4.0, 48.0
DUR_ESC = 60
CACHE = {}


def canais(kb, kv):
    if (kb, kv) in CACHE: return CACHE[(kb, kv)]
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}; out = {}
    for c in SIN:
        thr = BASE[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    CACHE[(kb, kv)] = out; return out


def constroi(lo, hi, kv, idade, ABS, frac=0.03, refrat=72):
    A, B = canais(lo, kv), canais(hi, kv)
    nsA = sum(A[c].astype(int) for c in SIN); nsB = sum(B[c].astype(int) for c in SIN)
    v = ((pd.Series(nsA >= 3, index=idx) & mask) |
         (pd.Series(nsB >= 2, index=idx) & mask & (B["sp"] | B["vb"])))
    K = {"t": hi, "p": hi, "sp": hi, "vb": kv}
    F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
    vv = quebra_idade(corta_rearma(v.to_numpy(), F, frac), F, ABS, idade)
    return pos_dur_esc(pd.Series(vv, index=idx), refrat, F, ABS, idade, DUR_ESC)


def mede(al):
    eps = AV.episodios(al); banda, leads, ini = 0, [], 0
    for t in alvo:
        c = [a for a, _ in eps
             if t - pd.Timedelta(hours=TMAX) <= a <= t - pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if any(t - pd.Timedelta(hours=TMAX) <= a <= t for a, _ in eps): ini += 1
    m = AV.avalia(al, alvo, mask); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    return banda, ini, m["det"], nfp/meses, h/meses, (np.mean(leads) if leads else np.nan)


print("DOIS NIVEIS + ESCALADA POR IDADE")
print("=" * 108)
print(f"{'kb_lo':>6} {'kb_hi':>6} {'kv':>5} {'idade':>6} {'ABS':>5} | {'banda':>7} "
      f"{'inicio':>7} {'det':>6} {'FP/mes':>9} {'h/mes':>8} {'lead':>8}")
print("-" * 108)
res = []
for lo, hi, kv, ida, ab in itertools.product([0.8, 1.0, 1.2], [1.7, 2.0], [2.2],
                                             [72, 96, 120], [10, 20, 50]):
    al = constroi(lo, hi, kv, ida, ab)
    r = mede(al)
    res.append((r[0], r[1], -r[3], lo, hi, kv, ida, ab) + r)
res.sort(reverse=True)
for row in res[:14]:
    _, _, _, lo, hi, kv, ida, ab, banda, ini, det, fp, h, lm = row
    print(f"{lo:6.1f} {hi:6.1f} {kv:5.1f} {ida:5d}h {ab:5.0f} | {banda:5d}/8 {ini:5d}/8 "
          f"{det:5d}/8 {fp:9.3f} {h:8.1f} {lm:7.1f}h")
print("-" * 108)
print("  publicado : banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes, lead 10,1 h")
print("  atual     : banda 4/8, inicio 6/8, det 8/8, 0,775 FP/mes, lead 14,4 h")
print("  Diego     : banda 7/8, inicio 8/8, det 8/8, 2,880 FP/mes, lead 23,8 h")
