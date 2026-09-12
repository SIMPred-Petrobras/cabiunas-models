#!/usr/bin/env python3
"""O canal de MARGEM no TI_0305 -- validacao e teste na nossa maquina.

ACHADO (11/09/2026, planilha de limites trazida pelo usuario). A margem de
seguranca do mancal TI_0305 (tipico 70,25 degC -> trip 125 degC), com limiar de
15% de consumo:
    aceso 8,34% do tempo | banda [4h,48h] em 5/8 trips | lead mediano 27,2 h
O nosso canal `sp`, no MESMO sensor mas por spread relativo:
    aceso 41,75% do tempo | banda em 2/8

A 50% da margem e cascata (lead 0,1 h); a 15% e precursor. A diferenca e que 15%
pega a APROXIMACAO lenta e 50% pega o pico final.

Valida contra o nulo e testa como 5o canal na nossa maquina.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import g, partes, mask, idx, alvo, op, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h
from precursor_ou_cascata import margem

RNG = np.random.default_rng(20260911)
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
M, tip, lim = margem("954005_624_TI_0305")

print("1. VALIDACAO CONTRA O NULO")
print("=" * 84)
ti = np.asarray(idx.astype("int64"))
lo_us, hi_us = int(pd.Timedelta(hours=TMAX).value)//1000, int(pd.Timedelta(hours=TMIN).value)//1000
eleg = idx[(idx >= idx[0] + JAN) & op.to_numpy() & mask.to_numpy()]
sort = RNG.choice(np.asarray(eleg.astype("int64")), size=(5000, len(alvo)))
for q in (0.10, 0.15, 0.20, 0.25):
    A = ((M >= q) & mask).to_numpy()
    def conta(T):
        a = np.searchsorted(ti, T - lo_us, "left"); b = np.searchsorted(ti, T - hi_us, "right")
        return sum(1 for x, y in zip(a, b) if y > x and A[x:y].any())
    obs = conta(np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo]))
    nul = np.array([conta(sort[k]) for k in range(5000)])
    p = float((nul >= obs).mean())
    print(f"  margem >= {100*q:2.0f}%: observado {obs}/8 | nulo {nul.mean():.2f}/8 "
          f"(p90 {np.percentile(nul,90):.0f}) | razao {obs/nul.mean():.2f}x | p = {p:.4f}"
          + ("  ***" if p < 0.05 else ""))

print("\n\n2. COMO 5o CANAL NA NOSSA MAQUINA")
print("=" * 100)
ON = partes(KB, KV)
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

ns0 = sum(ON[c].astype(int) for c in SIN)
b0 = mede(pos(pd.Series(ns0 >= 2, index=idx) & mask & (ON["sp"] | ON["vb"]), REFRAT_H, DUR_MIN))
print(f"{'configuracao':>34} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 100)
print(f"{'os nossos 4 (publicado)':>34} | {b0[0]:5d}/8{b0[1]:7d}/8{b0[2]:6d}/8"
      f"{b0[3]:10.3f}{b0[4]:9.1f}{b0[5]:8.1f}h")
print("-" * 100)
melhor = None
for q in (0.10, 0.15, 0.20):
    mg = (M >= q) & mask
    for nv, pt in itertools.product((2, 3),
            (("sp|vb", ON["sp"] | ON["vb"]),
             ("sp|vb|margem", ON["sp"] | ON["vb"] | mg),
             ("margem obrigatoria", mg))):
        ns = ns0 + mg.astype(int)
        v = pd.Series(ns >= nv, index=idx) & mask & pt[1]
        r = mede(pos(v, REFRAT_H, DUR_MIN))
        if r[2] == 8 and (melhor is None or (r[0], -r[3]) > melhor[0]):
            melhor = ((r[0], -r[3]), q, nv, pt[0], r)
        if r[0] > b0[0] or (r[0] == b0[0] and r[3] < b0[3]):
            print(f"{f'4 + margem>={100*q:.0f}%, voto>={nv}, {pt[0]}':>34} | "
                  f"{r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
                  f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 100)
if melhor:
    _, q, nv, pn, r = melhor
    print(f"  MELHOR com det=8/8: margem>={100*q:.0f}%, voto>={nv}, portao={pn}")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
