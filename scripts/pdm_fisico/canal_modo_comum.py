#!/usr/bin/env python3
"""O canal de MODO COMUM dos termopares na nossa maquina, e a ablacao de sp e p.

DIAGNOSTICO (dois_pontos_cegos.py, 11/09/2026), razao de cobertura na banda
acionavel contra o nulo:
    modo comum (novo) 10,4% duty | 6/8 | 2,52x | p=0,013
    t                 24,1%      | 5/8 | 2,09x
    vb                52,0%      | 7/8 | 1,48x
    sp                41,8%      | 2/8 | 0,55x   <- ABAIXO do acaso
    p                 34,4%      | 3/8 | 0,83x   <- ABAIXO do acaso

O modo comum e o ponto cego da PCA: deslocamento coerente de todos os termopares
e absorvido pelas primeiras componentes. Em 17/03 os seis TC382 e o T5_AVG sobem
juntos em z de 4,9 a 6,5 e o nosso canal `t` nao acende.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import g, partes, mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h
from cabiunas_pdm import config as C
from precursor_ou_cascata import margem

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0

TC = [c for c in C.TEMPERATURE_TAGS if c.startswith("TC382") or c == "T5_AVG_A"]
mc = g[TC].astype("float64").where(mask).mean(axis=1)
n400 = int(pd.Timedelta("400h")/pd.Timedelta("2min"))
ref = mc.rolling(n400, min_periods=2000)
ZMC = ((mc - ref.median())/(ref.quantile(.75)-ref.quantile(.25)).replace(0, np.nan)).abs()
MG = margem("954005_624_TI_0305")[0]
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

print(f"{'configuracao':>44} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 106)
base = {c: ON[c] for c in SIN}
b0 = mede(pos(pd.Series(sum(base[c].astype(int) for c in SIN) >= 2, index=idx)
              & mask & (ON["sp"] | ON["vb"]), REFRAT_H, DUR_MIN))
print(f"{'os nossos 4 (publicado)':>44} | {b0[0]:5d}/8{b0[1]:7d}/8{b0[2]:6d}/8"
      f"{b0[3]:10.3f}{b0[4]:9.1f}{b0[5]:8.1f}h")
print("-" * 106)

melhor = None
CONJ = {
    "4 + modo comum":            lambda z, m: {**base, "mc": z},
    "4 + modo comum + margem":   lambda z, m: {**base, "mc": z, "mg": m},
    "sem sp: t,p,vb + mc":       lambda z, m: {"t": base["t"], "p": base["p"],
                                               "vb": base["vb"], "mc": z},
    "sem p: t,sp,vb + mc":       lambda z, m: {"t": base["t"], "sp": base["sp"],
                                               "vb": base["vb"], "mc": z},
    "sem sp e p: t,vb + mc + mg":lambda z, m: {"t": base["t"], "vb": base["vb"],
                                               "mc": z, "mg": m},
}
for nome, f in CONJ.items():
    for zq, mq in itertools.product((2.0, 3.0), (0.15, 0.20)):
        ch = f((ZMC >= zq).fillna(False) & mask, (MG >= mq) & mask)
        ns = sum(ch[k].astype(int) for k in ch)
        for nv in (2, 3):
            porta = (ON["sp"] | ON["vb"]) if "sp" in ch else ON["vb"]
            v = pd.Series(ns >= nv, index=idx) & mask & porta
            r = mede(pos(v, REFRAT_H, DUR_MIN))
            if r[2] == 8 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
                melhor = ((r[0], -r[3], -r[4]), nome, zq, mq, nv, r)
            if r[0] > b0[0] and r[2] >= 7:
                print(f"{f'{nome}, z>={zq:.0f}, m>={mq:.2f}, voto>={nv}':>44} | "
                      f"{r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
                      f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 106)
if melhor:
    _, nome, zq, mq, nv, r = melhor
    print(f"  MELHOR com det=8/8: {nome}, z>={zq:.0f}, margem>={mq:.2f}, voto>={nv}")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
