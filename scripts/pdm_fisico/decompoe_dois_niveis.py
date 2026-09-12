#!/usr/bin/env python3
"""Cada nivel contribui com o que? E o portao esta mesmo inerte?

O portao `sp|vb` so existe no nivel ESPECIFICO (B). Se o nivel SENSIVEL (A) ja
cobre tudo que o B cobre, o portao e irrelevante -- mas ai o B inteiro tambem e.
Decompoe para saber qual das duas coisas e verdade.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import paradas_reais_2h
from confirma_vizinhanca import canal, P

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
p = {k: (dict(P["lo"]) if k == "lo" else P[k]) for k in P}; p["frac"] = 0.0
A = {c: canal(c, p["lo"][c]) for c in SIN}
B = {c: canal(c, p["hi"] if c != "vb" else p["kvhi"]) for c in SIN}
KH = {"t": p["hi"], "p": p["hi"], "sp": p["hi"], "vb": p["kvhi"]}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()

vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
vB_sem = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask
vB_com = vB_sem & (B["sp"] | B["vb"])

def final(v):
    vv = quebra_idade(np.asarray(v), F, p["abs_"], p["idade"])
    return pos_dur_esc(pd.Series(vv, index=idx), p["refrat"], F,
                       p["abs_"], p["idade"], p["dur_esc"])

def mede(al):
    eps = AV.episodios(al); banda, ini, leads = 0, 0, []
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

print("QUANTO TEMPO CADA NIVEL FICA ACESO, E QUANTO E EXCLUSIVO")
print("=" * 92)
tot = float(mask.sum())
print(f"  nivel A (sensivel, >=3 de 4, limiar baixo) : {100*float(vA.sum())/tot:6.2f}%")
print(f"  nivel B SEM portao (>=2 de 4, limiar alto) : {100*float(vB_sem.sum())/tot:6.2f}%")
print(f"  nivel B COM portao sp|vb                   : {100*float(vB_com.sum())/tot:6.2f}%")
print(f"     o portao corta {100*float((vB_sem & ~vB_com).sum())/tot:.2f}% "
      f"({100*float((vB_sem & ~vB_com).sum())/max(float(vB_sem.sum()),1):.1f}% do nivel B)")
print(f"\n  B COM portao que NAO esta em A (exclusivo) : "
      f"{100*float((vB_com & ~vA).sum())/tot:6.2f}%")
print(f"  B SEM portao que NAO esta em A (exclusivo) : "
      f"{100*float((vB_sem & ~vA).sum())/tot:6.2f}%")
print(f"  A que nao esta em B COM portao             : "
      f"{100*float((vA & ~vB_com).sum())/tot:6.2f}%")

print("\n\nRESULTADO DE CADA COMBINACAO")
print("=" * 108)
print(f"{'gatilho':>34} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}"
      f"{'lead':>9}{'eps':>6}")
print("-" * 108)
for nome, v in [("so o nivel A (sensivel)", vA),
                ("so o nivel B com portao", vB_com),
                ("so o nivel B SEM portao", vB_sem),
                ("A ou B com portao  (o ponto)", vA | vB_com),
                ("A ou B SEM portao", vA | vB_sem)]:
    r = mede(final(v.to_numpy()))
    print(f"{nome:>34} | {r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
          f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h{r[6]:6d}")
