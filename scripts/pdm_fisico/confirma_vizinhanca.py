#!/usr/bin/env python3
"""Confirmacao de vizinhanca do ponto de dois niveis, sob a REGUA DE INICIO.

Adotada como metrica de trabalho em 11/09/2026. O ponto da 6/8 na regua de inicio
contra os 4/8 do publicado, ao mesmo custo. Antes de adotar: variar um parametro
por vez e verificar que nao esta numa borda.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import paradas_reais_2h

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()
P = dict(lo={"t":1.1,"p":0.7,"sp":0.9,"vb":1.8}, hi=1.7, kvhi=2.2,
         frac=0.03, idade=96, abs_=20, refrat=72, dur_esc=60)
CACHE = {}

def canal(c, k):
    if (c, k) in CACHE: return CACHE[(c, k)]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    CACHE[(c, k)] = (deg | cu) & mask
    return CACHE[(c, k)]

def roda(p):
    A = {c: canal(c, p["lo"][c]) for c in SIN}
    B = {c: canal(c, p["hi"] if c != "vb" else p["kvhi"]) for c in SIN}
    KH = {"t": p["hi"], "p": p["hi"], "sp": p["hi"], "vb": p["kvhi"]}
    F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
    vB = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask & (B["sp"] | B["vb"])
    vv = quebra_idade(corta_rearma((vA | vB).to_numpy(), F, p["frac"]), F, p["abs_"], p["idade"])
    al = pos_dur_esc(pd.Series(vv, index=idx), p["refrat"], F, p["abs_"], p["idade"], p["dur_esc"])
    eps = AV.episodios(al); ini, banda, leads = 0, 0, []
    for t in alvo:
        if any(t-JAN <= a <= t for a, _ in eps): ini += 1
        c = [a for a, _ in eps if t-JAN <= a <= t-pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t-max(c)).total_seconds()/3600)
    jw = [(t-JAN, t) for t in alvo]; fp = h = 0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw): continue
        if len(paradas[(paradas.ini >= a) & (paradas.ini <= b+JAN)]): continue
        fp += 1; h += (b-a).total_seconds()/3600
    return ini, banda, AV.avalia(al, alvo, mask)["det"], fp/meses, h/meses, \
           (np.mean(leads) if leads else np.nan)

r0 = roda(P)
print("PONTO CANDIDATO -- dois niveis por canal")
print(f"  inicio {r0[0]}/8 | banda {r0[1]}/8 | det {r0[2]}/8 | {r0[3]:.3f} FP/mes | "
      f"{r0[4]:.1f} h/mes | lead {r0[5]:.1f}h")
print(f"  publicado: inicio 4/8 | banda 3/8 | det 8/8 | 0,517 FP/mes | 7,1 h/mes\n")
print("VIZINHANCA -- um parametro por vez")
print("=" * 96)
EIXOS = [("lo.t", [0.8,0.9,1.0,1.1,1.3,1.5]), ("lo.p", [0.5,0.6,0.7,0.8,1.0,1.2]),
         ("lo.sp", [0.6,0.7,0.9,1.1,1.3,1.5]), ("lo.vb", [1.4,1.6,1.8,2.0,2.4,2.8]),
         ("hi", [1.4,1.5,1.7,2.0,2.2]), ("kvhi", [1.8,2.0,2.2,2.6,3.0]),
         ("frac", [0.0,0.01,0.02,0.03,0.05,0.08]), ("idade", [48,72,96,120,144]),
         ("abs_", [8,14,20,35,60,120]), ("refrat", [48,60,72,84,96]),
         ("dur_esc", [30,45,60,90,120])]
for nome, gr in EIXOS:
    print(f"\n  {nome}")
    print(f"    {'valor':>8}{'inicio':>9}{'banda':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
    for v in gr:
        p = {k: (dict(P["lo"]) if k == "lo" else P[k]) for k in P}
        if nome.startswith("lo."): p["lo"][nome[3:]] = v
        else: p[nome] = v
        r = roda(p)
        atual = ((nome.startswith("lo.") and P["lo"][nome[3:]] == v) or
                 (not nome.startswith("lo.") and P.get(nome) == v))
        print(f"    {v:>8}{r[0]:7d}/8{r[1]:6d}/8{r[2]:5d}/8{r[3]:10.3f}{r[4]:9.1f}"
              f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h" + ("   <<<" if atual else ""))
