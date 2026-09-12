#!/usr/bin/env python3
"""O melhor ponto medido (dois niveis por canal) + o canal de margem ao setpoint.

  dois niveis por canal : banda 5/8, inicio 6/8, det 8/8, 0,517 FP/mes, 7,8 h/mes
  4 canais + margem     : banda 4/8, inicio 4/8, det 8/8, 0,603 FP/mes, 7,7 h/mes

Sao ortogonais: um mexe em QUANDO o alarme nasce (dois regimes de confirmacao),
o outro traz informacao de natureza nova (limite de protecao de engenharia).
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import paradas_reais_2h
from precursor_ou_cascata import margem

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()
FRAC, IDADE, ABS, REFRAT, DUR_ESC = 0.03, 96, 20, 72, 60
MG_ALL = margem("954005_624_TI_0305")[0]
CACHE = {}

def canal(c, k):
    if (c, k) in CACHE: return CACHE[(c, k)]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    CACHE[(c, k)] = (deg | cu) & mask
    return CACHE[(c, k)]

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

# nivel especifico fixo no ponto validado
HI, KVHI = 1.7, 2.2
B = {c: canal(c, HI if c != "vb" else KVHI) for c in SIN}
vB = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask & (B["sp"] | B["vb"])
KH = {"t": HI, "p": HI, "sp": HI, "vb": KVHI}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
LO = {"t": 1.1, "p": 0.7, "sp": 0.9, "vb": 1.8}      # ponto validado do nivel sensivel

print(f"{'configuracao':>42} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}"
      f"{'h/mes':>9}{'lead':>9}")
print("-" * 100)
A = {c: canal(c, LO[c]) for c in SIN}
vA0 = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
base = pos_dur_esc(pd.Series(quebra_idade(corta_rearma((vA0 | vB).to_numpy(), F, FRAC),
                                          F, ABS, IDADE), index=idx),
                   REFRAT, F, ABS, IDADE, DUR_ESC)
r0 = mede(base)
print(f"{'dois niveis por canal (melhor medido)':>42} | {r0[0]:5d}/8{r0[1]:7d}/8{r0[2]:6d}/8"
      f"{r0[3]:10.3f}{r0[4]:9.1f}{r0[5]:8.1f}h")
print("-" * 100)
melhor = None
for mq, onde in itertools.product((0.10, 0.15, 0.20, 0.25),
                                  ("nivel sensivel", "nivel especifico", "os dois")):
    mg = (MG_ALL >= mq) & mask
    nA = sum(A[c].astype(int) for c in SIN) + (mg.astype(int) if onde != "nivel especifico" else 0)
    nB = sum(B[c].astype(int) for c in SIN) + (mg.astype(int) if onde != "nivel sensivel" else 0)
    vA = pd.Series(nA >= 3, index=idx) & mask
    vBB = pd.Series(nB >= 2, index=idx) & mask & (B["sp"] | B["vb"] | mg)
    al = pos_dur_esc(pd.Series(quebra_idade(corta_rearma((vA | vBB).to_numpy(), F, FRAC),
                                            F, ABS, IDADE), index=idx),
                     REFRAT, F, ABS, IDADE, DUR_ESC)
    r = mede(al)
    if r[2] == 8 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
        melhor = ((r[0], -r[3], -r[4]), mq, onde, r)
    if r[0] >= r0[0] and r[2] >= 7:
        print(f"{f'+ margem>={100*mq:.0f}% no {onde}':>42} | {r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8"
              f"{r[3]:10.3f}{r[4]:9.1f}{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 100)
if melhor:
    _, mq, onde, r = melhor
    print(f"  MELHOR com det=8/8: margem>={100*mq:.0f}% no {onde}")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
