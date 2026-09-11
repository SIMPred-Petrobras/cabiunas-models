#!/usr/bin/env python3
"""Vizinhanca do ponto novo e tabela por evento."""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo
from publica_clearml import SIN, BASE
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from limiar_por_canal import canal, avalia_tudo, FRAC, IDADE, ABS, REFRAT, DUR_ESC

HI, KVHI = 1.7, 2.2
BEST = {"t": 1.1, "p": 0.7, "sp": 0.9, "vb": 1.8}
B = {c: canal(c, HI if c != "vb" else KVHI) for c in SIN}
vB = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask & (B["sp"] | B["vb"])
KH = {"t": HI, "p": HI, "sp": HI, "vb": KVHI}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()

def constroi(KL):
    A = {c: canal(c, KL[c]) for c in SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
    vv = quebra_idade(corta_rearma((vA | vB).to_numpy(), F, FRAC), F, ABS, IDADE)
    return pos_dur_esc(pd.Series(vv, index=idx), REFRAT, F, ABS, IDADE, DUR_ESC)

print("VIZINHANCA -- variando um canal por vez em torno do ponto")
print("=" * 92)
for c in SIN:
    print(f"\n  k_lo[{c}]  (demais fixos em {', '.join(f'{k}={v}' for k, v in BEST.items() if k != c)})")
    print(f"  {'valor':>7} {'banda':>7} {'ini':>6} {'det':>6} {'FP/mes':>9} {'h/mes':>8} {'lead':>8}")
    for k in ([0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5] if c != "vb" else [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.6]):
        KL = dict(BEST); KL[c] = k
        b, i, d, fp, h, lm = avalia_tudo(constroi(KL))
        m = "  <<<" if k == BEST[c] else ""
        print(f"  {k:7.1f} {b:5d}/8 {i:4d}/8 {d:4d}/8 {fp:9.3f} {h:8.1f} "
              f"{(lm if np.isfinite(lm) else 0):7.1f}h{m}")

print("\n\nPOR EVENTO -- ponto novo contra o publicado")
print("=" * 100)
al = constroi(BEST); eps = AV.episodios(al); JAN = pd.Timedelta("48h")
PUB = {"27/02/2025": "de pe 144h", "17/03/2025": "de pe 195h", "07/04/2025": "nasce  5,0h",
       "11/04/2025": "nasce  2,8h", "29/04/2025": "de pe  17h", "04/11/2025": "nasce  8,8h",
       "09/12/2025": "nasce 23,6h", "26/02/2026": "de pe 670h"}
print(f"{'evento':>12} | {'publicado':>14} | {'novo':>26} | banda?")
print("-" * 100)
nb = 0
for t in alvo:
    k = t.strftime("%d/%m/%Y"); t0 = t - JAN
    nasc = [(a, b) for a, b in eps if t0 <= a <= t]
    if nasc:
        a, b = max(nasc, key=lambda ab: ab[0])
        lead = (t-a).total_seconds()/3600; dur = (b-a).total_seconds()/60
        s = f"nasce {lead:5.1f}h  dura {dur:5.0f}min"
        ok = "SIM" if lead >= 4 else "nao (<4h)"
        nb += lead >= 4
    else:
        dep = [(a, b) for a, b in eps if a <= t and b >= t0]
        s = f"de pe (ep de {(dep[0][1]-dep[0][0]).total_seconds()/3600:.0f}h)" if dep else "nao detecta"
        ok = "nao"
    print(f"{k:>12} | {PUB[k]:>14} | {s:>26} | {ok}")
print("-" * 100)
print(f"  na banda acionavel: {nb}/8")
