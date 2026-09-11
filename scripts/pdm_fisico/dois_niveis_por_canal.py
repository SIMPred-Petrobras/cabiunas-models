#!/usr/bin/env python3
"""Dois niveis COM limiar livre por canal no nivel sensivel.

Limiar livre por canal sozinho (limiar_por_canal.py) da banda 5/8 a 0,947 FP/mes
-- pior que o gatilho de dois niveis (5/8 a 0,689). Mas os dois atacam a mesma
coisa por caminhos diferentes: o dois niveis ganha sensibilidade cedo SEM perder
cobertura, porque mantem o nivel especifico; o limiar livre tem de escolher um
regime so.

Aqui o nivel SENSIVEL (3 de 4 canais) ganha limiar proprio por canal, e o nivel
ESPECIFICO (2 de 4 + portao) fica no ponto validado 1,7/2,2. E a combinacao das
duas pecas que passaram.
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
from limiar_por_canal import canal, avalia_tudo, FRAC, IDADE, ABS, REFRAT, DUR_ESC

HI, KVHI = 1.7, 2.2                       # nivel especifico, ponto validado
GR = {"t":  [0.7, 0.9, 1.0, 1.1, 1.3],
      "p":  [0.7, 0.9, 1.0, 1.1, 1.3],
      "sp": [0.7, 0.9, 1.0, 1.1, 1.3],
      "vb": [1.4, 1.8, 2.2, 2.6, 3.0]}
print(f"varrendo {np.prod([len(v) for v in GR.values()])} combinacoes ...", flush=True)

B = {c: canal(c, HI if c != "vb" else KVHI) for c in SIN}
nsB = sum(B[c].astype(int) for c in SIN)
vB = pd.Series(nsB >= 2, index=idx) & mask & (B["sp"] | B["vb"])
KH = {"t": HI, "p": HI, "sp": HI, "vb": KVHI}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()

res = []
for kt, kp, ks, kv in itertools.product(GR["t"], GR["p"], GR["sp"], GR["vb"]):
    KL = {"t": kt, "p": kp, "sp": ks, "vb": kv}
    A = {c: canal(c, KL[c]) for c in SIN}
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
    v = (vA | vB).to_numpy()
    vv = quebra_idade(corta_rearma(v, F, FRAC), F, ABS, IDADE)
    al = pos_dur_esc(pd.Series(vv, index=idx), REFRAT, F, ABS, IDADE, DUR_ESC)
    b, i, d, fp, h, lm = avalia_tudo(al)
    res.append(dict(kt=kt, kp=kp, ksp=ks, kvb=kv, banda=b, ini=i, det=d,
                    fp=round(fp, 3), h=round(h, 1),
                    lead=round(lm, 1) if np.isfinite(lm) else np.nan))
D = pd.DataFrame(res); D.to_csv("dois_niveis_por_canal.csv", index=False)

print(f"\n{len(D)} combinacoes\n")
print("MELHOR POR NIVEL DE BANDA ACIONAVEL, mantendo det = 8/8")
print("=" * 108)
print(f"{'banda':>7} {'n':>5} {'ini':>5} {'FP/mes':>9} {'h/mes':>8} {'lead':>8}   "
      f"k_lo t/p/sp/vb")
print("-" * 108)
ok = D[D.det == 8]
for k in sorted(ok.banda.unique(), reverse=True):
    s = ok[ok.banda == k].sort_values(["fp", "h"])
    b = s.iloc[0]
    print(f"{int(k):5d}/8 {len(s):5d} {int(b.ini):3d}/8 {b.fp:9.3f} {b.h:8.1f} {b.lead:7.1f}h"
          f"   {b.kt}/{b.kp}/{b.ksp}/{b.kvb}")
print("-" * 108)
print("  referencia -- dois niveis com k_lo uniforme 1,0: banda 5/8, inicio 6/8, "
      "0,689 FP/mes, 8,7 h/mes, lead 19,3 h")
u = D[(D.kt == 1.0) & (D.kp == 1.0) & (D.ksp == 1.0) & (D.kvb == 2.2)]
if len(u):
    a = u.iloc[0]
    print(f"  confere no grid: banda {int(a.banda)}/8, inicio {int(a.ini)}/8, "
          f"{a.fp:.3f} FP/mes, {a.h:.1f} h/mes, lead {a.lead:.1f} h")
