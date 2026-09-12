#!/usr/bin/env python3
"""Dois pontos cegos diagnosticados na pericia dos 4 eventos faltantes.

A) MARGEM DE VIBRACAO POR MANCAL. A planilha da limites DIFERENTES por mancal:
   TV_351/352/353 -> H=114, HH=125 um ; TV_354/355 -> H=50, HH=64 um.
   O nosso `vb` e o max do z robusto sobre as 10 sondas contra referencia rolante
   -- trata todas igual em termos estatisticos, quando fisicamente o mancal 4 e 5
   tem metade da folga. Nos eventos 27/02 e 26/02 as sondas 353/354 disparam
   (z de 4,6 a 14,6) e o `vb` nao acende.

B) MODO COMUM NOS TERMOPARES. Em 17/03 os seis TC382 e o T5_AVG sobem JUNTOS
   (z de 4,9 a 6,5). E o ponto cego classico da PCA: deslocamento coerente de
   todos os sensores e absorvido pelas primeiras componentes e o erro de
   reconstrucao nao sobe. Um canal de MEDIA dos termopares pega o que a PCA perde.

Mede cada um contra o nulo antes de propor como canal.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo, op
from cabiunas_pdm import config as C

LIM = pd.read_csv("limites_alarme.csv")
RNG = np.random.default_rng(20260911)
TMIN, TMAX = 4.0, 48.0
ti = np.asarray(idx.astype("int64"))
lo_us = int(pd.Timedelta(hours=TMAX).value)//1000
hi_us = int(pd.Timedelta(hours=TMIN).value)//1000
eleg = idx[(idx >= idx[0] + pd.Timedelta(hours=TMAX)) & op.to_numpy() & mask.to_numpy()]
SORT = RNG.choice(np.asarray(eleg.astype("int64")), size=(5000, len(alvo)))
T_OBS = np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo])

def valida(A, rot):
    A = np.asarray(A)
    def conta(T):
        a = np.searchsorted(ti, T - lo_us, "left"); b = np.searchsorted(ti, T - hi_us, "right")
        return sum(1 for x, y in zip(a, b) if y > x and A[x:y].any())
    obs = conta(T_OBS)
    nul = np.array([conta(SORT[k]) for k in range(5000)])
    p = float((nul >= obs).mean())
    duty = 100*float((A & mask.to_numpy()).sum())/float(mask.sum())
    print(f"  {rot:<38} duty {duty:5.2f}% | banda {obs}/8 | nulo {nul.mean():.2f}"
          f" | {obs/max(nul.mean(),1e-9):5.2f}x | p={p:.4f}"
          + ("  ***" if p < 0.05 else ""))
    return obs, p, duty

print("A) MARGEM DE VIBRACAO POR MANCAL (limites da planilha)")
print("=" * 100)
mv = {}
for _, r in LIM.iterrows():
    c = r["col"]
    if not str(c).startswith("TV_") or c not in g.columns: continue
    s = g[c].astype("float64").where(mask); tip = float(s.median())
    alto = r["HH"] if pd.notna(r["HH"]) else r["H"]
    mv[c] = (s - tip) / (alto - tip)
    print(f"  {c:>12}  tipico {tip:6.2f}  HH {alto:6.1f} um  "
          f"margem mediana {float(mv[c].median()):6.3f}  p99 {float(mv[c].quantile(.99)):6.3f}")
MV = pd.DataFrame(mv).max(axis=1)
print()
for q in (0.20, 0.30, 0.40, 0.50, 0.60):
    valida(((MV >= q) & mask).to_numpy(), f"margem de vibracao >= {100*q:.0f}%")

print("\n\nB) MODO COMUM NOS TERMOPARES (o que a PCA nao ve)")
print("=" * 100)
TC = [c for c in C.TEMPERATURE_TAGS if c.startswith("TC382") or c == "T5_AVG_A"]
X = g[TC].astype("float64").where(mask)
mc = X.mean(axis=1)                                  # modo comum = media dos termopares
ref = mc.rolling(int(pd.Timedelta("400h")/pd.Timedelta("2min")), min_periods=2000)
z = ((mc - ref.median()) / (ref.quantile(.75) - ref.quantile(.25)).replace(0, np.nan)).abs()
print(f"  {len(TC)} termopares | z do modo comum: mediana {float(z.median()):.2f}"
      f" | p99 {float(z.quantile(.99)):.2f}")
for q in (2.0, 3.0, 4.0, 5.0, 6.0):
    valida(((z >= q) & mask).fillna(False).to_numpy(), f"z do modo comum >= {q:.0f}")

print("\n\n  referencia: o nosso canal vb e o t, para comparacao")
from pos_processamento import partes
from plota_estilo_francisco import KB, KV
ON = partes(KB, KV)
for c in ("t", "vb", "sp", "p"):
    valida(ON[c].to_numpy(), f"nosso canal {c}")
