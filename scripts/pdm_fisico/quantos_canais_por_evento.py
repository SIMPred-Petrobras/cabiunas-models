#!/usr/bin/env python3
"""A hipotese que explica ~30 refutacoes: cada evento e detectado por UM canal,
e canais diferentes em eventos diferentes.

Se for isso, qualquer regra que exija corroboracao perde -- e acrescentar canal
bom nao ajuda, porque o canal bom continua tendo de esperar um canal ruim.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import partes, mask, idx, alvo
from publica_clearml import SIN
from plota_estilo_francisco import KB, KV
from sp_vibracao import constroi
from precursor_ou_cascata import margem
from cabiunas_pdm import config as C

TMIN, TMAX = 4.0, 48.0
ON = partes(KB, KV)
SV = (constroi(2400) >= 6) & mask
MG = (margem("954005_624_TI_0305")[0] >= 0.20) & mask
TC = [c for c in C.TEMPERATURE_TAGS if c.startswith("TC382") or c == "T5_AVG_A"]
from pos_processamento import g
mc = g[TC].astype("float64").where(mask).mean(axis=1)
n400 = int(pd.Timedelta("400h")/pd.Timedelta("2min"))
r = mc.rolling(n400, min_periods=2000)
MC = (((mc - r.median())/(r.quantile(.75)-r.quantile(.25)).replace(0, np.nan)).abs() >= 2) & mask

CH = {**{c: ON[c] for c in SIN}, "sp_vib": SV, "margem": MG, "mod_com": MC.fillna(False)}
print("QUAIS CANAIS ESTAO ACESOS NA BANDA [4h, 48h] DE CADA EVENTO")
print("=" * 100)
print(f"{'evento':>12} | " + "".join(f"{k:>9}" for k in CH) + f"{'n':>5}")
print("-" * 100)
cont = []
for t in alvo:
    a, b = t - pd.Timedelta(hours=TMAX), t - pd.Timedelta(hours=TMIN)
    ac = {k: bool(v.loc[a:b].any()) for k, v in CH.items()}
    n = sum(ac.values()); cont.append(n)
    print(f"{t:%d/%m/%Y} | " + "".join(f"{('  X' if ac[k] else '  .'):>9}" for k in CH)
          + f"{n:5d}")
print("-" * 100)
print(f"  canais acesos por evento: mediana {np.median(cont):.0f} de {len(CH)}")

print("\n\nE QUANTOS ESTAO ACESOS SIMULTANEAMENTE, NO MESMO INSTANTE?")
print("=" * 100)
print(f"{'evento':>12}{'max simultaneo na banda':>26}{'horas com >=2':>15}{'horas com >=3':>15}")
print("-" * 100)
sim = []
for t in alvo:
    a, b = t - pd.Timedelta(hours=TMAX), t - pd.Timedelta(hours=TMIN)
    S = sum(v.loc[a:b].astype(int) for v in CH.values())
    mx = int(S.max()) if len(S) else 0
    sim.append(mx)
    print(f"{t:%d/%m/%Y}{mx:26d}{float((S>=2).sum())*2/60:14.1f}h"
          f"{float((S>=3).sum())*2/60:14.1f}h")
print("-" * 100)
print(f"  max simultaneo: mediana {np.median(sim):.0f}")
print("\n  -> se os canais acendem em MOMENTOS diferentes dentro da janela, exigir")
print("     voto>=2 no MESMO instante perde, mesmo com todos os canais 'certos'")
