"""CONTROLE NEGATIVO do lead fisico: quanto da a mesma medida SEM trip?

Se em instantes aleatorios o sinal tambem aparece 'desviado' ha 160 h, a medida
nao mede precursor -- mede que os canais ficam altos quase sempre.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lead_fisico as LF
from pos_processamento import mask, idx, alvo

RNG = np.random.default_rng(20260919)
N = 200
m = mask.to_numpy()
# instantes elegiveis: em regime, com 37 dias de historico antes
cand = idx[(idx >= idx[0] + LF.BUSCA + LF.REF) & m]
# e longe de qualquer trip (>= 14 dias), para o nulo ser mesmo nulo
longe = [t for t in cand if all(abs((t - a).total_seconds()) > 14*86400 for a in alvo)]
sorteio = RNG.choice(np.asarray(longe), size=N, replace=False)

print(f"{N} instantes aleatorios em regime, a >= 14 dias de qualquer trip\n")
obs, nul = {}, {}
for c in LF.SIN:
    o = [LF.inicio_desvio(c, t) for t in alvo]
    n = [LF.inicio_desvio(c, pd.Timestamp(t)) for t in sorteio]
    obs[c] = [x for x in o if np.isfinite(x)]
    nul[c] = [x for x in n if np.isfinite(x)]

print(f"{'canal':>6}{'trips: cruzam':>16}{'mediana':>10}   "
      f"{'nulo: cruzam':>15}{'mediana':>10}   razao")
print("-"*76)
for c in LF.SIN:
    fo = len(obs[c])/len(alvo); fn = len(nul[c])/N
    mo = np.median(obs[c]) if obs[c] else float("nan")
    mn = np.median(nul[c]) if nul[c] else float("nan")
    print(f"{c:>6}{fo:>15.0%}{mo:>10.1f}h{fn:>15.0%}{mn:>10.1f}h"
          f"{fo/max(fn,1e-9):>8.2f}x")

# o max entre canais, que e o que reportei
mo = [np.nanmax([LF.inicio_desvio(c, t) for c in LF.SIN]) for t in alvo]
mn = [np.nanmax([LF.inicio_desvio(c, pd.Timestamp(t)) for c in LF.SIN]) for t in sorteio]
mo = [x for x in mo if np.isfinite(x)]; mn = [x for x in mn if np.isfinite(x)]
print(f"\nMAX entre canais (o numero que eu havia reportado):")
print(f"   trips: {len(mo)}/{len(alvo)} cruzam, mediana {np.median(mo):.1f}h")
print(f"   nulo : {len(mn)}/{N} cruzam ({100*len(mn)/N:.0f}%), mediana {np.median(mn):.1f}h")
p = float(np.mean([x >= np.median(mo) for x in mn])) if mn else float("nan")
print(f"   fracao de janelas nulas com lead >= {np.median(mo):.0f}h: {p:.1%}")
