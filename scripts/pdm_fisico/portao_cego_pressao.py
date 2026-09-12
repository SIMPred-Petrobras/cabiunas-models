#!/usr/bin/env python3
"""O portao `sp|vb` e cego para evento puro de pressao. Relaxa-lo custa?

DIAGNOSTICO (caso_2411.py). Em 24/11/2025 -- parada de 43 h com alarme de pressao
baixa no header de oleo lubrificante, o unico entre 14 paradas longas fora do
alvo com alarme de SAUDE MECANICA -- o canal `p` fica a 6,14x o limiar por 48 h
seguidas e o detector nao dispara, porque o portao exige `sp OU vb` e os dois
estao mudos (0,25x e 0,26x).

O portao foi desenhado para eventos de mancal, e nenhum dos 8 alvos e evento puro
de pressao -- por isso a cegueira nunca apareceu. Testa relaxar o portao para
aceitar tambem pressao FORTE, e mede o que isso custa nos 8 que temos.
"""
from __future__ import annotations
import itertools
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
T2411 = pd.Timestamp("2025-11-24 09:22", tz="UTC")
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
p = {k: (dict(P["lo"]) if k == "lo" else P[k]) for k in P}; p["frac"] = 0.0
A = {c: canal(c, p["lo"][c]) for c in SIN}
B = {c: canal(c, p["hi"] if c != "vb" else p["kvhi"]) for c in SIN}
KH = {"t": p["hi"], "p": p["hi"], "sp": p["hi"], "vb": p["kvhi"]}
RZ = {c: EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN}
F = pd.concat([RZ[c] for c in SIN], axis=1).max(axis=1).to_numpy()

def mede(al):
    eps = AV.episodios(al); banda, leads, ini = 0, [], 0
    for t in alvo:
        c = [a for a, _ in eps if t-JAN <= a <= t-pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t-max(c)).total_seconds()/3600)
        if any(t-JAN <= a <= t for a, _ in eps): ini += 1
    jw = [(t-JAN, t) for t in alvo]; fp = h = 0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw): continue
        if len(paradas[(paradas.ini >= a) & (paradas.ini <= b+JAN)]): continue
        fp += 1; h += (b-a).total_seconds()/3600
    pega2411 = any(T2411-JAN <= a <= T2411 for a, _ in eps)
    return banda, ini, AV.avalia(al, alvo, mask)["det"], fp/meses, h/meses, \
           (np.mean(leads) if leads else np.nan), pega2411

def roda(porta):
    vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
    vB = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask & porta
    vv = quebra_idade((vA | vB).to_numpy(), F, p["abs_"], p["idade"])
    return mede(pos_dur_esc(pd.Series(vv, index=idx), p["refrat"], F,
                            p["abs_"], p["idade"], p["dur_esc"]))

print(f"{'portao':>40} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}"
      f"{'lead':>9}{'24/11':>8}")
print("-" * 108)
r0 = roda(B["sp"] | B["vb"])
print(f"{'sp|vb (atual)':>40} | {r0[0]:5d}/8{r0[1]:7d}/8{r0[2]:6d}/8{r0[3]:10.3f}"
      f"{r0[4]:9.1f}{r0[5]:8.1f}h{('SIM' if r0[6] else 'nao'):>8}")
print("-" * 108)
for nome, porta in [
        ("sp|vb OU p>=3x",   B["sp"] | B["vb"] | ((RZ["p"] >= 3.0) & mask)),
        ("sp|vb OU p>=5x",   B["sp"] | B["vb"] | ((RZ["p"] >= 5.0) & mask)),
        ("sp|vb OU p>=8x",   B["sp"] | B["vb"] | ((RZ["p"] >= 8.0) & mask)),
        ("sp|vb OU t>=3x",   B["sp"] | B["vb"] | ((RZ["t"] >= 3.0) & mask)),
        ("sp|vb|p (canal p cru)", B["sp"] | B["vb"] | B["p"]),
        ("sem portao",       pd.Series(True, index=idx))]:
    r = roda(porta)
    d = ""
    if r[0] != r0[0] or abs(r[3]-r0[3]) > 1e-6: d = "  <<<"
    print(f"{nome:>40} | {r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}"
          f"{r[4]:9.1f}{(r[5] if np.isfinite(r[5]) else 0):8.1f}h"
          f"{('SIM' if r[6] else 'nao'):>8}{d}")
