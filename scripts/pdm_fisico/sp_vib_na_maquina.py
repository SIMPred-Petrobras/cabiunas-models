#!/usr/bin/env python3
"""`sp_vib` na maquina: substitui o `sp`, soma como 5o, e a ablacao (Testes 1 e 3)."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, partes, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import KB, KV, paradas_reais_2h
from sp_vibracao import constroi

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()
FRAC, IDADE, ABS, REFRAT_E, DUR_ESC = 0.03, 96, 20, 72, 60
SV = constroi(2400)
ON = partes(KB, KV)
CACHE = {}

def canal(c, k):
    if (c, k) in CACHE: return CACHE[(c, k)]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    CACHE[(c, k)] = (deg | cu) & mask
    return CACHE[(c, k)]

def pos_simples(v, rf, dm):
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

print(f"{'configuracao':>46} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 110)
b0 = mede(pos_simples(pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx)
                      & mask & (ON["sp"] | ON["vb"]), REFRAT_H, DUR_MIN))
print(f"{'os nossos 4 (publicado)':>46} | {b0[0]:5d}/8{b0[1]:7d}/8{b0[2]:6d}/8"
      f"{b0[3]:10.3f}{b0[4]:9.1f}{b0[5]:8.1f}h")
print("-" * 110)
melhor = None
for kv_sv, conj, nv in itertools.product((4, 5, 6, 8), ("substitui sp", "5o canal",
                                                        "substitui sp E p"), (2, 3)):
    sv = ((SV >= kv_sv) & mask).fillna(False)
    if conj == "substitui sp":
        ch = {"t": ON["t"], "p": ON["p"], "sv": sv, "vb": ON["vb"]}
        porta = sv | ON["vb"]
    elif conj == "5o canal":
        ch = {**{c: ON[c] for c in SIN}, "sv": sv}
        porta = ON["sp"] | ON["vb"] | sv
    else:
        ch = {"t": ON["t"], "sv": sv, "vb": ON["vb"]}
        porta = sv | ON["vb"]
    v = pd.Series(sum(ch[k].astype(int) for k in ch) >= nv, index=idx) & mask & porta
    r = mede(pos_simples(v, REFRAT_H, DUR_MIN))
    if r[2] == 8 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
        melhor = ((r[0], -r[3], -r[4]), kv_sv, conj, nv, r)
    if r[0] > b0[0] and r[2] >= 7:
        print(f"{f'sp_vib>={kv_sv}, {conj}, voto>={nv}':>46} | {r[0]:5d}/8{r[1]:7d}/8"
              f"{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
              f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 110)
if melhor:
    _, kv_sv, conj, nv, r = melhor
    print(f"  MELHOR com det=8/8: sp_vib>={kv_sv}, {conj}, voto>={nv}")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
print(f"  melhor ja medido (dois niveis por canal): banda 5/8, 0,517 FP/mes, 7,8 h/mes")
