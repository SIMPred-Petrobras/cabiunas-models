#!/usr/bin/env python3
"""Duas premissas nunca reexaminadas sob a REGUA DE INICIO: blackout e halflife.

POR QUE AGORA. A memoria registra que 9 dos 15 FP nasciam em `dist_partida` =
6,4667 h CRAVADO -- o primeiro instante em que a mascara libera. Ou seja: o
blackout nao elimina episodios, **empurra o inicio deles para um offset fixo**.
Sob a regua "de pe" isso era irrelevante (o que importava era estar aceso); sob a
regua de INICIO e exatamente um mecanismo de lead erratico -- o defeito medido em
[[banda-de-acionabilidade]].

`blackout-6h-e-o-orcamento` refutou encurtar o blackout, mas MEDIU SOB A REGUA DE
PE. Mesma situacao do teto de permanencia, que tambem parecia refutado e mudou de
sinal quando reavaliado na regua certa.

E os `halflife` do EWMA -- {t:1h, p:1h, sp:30min, vb:30min} -- nunca foram
varridos em lugar nenhum. Eles controlam DIRETAMENTE quando o sinal cruza o
limiar, entao sao o botao mais direto de lead que existe no detector.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import g, idx, alvo, op, estavel, part, sel, T0
from publica_clearml import (SIN, BASE, GRID, SUSTAIN, KAPPA, H_CUSUM,
                             REFRAT_H, DUR_MIN)
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h()
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}

z = np.load("piso_fisico_cache.npz")
spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
with np.errstate(invalid="ignore", divide="ignore"):
    Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
vbv = np.full(len(idx), np.nan)
vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
vbv[~np.isfinite(vbv)] = np.nan
CRU = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)


def monta(bl_h, hl):
    n_bl = max(int(pd.Timedelta(hours=bl_h) / pd.Timedelta(GRID)), 1)
    blk = part.rolling(n_bl, min_periods=1).max().astype(bool) if bl_h > 0 \
          else pd.Series(False, index=idx)
    mask = (estavel & ~blk) & sel
    reset = ((~mask) | part).to_numpy()
    EW = {c: CRU[c].ewm(halflife=pd.Timedelta(hl[c]), times=idx).mean() for c in SIN}
    ON = {}
    for c in SIN:
        thr = BASE[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        ON[c] = (deg | cu) & mask
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(v):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b-a).total_seconds()/60 + 2 >= DUR_MIN: fin.loc[a:b] = True
    return fin & sel, mask


def mede(al, mask):
    eps = AV.episodios(al); meses = float(mask.sum())*2/60.0/730.0
    banda, leads, ini = 0, [], 0
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
    return banda, ini, fp/meses, h/meses, (np.mean(leads) if leads else np.nan), len(eps)


HL0 = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
print("A. BLACKOUT sob a REGUA DE INICIO  (halflife original)")
print("=" * 92)
print(f"{'blackout':>9} | {'banda':>7}{'inicio':>8}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}{'eps':>6}")
print("-" * 92)
for bl in [0, 0.5, 1, 2, 3, 4, 6, 9, 12]:
    al, mk = monta(bl, HL0)
    b, i, fp, h, lm, ne = mede(al, mk)
    m = "  <<< publicado" if bl == 6 else ""
    print(f"{bl:8.1f}h | {b:5d}/8{i:7d}/8{fp:10.3f}{h:9.1f}"
          f"{(lm if np.isfinite(lm) else 0):8.1f}h{ne:6d}{m}")

print("\n\nB. HALFLIFE sob a REGUA DE INICIO  (blackout de 6 h)")
print("=" * 92)
print(f"{'hl t/p':>8}{'hl sp/vb':>10} | {'banda':>7}{'inicio':>8}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 92)
melhor = None
for a_, b_ in itertools.product(["15min", "30min", "1h", "2h", "4h", "8h"],
                                ["10min", "30min", "1h", "2h"]):
    hl = {"t": a_, "p": a_, "sp": b_, "vb": b_}
    al, mk = monta(6, hl)
    bd, i, fp, h, lm, ne = mede(al, mk)
    if melhor is None or (bd, -fp) > melhor[0]:
        melhor = ((bd, -fp), a_, b_, bd, i, fp, h, lm)
    if bd >= 3:
        mrk = "  <<< publicado" if (a_, b_) == ("1h", "30min") else ""
        print(f"{a_:>8}{b_:>10} | {bd:5d}/8{i:7d}/8{fp:10.3f}{h:9.1f}"
              f"{(lm if np.isfinite(lm) else 0):8.1f}h{mrk}")
print("-" * 92)
_, a_, b_, bd, i, fp, h, lm = melhor
print(f"  melhor: hl={a_}/{b_} -> banda {bd}/8, inicio {i}/8, {fp:.3f} FP/mes, "
      f"{h:.1f} h/mes, lead {lm:.1f} h")
print("  publicado: banda 3/8, inicio 4/8, 0,517 FP/mes, 7,1 h/mes, lead 12,5 h")
