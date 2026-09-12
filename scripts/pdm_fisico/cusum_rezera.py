#!/usr/bin/env python3
"""CUSUM com REZERO APOS SINALIZAR -- a pratica padrao que nunca aplicamos.

DIAGNOSTICO. O CUSUM responde pela maior parte do ciclo de trabalho dos nossos
canais -- o de pressao vai de 10,9% (so nivel) para 34,4%, e 23,4% do tempo ele
esta aceso com o NIVEL JA ABAIXO do limiar. O acumulador so zera quando a
mascara quebra; uma excursao forte o deixa acima de H por semanas.

Em carta de controle CUSUM, a pratica padrao e **rezerar S ao sinalizar** (ou dar
headstart). Nos nunca fizemos. E DIFERENTE do CUSUM vazante que ja foi refutado
([[residuo-condicionado-a-carga-refutado]] nao, o outro -- `cusum_vazante.py`):
aquele decaia SEMPRE e enfraquecia a deteccao de deriva fraca, que e justamente
o que o CUSUM entrega. Rezerar so age DEPOIS de o alarme ja ter disparado, entao
nao custa sensibilidade -- custa persistencia, que e o que sobra demais.

Variantes: rezera para 0, para H/2 (headstart), e teto S <= f*H.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, CARGA, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()
K = {"t": KB, "p": KB, "sp": KB, "vb": KV}


def cusum_rz(x, rst, H, modo, par):
    """modo: 'nunca' (atual) | 'zero' | 'head' (rezera para par*H) | 'teto' (S<=par*H)."""
    S = np.empty(len(x)); a = 0.0
    for i in range(len(x)):
        if rst[i]:
            a = a * CARGA
        else:
            a = max(0.0, a + x[i])
            if modo == "teto":
                a = min(a, par * H)
        S[i] = a
        if modo in ("zero", "head") and a > H:
            a = 0.0 if modo == "zero" else par * H
    return S


def monta(modo, par):
    ON = {}
    for c in SIN:
        thr = BASE[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        x = ((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
        cu = pd.Series(cusum_rz(x, reset, H_CUSUM, modo, par) > H_CUSUM, index=idx)
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
    return fin & sel, ON


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
    d = AV.avalia(al, alvo, mask)["det"]
    return banda, ini, d, fp/meses, h/meses, (np.mean(leads) if leads else np.nan), len(eps)


print("CUSUM: REZERO AO SINALIZAR")
print("=" * 108)
print(f"{'modo':>18} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}"
      f"{'lead':>9}{'eps':>6} | duty vb")
print("-" * 108)
melhor = None
for modo, par, rot in [("nunca", 0, "atual (nunca zera)"),
                       ("zero", 0, "rezera para 0"),
                       ("head", 0.25, "rezera para 0,25H"),
                       ("head", 0.50, "rezera para 0,50H"),
                       ("head", 0.75, "rezera para 0,75H"),
                       ("teto", 1.5, "teto S <= 1,5H"),
                       ("teto", 2.0, "teto S <= 2H"),
                       ("teto", 4.0, "teto S <= 4H")]:
    al, ON = monta(modo, par)
    b, i, d, fp, h, lm, ne = mede(al)
    duty = 100*float(ON["vb"].sum())/float(mask.sum())
    mk = "  <<<" if d == 8 and (melhor is None or (b, -fp) > melhor[0]) else ""
    if d == 8 and (melhor is None or (b, -fp) > melhor[0]):
        melhor = ((b, -fp), rot, b, i, d, fp, h, lm)
    print(f"{rot:>18} | {b:5d}/8{i:7d}/8{d:6d}/8{fp:10.3f}{h:9.1f}"
          f"{(lm if np.isfinite(lm) else 0):8.1f}h{ne:6d} | {duty:6.1f}%{mk}")
print("-" * 108)
if melhor:
    _, rot, b, i, d, fp, h, lm = melhor
    print(f"  melhor com det=8/8: {rot} -> banda {b}/8, inicio {i}/8, "
          f"{fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
print("  publicado: banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes, 7,1 h/mes, lead 12,5 h")
