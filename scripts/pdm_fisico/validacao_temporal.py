#!/usr/bin/env python3
"""VALIDACAO TEMPORAL do v2 -- ajustar no passado, testar no futuro.

POR QUE. O v2 foi ajustado olhando os 8 eventos e validado por LOEO. Mas o LOEO
EMBARALHA os eventos: ao avaliar o evento 3 ele usa os eventos 6, 7 e 8, que no
tempo real ainda nao tinham acontecido. E otimista por construcao.

O teste que imita o uso real e o holdout TEMPORAL: escolher a configuracao
olhando so os eventos ANTIGOS e medir nos NOVOS. E o unico que responde "se
tivessemos calibrado em 2025, o detector teria funcionado em 2026?".

Se o v2 sobreviver, e muito mais defensavel do que o LOEO sugere. Se nao
sobreviver, o ganho e ajuste -- e e melhor saber agora.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, sel
from publica_clearml import (SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, K, K_LO,
                             VOTO_LO, VOTO_HI, REFRAT_H, REFRAT_V2, DUR_MIN,
                             ESC_IDADE, ESC_ABS, ESC_DUR)
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h()
reset = (~mask).to_numpy()
CACHE = {}

def canal(c, k):
    if (c, k) in CACHE: return CACHE[(c, k)]
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    CACHE[(c, k)] = (deg | cu) & mask
    return CACHE[(c, k)]

N_GAP = int(pd.Timedelta(hours=AV.GAP_EP_H)/pd.Timedelta("2min")) + 1

def detecta(klo, khi, kvhi, refrat, v2=True):
    A = {c: canal(c, klo[c]) for c in SIN}
    B = {c: canal(c, khi if c != "vb" else kvhi) for c in SIN}
    KH = {"t": khi, "p": khi, "sp": khi, "vb": kvhi}
    if v2:
        vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= VOTO_LO, index=idx) & mask
        vB = (pd.Series(sum(B[c].astype(int) for c in SIN) >= VOTO_HI, index=idx)
              & mask & (B["sp"] | B["vb"]))
        voto = (vA | vB)
        F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1)
        f = F.fillna(0.0).to_numpy(); v = voto.to_numpy().copy()
        n_id = int(ESC_IDADE*30); dentro, ini, ja = False, 0, False
        for i in range(len(v)):
            if not v[i]: dentro, ja = False, False; continue
            ac = f[i] > ESC_ABS
            if not dentro: dentro, ini, ja = True, i, ac; continue
            if ac and not ja and (i-ini) >= n_id:
                v[max(ini+1, i-N_GAP):i] = False; ini = i
            ja = ac
        voto = pd.Series(v, index=idx); forca = F
    else:
        voto = (pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx)
                & mask & (B["sp"] | B["vb"]))
        forca = None
    al = pd.Series(False, index=idx); bloq = None; ini_b = None; fortes = []
    for a, b in AV.episodios(voto):
        forte = bool(forca is not None and float(forca.loc[a:b].max()) > ESC_ABS)
        velho = ini_b is not None and (a-ini_b).total_seconds()/3600 >= ESC_IDADE
        if bloq is not None and a <= bloq and not (forte and velho): continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat); ini_b = a
        if forte: fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        d = (b-a).total_seconds()/60 + 2
        if (any(x >= a and y <= b for x, y in fortes) and d >= ESC_DUR) or d >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel

def mede(al, evs, t0=None, t1=None):
    eps = AV.episodios(al)
    if t0 is not None:
        eps = [(a, b) for a, b in eps if a >= t0 and (t1 is None or a <= t1)]
    banda = sum(1 for t in evs if any(t-JAN <= a <= t-pd.Timedelta(hours=TMIN) for a,_ in eps))
    ini = sum(1 for t in evs if any(t-JAN <= a <= t for a, _ in eps))
    jw = [(t-JAN, t) for t in evs]; fp = 0
    for a, b in eps:
        if any(a <= y and b >= x for x, y in jw): continue
        if len(paradas[(paradas.ini >= a) & (paradas.ini <= b+JAN)]): continue
        fp += 1
    m = mask if t0 is None else mask.loc[t0:t1 if t1 is not None else idx[-1]]
    meses = float(m.sum())*2/60/730
    return banda, ini, fp, fp/max(meses, 1e-9), meses

CORTE = pd.Timestamp("2025-07-01", tz="UTC")
ant = [t for t in alvo if t < CORTE]; dep = [t for t in alvo if t >= CORTE]
print(f"CORTE TEMPORAL em {CORTE:%d/%m/%Y}")
print(f"  ajuste (passado): {len(ant)} eventos -- {', '.join(t.strftime('%d/%m/%y') for t in ant)}")
print(f"  teste  (futuro) : {len(dep)} eventos -- {', '.join(t.strftime('%d/%m/%y') for t in dep)}")

print("\n\nESCOLHA DA CONFIGURACAO OLHANDO SO O PASSADO")
print("=" * 96)
GR_LO = [0.9, 1.0, 1.1, 1.3]
melhor = None
for kt, kp, ks, kv in itertools.product(GR_LO, [0.7, 1.0], [0.9, 1.2], [1.6, 1.8, 2.2]):
    klo = {"t": kt, "p": kp, "sp": ks, "vb": kv}
    al = detecta(klo, 1.7, 2.2, REFRAT_V2)
    b, i, fp, fpm, _ = mede(al, ant, idx[0], CORTE)
    if melhor is None or (b, -fpm) > melhor[0]:
        melhor = ((b, -fpm), klo, b, i, fpm)
_, klo_esc, b_tr, i_tr, fp_tr = melhor
print(f"  escolhida: {klo_esc}")
print(f"  no TREINO: banda {b_tr}/{len(ant)}, inicio {i_tr}/{len(ant)}, {fp_tr:.3f} FP/mes")
print(f"  o v2 publicado usa: {K_LO}")
print(f"  bateu com o publicado? {'SIM' if klo_esc == K_LO else 'NAO'}")

print("\n\nTESTE NO FUTURO -- os 3 eventos nunca vistos")
print("=" * 96)
print(f"{'configuracao':>34}{'banda':>9}{'inicio':>9}{'FP/mes':>10}")
print("-" * 96)
for rot, klo, khi, kvhi, rf, v2 in [
        ("v1 (um nivel)", K_LO, 1.7, 2.2, REFRAT_H, False),
        ("v2 publicado", K_LO, 1.7, 2.2, REFRAT_V2, True),
        ("v2 escolhido SO no passado", klo_esc, 1.7, 2.2, REFRAT_V2, True)]:
    al = detecta(klo, khi, kvhi, rf, v2)
    b, i, fp, fpm, _ = mede(al, dep, CORTE, None)
    print(f"{rot:>34}{b:7d}/{len(dep)}{i:7d}/{len(dep)}{fpm:10.3f}")
print("-" * 96)
print("  se o v2 mantem a vantagem sobre o v1 NO FUTURO, o ganho nao e ajuste")
