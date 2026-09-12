#!/usr/bin/env python3
"""O `vb` sozinho, com limiar esparso -- o voto e que esta custando?

DIAGNOSTICO (11/09/2026). O canal `vb` cobre 7/8 na banda acionavel; o detector
inteiro cobre 3/8. O `vb` fica aceso 52% do tempo (limiar em p69), entao nunca
dispara sozinho -- precisa de um segundo voto. E os candidatos a segundo voto,
`sp` e `p`, tem razao ABAIXO do acaso na banda (0,55x e 0,83x).

Hipotese: subir o limiar do vb ate ele ser esparso e deixa-lo disparar sozinho
entrega mais que o voto de dois com canais anti-informativos.

`limiar-nao-e-alavanca-de-custo` mediu isso SOB A REGUA DE PE. Aqui e a regua de
inicio e a banda acionavel -- mesma situacao do teto de permanencia, que mudou de
sinal quando reavaliado na regua certa.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, partes, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from plota_estilo_francisco import KB, KV, paradas_reais_2h

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()

def canal(c, k, com_cusum=True):
    thr = BASE[c]*k; E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    if not com_cusum:
        return deg & mask
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    return (deg | cu) & mask

def pos(v, rf, dm):
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

ON = partes(KB, KV)
b0 = mede(pos(pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx)
              & mask & (ON["sp"] | ON["vb"]), REFRAT_H, DUR_MIN))
print(f"{'configuracao':>40} | {'duty vb':>8}{'banda':>7}{'inicio':>8}{'det':>7}"
      f"{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 108)
print(f"{'os nossos 4, voto>=2 + portao (publicado)':>40} | "
      f"{100*float(ON['vb'].sum())/float(mask.sum()):7.1f}%{b0[0]:6d}/8{b0[1]:7d}/8"
      f"{b0[2]:6d}/8{b0[3]:10.3f}{b0[4]:9.1f}{b0[5]:8.1f}h")
print("-" * 108)
melhor = None
for kv, cus, rf, dm in itertools.product(
        [2.2, 3.0, 4.0, 5.5, 7.0, 9.0, 12.0], (True, False), (48, 72), (120, 240)):
    vb = canal("vb", kv, cus)
    duty = 100*float(vb.sum())/float(mask.sum())
    r = mede(pos(vb, rf, dm))
    if r[2] == 8 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
        melhor = ((r[0], -r[3], -r[4]), kv, cus, rf, dm, r, duty)
    if r[0] >= b0[0] and r[2] >= 7:
        print(f"{f'vb SOZINHO k={kv}, cusum={cus}, refr={rf}h, dur={dm}min':>40} | "
              f"{duty:7.1f}%{r[0]:6d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
              f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 108)
if melhor:
    _, kv, cus, rf, dm, r, duty = melhor
    print(f"  MELHOR com det=8/8: vb sozinho k={kv}, cusum={cus}, refrat={rf}h, dur={dm}min")
    print(f"     duty {duty:.1f}% -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f} h")
else:
    print("  nenhuma configuracao de vb sozinho mantem det = 8/8")
