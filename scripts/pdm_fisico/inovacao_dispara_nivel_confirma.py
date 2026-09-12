#!/usr/bin/env python3
"""A sintese: INOVACAO dispara, NIVEL confirma.

Cada formulacao e boa numa coisa diferente, e as duas foram medidas:
  inovacao : esparsa (1-4% duty), razao ate 3,26x, nasce perto do evento --
             mas sozinha nao corrobora (voto>=2 quase nunca fecha), entao
             precisa de voto>=1 e o FP explode para 3,36/mes
  nivel    : densa (24-52% duty), corrobora bem, mas os episodios nascem 52 a
             670 h antes -- a causa raiz medida em onde_morre.py

Estrutura: o alarme abre na INOVACAO (que da o instante) e so vale se o NIVEL
corroborar (que da a especificidade). O nivel nao abre episodio nenhum -- so
autoriza.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h
from inovacao import INOV

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
NIV = partes(KB, KV)

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
        c = [a for a, _ in eps if t-JAN <= a <= t-pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t-max(c)).total_seconds()/3600)
        if any(t-JAN <= a <= t for a, _ in eps): ini += 1
    jw = [(t-JAN, t) for t in alvo]; fp = h = 0
    for a, b in eps:
        if any(a <= t1 and b >= t0 for t0, t1 in jw): continue
        if len(paradas[(paradas.ini >= a) & (paradas.ini <= b+JAN)]): continue
        fp += 1; h += (b-a).total_seconds()/3600
    return banda, ini, AV.avalia(al, alvo, mask)["det"], fp/meses, h/meses, \
           (np.mean(leads) if leads else np.nan), len(eps)

print(f"{'configuracao':>52} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}"
      f"{'h/mes':>9}{'lead':>9}{'eps':>6}")
print("-" * 118)
print(f"{'publicado':>52} | {3:5d}/8{4:7d}/8{8:6d}/8{0.517:10.3f}{7.1:9.1f}{12.5:8.1f}h{20:6d}")
print(f"{'melhor atual (dois niveis de nivel)':>52} | {5:5d}/8{6:7d}/8{8:6d}/8"
      f"{0.344:10.3f}{6.6:9.1f}{19.7:8.1f}h{21:6d}")
print("-" * 118)
melhor = None
n_niv = sum(NIV[c].astype(int) for c in SIN)
for hl, k, nv_i, nv_n in itertools.product(("30min", "2h", "8h"), (6, 8, 10, 15),
                                           (1, 2), (1, 2, 3)):
    I = {c: ((INOV[(c, hl)] >= k) & mask).fillna(False) for c in SIN}
    disp = pd.Series(sum(I[c].astype(int) for c in SIN) >= nv_i, index=idx) & mask
    for conf in ("voto de nivel", "voto + portao"):
        ok = pd.Series(n_niv >= nv_n, index=idx)
        if conf == "voto + portao": ok = ok & (NIV["sp"] | NIV["vb"])
        v = disp & ok & mask
        r = mede(pos(v, REFRAT_H, DUR_MIN))
        if r[2] >= 7 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
            melhor = ((r[0], -r[3], -r[4]), hl, k, nv_i, nv_n, conf, r)
        if r[0] >= 5 and r[2] == 8 and r[3] <= 0.6:
            print(f"{f'inov hl={hl} k={k} v>={nv_i} & nivel>={nv_n} ({conf})':>52} | "
                  f"{r[0]:5d}/8{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
                  f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h{r[6]:6d}")
print("-" * 118)
if melhor:
    _, hl, k, nv_i, nv_n, conf, r = melhor
    print(f"  MELHOR: inovacao hl={hl} k={k} voto>={nv_i}, confirmada por nivel>={nv_n} ({conf})")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, det {r[2]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f}h, {r[6]} episodios")
