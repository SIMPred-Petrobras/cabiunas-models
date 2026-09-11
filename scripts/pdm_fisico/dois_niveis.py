#!/usr/bin/env python3
"""IDEIA 5 -- gatilho de DOIS NIVEIS (sensivel-corroborado OU especifico).

DESCOBERTA QUE MOTIVA (troca_limiar_voto.py): com kb=1,0 e voto>=3 o lead medio
sobe de 14,4 h para 27,7 h -- na faixa do Diego (23,8 h) -- mas a cobertura cai
para 5/8. Com kb=1,7 e voto>=2 a cobertura e 8/8 mas o lead e 14,4 h.

Os dois regimes acertam eventos por caminhos diferentes: o sensivel pega a
deriva cedo exigindo corroboracao ampla; o especifico pega a excursao forte com
poucos canais. Nunca testamos a UNIAO.

  nivel A (precoce)   : voto >= 3 canais em limiar BAIXO  (kb_lo)
  nivel B (especifico): voto >= 2 canais em limiar ALTO   (kb_hi) + portao sp|vb
  alarme = A ou B

HIPOTESE: cobertura do B com o lead do A. O custo extra do A deve ser pequeno
porque exigir 3 de 4 canais simultaneos e raro.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, pos, mask, idx, alvo
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, DUR_MIN
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

reset = (~mask).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
TMIN, TMAX = 4.0, 48.0
CACHE = {}


def canais(kb, kv):
    if (kb, kv) in CACHE:
        return CACHE[(kb, kv)]
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    out = {}
    for c in SIN:
        thr = BASE[c]*K[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    CACHE[(kb, kv)] = out
    return out


def mede(al):
    eps = AV.episodios(al)
    banda, leads, ini = 0, [], 0
    for t in alvo:
        c = [a for a, _ in eps
             if t - pd.Timedelta(hours=TMAX) <= a <= t - pd.Timedelta(hours=TMIN)]
        if c:
            banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if any(t - pd.Timedelta(hours=TMAX) <= a <= t for a, _ in eps):
            ini += 1
    m = AV.avalia(al, alvo, mask)
    cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    return banda, ini, m["det"], nfp/meses, h/meses, (np.mean(leads) if leads else np.nan)


LO = [0.6, 0.8, 1.0, 1.2, 1.4]
HI = [1.4, 1.7, 2.0]
KVs = [1.8, 2.2, 2.8]
print(f"UNIAO DOS DOIS NIVEIS -- banda acionavel [{TMIN:.0f} h, {TMAX:.0f} h]")
print("=" * 104)
print(f"{'kb_lo':>6} {'kb_hi':>6} {'kv':>5} | {'banda':>7} {'inicio':>7} {'det':>6} "
      f"{'FP/mes':>9} {'h/mes':>8} {'lead':>8}")
print("-" * 104)
res = []
for lo, hi, kv in itertools.product(LO, HI, KVs):
    if lo >= hi:
        continue
    A = canais(lo, kv); B = canais(hi, kv)
    nsA = sum(A[c].astype(int) for c in SIN); nsB = sum(B[c].astype(int) for c in SIN)
    vA = pd.Series(nsA >= 3, index=idx) & mask
    vB = (pd.Series(nsB >= 2, index=idx) & mask & (B["sp"] | B["vb"]))
    v = (vA | vB)
    K = {"t": hi, "p": hi, "sp": hi, "vb": kv}
    F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
    al = pos(pd.Series(corta_rearma(v.to_numpy(), F, 0.03), index=idx),
             nsB, 72, DUR_MIN, False)
    r = mede(al)
    res.append((r[0], -r[3], lo, hi, kv) + r)
res.sort(reverse=True)
for _, _, lo, hi, kv, banda, ini, det, fp, h, lm in res[:14]:
    marca = "  <<< bate o atual" if banda > 4 else ""
    print(f"{lo:6.1f} {hi:6.1f} {kv:5.1f} | {banda:5d}/8 {ini:5d}/8 {det:5d}/8 "
          f"{fp:9.3f} {h:8.1f} {lm:7.1f}h{marca}")
print("-" * 104)
print("  atual (um nivel, voto>=2, kb=1,7 kv=2,2): banda 4/8, inicio 6/8, det 8/8, "
      "0,775 FP/mes, lead 14,4 h")
print("  Diego: banda 7/8, lead 23,8 h, 2,88 FP/mes")
