#!/usr/bin/env python3
"""Histerese: limiar de ACENDER maior que o de APAGAR.

Diagnostico que motiva (episodios_longos_tendencia.py): o episodio de 670 h antes
de 26/02/2026 se sustenta por dois canais parados em 1,02x o limiar, enquanto o
canal que o abriu (p) despenca de 22,20x para 1,02x. E alarme travado por residuo.

Hoje o mesmo limiar acende e mantem. A pratica de gerenciamento de alarme manda
separar os dois: acende em thr, mantem enquanto acima de thr/H. Com H > 1 o
alarme travado em 1,02x cai, e as deteccoes reais -- que vivem em 2x a 8x o
limiar -- nao sentem.

Isso NAO e o teto de permanencia (refutado duas vezes): aquele e por tempo, este
por margem de escore. O teto silencia um sinal ainda forte; a histerese so solta
um sinal que ja voltou para perto do limiar.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, pos, mask, idx, alvo
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from plota_estilo_francisco import (KB, KV, paradas_reais_2h, classifica_regra_c)
from regra_inicio_varredura import avalia_inicio

reset = (~mask).to_numpy()


def trava(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Liga em `hi`, mantem enquanto `lo`, desliga quando `lo` cai."""
    s = np.where(hi, 1.0, np.where(~lo, 0.0, np.nan))
    return pd.Series(s).ffill().fillna(0.0).to_numpy() > 0.5


def partes_hist(kb, kv, H):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    out = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        E = EW[c].where(mask)
        acc = pd.Series(trava((E > thr).to_numpy(), (E > thr / H).to_numpy()), index=idx)
        deg = (acc.astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out


paradas = paradas_reais_2h()
print("HISTERESE -- acende em thr, mantem ate thr/H")
print("=" * 104)
print(f"{'H':>6} | {'det':>5} {'FP':>4} {'NEUTRO':>7} {'FP/mes':>8} {'h/mes':>8} "
      f"{'lead':>7} | {'det_ini':>8} {'ep 26/02':>10}")
print("-" * 104)
for H in [1.0, 1.05, 1.10, 1.15, 1.25, 1.40, 1.60, 2.00]:
    ON = partes_hist(KB, KV, H)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    eps = AV.episodios(al)
    cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for *_, k, _ in [(a, b, k, l) for a, b, k, l in cls] if k == "FP")
    nnt = sum(1 for a, b, k, l in cls if k == "NEUTRO")
    fpc = nfp / (float(mask.sum()) * 2 / 60.0 / 730.0)
    hfc = sum((b - a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP") \
          / (float(mask.sum()) * 2 / 60.0 / 730.0)
    # o episodio que cobre 26/02/2026
    t = alvo.iloc[-1]; JAN = pd.Timedelta(hours=48)
    cand = [(a, b) for a, b in eps if a <= t and b >= t - JAN]
    dur = f"{(cand[0][1]-cand[0][0]).total_seconds()/3600:.0f}h" if cand else "--"
    print(f"{H:6.2f} | {m['det']:4d}/8 {nfp:4d} {nnt:7d} {fpc:8.3f} {hfc:8.1f} "
          f"{m['lead_med']:6.1f}h | {mi['det']:6d}/8 {dur:>10}")
print("-" * 104)
print("  H = 1,00 e o ponto de producao de hoje (esperado 8/8, 6 FP, 0,517, 7,15, 29,0)")
