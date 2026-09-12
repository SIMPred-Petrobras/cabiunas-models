#!/usr/bin/env python3
"""O residuo condicionado a carga melhora? Mede contaminacao E deteccao."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo, g
from publica_clearml import (SIN, BASE, HL, SUSTAIN, KAPPA, H_CUSUM,
                             REFRAT_H, DUR_MIN)
from blackout_curto import cusum
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
reset = (~mask).to_numpy()
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
carga = g["T5_AVG_A"].where(mask)
dcarga = carga.diff().abs().rolling(30, min_periods=5).mean()
q = pd.qcut(dcarga.dropna(), 5, labels=["q1", "q2", "q3", "q4", "q5"])


def sinais(cache):
    z = np.load(cache)
    spv = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbv = np.full(len(idx), np.nan)
    vbv[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbv[~np.isfinite(vbv)] = np.nan
    cru = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": spv, "vb": vbv}, index=idx)
    return {c: cru[c].ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean() for c in SIN}


def canais(EW, K):
    out = {}
    for c in SIN:
        thr = BASE[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out


def pos(voto, refrat_h, dur_min):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(voto):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b-a).total_seconds()/60 + 2 >= dur_min: fin.loc[a:b] = True
    return fin


def mede(al):
    eps = AV.episodios(al); banda, leads, ini = 0, [], 0
    for t in alvo:
        c = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if any(t - JAN <= a <= t for a, _ in eps): ini += 1
    m = AV.avalia(al, alvo, mask); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    return banda, ini, m["det"], nfp/meses, h/meses, (np.mean(leads) if leads else np.nan)


E0 = sinais("piso_fisico_cache.npz")
E1 = sinais("piso_fisico_carga_cache.npz")
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}

print("1. CONTAMINACAO POR CARGA -- razao E/limiar por quintil de |dcarga|")
print("=" * 86)
print(f"{'quintil':>9} | " + "".join(f"{c+' orig':>12}{c+' cond':>12}" for c in ("t", "p")))
print("-" * 86)
for lab in ["q1", "q2", "q3", "q4", "q5"]:
    m_ = q[q == lab].index; linha = ""
    for c in ("t", "p"):
        for E in (E0, E1):
            linha += f"{float((E[c].where(mask)/(BASE[c]*K[c])).reindex(m_).mean()):12.3f}"
    print(f"{lab:>9} | {linha}")
print("-" * 86)
for c in ("t", "p"):
    r = []
    for E in (E0, E1):
        s = E[c].where(mask)/(BASE[c]*K[c])
        a = float(s.reindex(q[q == 'q1'].index).mean()); b = float(s.reindex(q[q == 'q5'].index).mean())
        r.append(b/a if a else np.nan)
    print(f"  razao q5/q1 do canal {c}: original {r[0]:7.2f}x  ->  condicionado {r[1]:6.2f}x")

print("\n\n2. DETECCAO -- mesmo ponto de operacao, os dois caches")
print("=" * 96)
print(f"{'':<22}{'banda':>8}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 96)
for rot, E in (("original", E0), ("condicionado", E1)):
    ON = canais(E, K); ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    b, i, d, fp, h, lm = mede(pos(v, REFRAT_H, DUR_MIN))
    print(f"{rot:<22}{b:6d}/8{i:7d}/8{d:6d}/8{fp:10.3f}{h:9.1f}"
          f"{(lm if np.isfinite(lm) else 0):8.1f}h")

print("\n\n3. O CONDICIONADO PRECISA DE OUTRO LIMIAR?  varredura de k")
print("=" * 96)
print(f"{'kb':>5}{'kv':>5} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 96)
melhor = None
for kb, kv in itertools.product([0.7, 1.0, 1.3, 1.7, 2.2, 2.8], [1.4, 1.8, 2.2, 2.8]):
    ON = canais(E1, {"t": kb, "p": kb, "sp": kb, "vb": kv})
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    b, i, d, fp, h, lm = mede(pos(v, REFRAT_H, DUR_MIN))
    if d == 8 and (melhor is None or (b, -fp) > melhor[0]):
        melhor = ((b, -fp), kb, kv, b, i, d, fp, h, lm)
    if d >= 7:
        print(f"{kb:5.1f}{kv:5.1f} | {b:5d}/8{i:7d}/8{d:6d}/8{fp:10.3f}{h:9.1f}"
              f"{(lm if np.isfinite(lm) else 0):8.1f}h")
if melhor:
    _, kb, kv, b, i, d, fp, h, lm = melhor
    print(f"\n  melhor com det=8/8: k={kb}/{kv} -> banda {b}/8, inicio {i}/8, "
          f"{fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
print("\n  referencia publicada: banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes, 7,1 h/mes, lead 12,5 h")


# ---------------------------------------------------------------------------
# A varredura acima usou a faixa de k do sinal ORIGINAL. O condicionado tem
# escala diferente -- t medio de 5,8 a 8,1 contra 2,3 a 3,4 -- entao fica sempre
# aceso ali. Recalibra: escolhe BASE de modo que o percentil de operacao do sinal
# condicionado case com o do original, e varre k em torno de 1.
print("\n\n4. RECALIBRANDO A ESCALA DO SINAL CONDICIONADO")
print("=" * 96)
B2 = {}
for c in SIN:
    s0 = E0[c].where(mask).dropna(); s1 = E1[c].where(mask).dropna()
    # o limiar original corresponde a que percentil do sinal original?
    pc = float((s0 <= BASE[c]*K[c]).mean())
    B2[c] = float(s1.quantile(pc)) / K[c] if pc < 1 else BASE[c]
    print(f"  {c}: limiar orig {BASE[c]*K[c]:7.3f} = p{100*pc:5.2f} do original"
          f"  ->  mesmo percentil no condicionado = {s1.quantile(pc):8.3f}"
          f"   (BASE {BASE[c]:.2f} -> {B2[c]:.3f})")

print("\n   varredura com a escala recalibrada")
print(f"{'kb':>5}{'kv':>5} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 96)
import types
def canais2(EW, K, B):
    out = {}
    for c in SIN:
        thr = B[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out

melhor = None
for kb, kv in itertools.product([0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8], [1.4, 1.8, 2.2, 2.8]):
    ON = canais2(E1, {"t": kb, "p": kb, "sp": kb, "vb": kv}, B2)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    b, i, d, fp, h, lm = mede(pos(v, REFRAT_H, DUR_MIN))
    if d == 8 and (melhor is None or (b, -fp) > melhor[0]):
        melhor = ((b, -fp), kb, kv, b, i, d, fp, h, lm)
    if d >= 7:
        print(f"{kb:5.1f}{kv:5.1f} | {b:5d}/8{i:7d}/8{d:6d}/8{fp:10.3f}{h:9.1f}"
              f"{(lm if np.isfinite(lm) else 0):8.1f}h")
if melhor:
    _, kb, kv, b, i, d, fp, h, lm = melhor
    print(f"\n  MELHOR com det=8/8: k={kb}/{kv} -> banda {b}/8, inicio {i}/8, "
          f"{fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
else:
    print("\n  nenhuma configuracao com det = 8/8")
print("  referencia publicada: banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes, 7,1 h/mes, lead 12,5 h")
