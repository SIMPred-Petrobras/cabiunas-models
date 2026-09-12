#!/usr/bin/env python3
"""O canal de alarme (a ideia do canal 4 do Diego) DENTRO do ponto novo.

O teste anterior (`canal_alarme_na_nossa.py`) usou o pos-processamento da versao
publicada e deu banda 4/8 a 1,550 FP/mes. Mas o melhor ponto tem outra estrutura
-- dois niveis + escalada por idade -- e o mesmo erro ja aconteceu com o sp_vib,
que parecia refutado na maquina errada.

Testa em ambos os niveis, com varias janelas, e tambem com um subconjunto de tags
escolhido por FISICA (mancal e oleo) em vez das 5 mais movimentadas que ele usa.
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
from fp_alarmes import catalogo

TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
cat = catalogo(idx)

TAGS = {
  "as 5 dele": ["PI_6240319_AL","PAL_6240315","PDAL_6240302","TC382_05_A","PAH_6240319"],
  "so mancal": [t for t in cat["Tag Alarme"].astype(str).unique()
                if "6240301" in t or "6240303" in t or "6240305" in t or "6240307" in t],
  "so oleo":   [t for t in cat["Tag Alarme"].astype(str).unique()
                if "6240339" in t or "6240340" in t],
  "mancal+oleo": None,
}
TAGS["mancal+oleo"] = TAGS["so mancal"] + TAGS["so oleo"]

def canal_alarme(tags, jan_h):
    s = cat[cat["Tag Alarme"].astype(str).isin(tags)]["t"]
    if not len(s): return pd.Series(False, index=idx)
    tv = np.sort(np.asarray(pd.DatetimeIndex(s).astype("int64")))
    ti = np.asarray(idx.astype("int64"))
    p_ = np.searchsorted(tv, ti, "right") - 1
    out = np.zeros(len(ti), bool); v = p_ >= 0
    out[v] = (ti[v] - tv[p_[v]])/1e6/3600.0 <= jan_h
    return pd.Series(out, index=idx) & mask

p = {k: (dict(P["lo"]) if k == "lo" else P[k]) for k in P}; p["frac"] = 0.0
A0 = {c: canal(c, p["lo"][c]) for c in SIN}
B0 = {c: canal(c, p["hi"] if c != "vb" else p["kvhi"]) for c in SIN}
KH = {"t": p["hi"], "p": p["hi"], "sp": p["hi"], "vb": p["kvhi"]}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()

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
           (np.mean(leads) if leads else np.nan)

def roda(A, B, portaB):
    vA = pd.Series(sum(A[c].astype(int) for c in A) >= 3, index=idx) & mask
    vB = pd.Series(sum(B[c].astype(int) for c in B) >= 2, index=idx) & mask & portaB
    vv = quebra_idade((vA | vB).to_numpy(), F, p["abs_"], p["idade"])
    return mede(pos_dur_esc(pd.Series(vv, index=idx), p["refrat"], F,
                            p["abs_"], p["idade"], p["dur_esc"]))

print(f"{'configuracao':>48} | {'duty':>7}{'banda':>7}{'inicio':>8}{'det':>7}"
      f"{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 116)
r0 = roda(A0, B0, B0["sp"] | B0["vb"])
print(f"{'ponto novo, sem canal de alarme':>48} | {'--':>7}{r0[0]:5d}/8{r0[1]:7d}/8"
      f"{r0[2]:6d}/8{r0[3]:10.3f}{r0[4]:9.1f}{r0[5]:8.1f}h")
print("-" * 116)
melhor = None
for nome, tags in TAGS.items():
    for jan_h in (6, 12, 24):
        ca = canal_alarme(tags, jan_h)
        duty = 100*float(ca.sum())/float(mask.sum())
        for onde in ("so no sensivel", "so no especifico", "nos dois"):
            A = {**A0, "al": ca} if onde != "so no especifico" else dict(A0)
            B = {**B0, "al": ca} if onde != "so no sensivel" else dict(B0)
            r = roda(A, B, B0["sp"] | B0["vb"])
            if r[2] == 8 and (melhor is None or (r[0], -r[3], -r[4]) > melhor[0]):
                melhor = ((r[0], -r[3], -r[4]), nome, jan_h, onde, duty, r)
            if r[0] >= r0[0] and r[2] >= 7 and r[3] <= r0[3]*1.6:
                print(f"{f'{nome}, {jan_h}h, {onde}':>48} | {duty:6.1f}%{r[0]:5d}/8"
                      f"{r[1]:7d}/8{r[2]:6d}/8{r[3]:10.3f}{r[4]:9.1f}"
                      f"{(r[5] if np.isfinite(r[5]) else 0):8.1f}h")
print("-" * 116)
if melhor:
    _, nome, jan_h, onde, duty, r = melhor
    print(f"  MELHOR com det=8/8: tags={nome}, janela={jan_h}h, {onde} (duty {duty:.1f}%)")
    print(f"     -> banda {r[0]}/8, inicio {r[1]}/8, {r[3]:.3f} FP/mes, "
          f"{r[4]:.1f} h/mes, lead {r[5]:.1f}h")
