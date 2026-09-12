#!/usr/bin/env python3
"""Os canais DELE como sinais extras na NOSSA maquina -- a direcao inversa.

JA TESTADO E FALHOU: a nossa camada de decisao nos canais dele (128 configs, so
a config dele faz 8/8) e o nosso vb na varredura do Francisco (960 configs).

NUNCA TESTADO: os canais dele entrando na NOSSA maquina. E a combinacao que
faz sentido pelo diagnostico -- os canais dele tem o timing bom (leads de 3,8 a
43,2 h) e a nossa maquina tem a seletividade (0,52 contra 3,78 FP/mes). La nao
havia estrutura de consenso para o nosso portao operar; aqui os nossos 4 sinais
continuam existindo e os dele entram como votos adicionais.

Os artefatos dele sao BINARIOS, entao entram como voto, nao como escore -- o
CUSUM e a escala continuam valendo so para os nossos 4.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h

DIEGO = "/home/thallys/Documents/projeto-petrobras/wt-diego/canais/merged.parquet"
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0

m = pd.read_parquet(DIEGO); m.index = m.index.tz_localize("UTC")
DEL = {}
for c in ("temperatura", "vibracao", "oleo"):
    s = m[c].astype(bool).resample("2min").max().reindex(idx, fill_value=False)
    DEL["d_" + c[:4]] = s.fillna(False).astype(bool) & mask
print("canais dele, ciclo de trabalho na nossa mascara:")
for k, v in DEL.items():
    print(f"  {k:10} {100*float(v.sum())/float(mask.sum()):5.2f}%")

ON = partes(KB, KV)
print("canais nossos:")
for c in SIN:
    print(f"  {c:10} {100*float(ON[c].sum())/float(mask.sum()):5.2f}%")


def pos(v, refrat_h, dur_min):
    al = pd.Series(False, index=idx); bloq = None
    for a, b in AV.episodios(v):
        if bloq is not None and a <= bloq: continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat_h)
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        if (b-a).total_seconds()/60 + 2 >= dur_min: fin.loc[a:b] = True
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
    d = AV.avalia(al, alvo, mask)["det"]
    return banda, ini, d, fp/meses, h/meses, (np.mean(leads) if leads else np.nan)


print("\n\nNOSSOS 4 + OS 3 DELE, com a NOSSA camada de decisao")
print("=" * 104)
print(f"{'sinais':>22}{'voto':>6}{'portao':>18} | {'banda':>7}{'inicio':>8}{'det':>7}"
      f"{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 104)
PORT = {"sp|vb (nosso)": lambda C: C["sp"] | C["vb"],
        "sp|vb ou vib dele": lambda C: C["sp"] | C["vb"] | C["d_vibr"],
        "nenhum": lambda C: pd.Series(True, index=idx)}
res = []
base = {**ON, **DEL}
for conj, nomes in (("so nossos 4", list(SIN)),
                    ("4 + 3 dele", list(SIN) + list(DEL)),
                    ("4 + so vib dele", list(SIN) + ["d_vibr"]),
                    ("4 + so oleo dele", list(SIN) + ["d_oleo"])):
    ns = sum(base[c].astype(int) for c in nomes)
    for nv in (2, 3):
        for pn, pf in PORT.items():
            if "dele" in pn and "d_vibr" not in nomes: continue
            v = pd.Series(ns >= nv, index=idx) & mask & pf(base)
            b, i, d, fp, h, lm = mede(pos(v, REFRAT_H, DUR_MIN))
            res.append((b, -fp, conj, nv, pn, b, i, d, fp, h, lm))
            if d >= 7 or b >= 4:
                print(f"{conj:>22}{nv:6d}{pn:>18} | {b:5d}/8{i:7d}/8{d:6d}/8"
                      f"{fp:10.3f}{h:9.1f}{(lm if np.isfinite(lm) else 0):8.1f}h")
print("-" * 104)
res.sort(reverse=True)
b, _, conj, nv, pn, b_, i, d, fp, h, lm = res[0]
print(f"  melhor banda: {conj}, voto>={nv}, portao={pn} -> banda {b_}/8, inicio {i}/8, "
      f"det {d}/8, {fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
print("  publicado   : banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes, 7,1 h/mes, lead 12,5 h")
