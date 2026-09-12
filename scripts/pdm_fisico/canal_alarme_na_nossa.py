#!/usr/bin/env python3
"""O canal 4 do Diego dentro da NOSSA maquina de decisao.

POR QUE VALE TESTAR apesar de fraco. Na maquina dele o canal entra numa contagem
simples (voto >= 2 de 4) sem controle nenhum, e fica aceso 47,6% do tempo -- daí
ser essencial em 5 das 8 deteccoes com enriquecimento de so 1,45x. Na NOSSA
maquina existe o portao `sp OU vb`, que impede o canal de alarme de fechar o voto
com qualquer outro: para disparar, um canal de MANCAL tem de estar aceso.

Entao a pergunta e legitima: a nossa camada consegue extrair valor de um sinal
fraco que a maquina dele nao consegue?

Testa tambem a versao construida por NOS: em vez das 5 tags mais movimentadas do
catalogo (escolha dele), as tags com ENRIQUECIMENTO real antes dos trips.
"""
from __future__ import annotations
import itertools, os
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV, paradas_reais_2h
from fp_alarmes import catalogo, ALARMES

TAGS_DELE = ["PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"]
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
cat = catalogo(idx)


def canal_alarme(tags, jan_h):
    """True se alguma das tags disparou nas ultimas jan_h horas."""
    s = cat[cat["Tag Alarme"].astype(str).isin(tags)]["t"]
    if not len(s):
        return pd.Series(False, index=idx)
    tv = np.sort(np.asarray(pd.DatetimeIndex(s).astype("int64")))
    ti = np.asarray(idx.astype("int64"))
    pos_ = np.searchsorted(tv, ti, "right") - 1
    out = np.zeros(len(ti), dtype=bool)
    v = pos_ >= 0
    dt = (ti[v] - tv[pos_[v]]) / 1e6 / 3600.0        # us -> h
    out[v] = dt <= jan_h
    return pd.Series(out, index=idx) & mask


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


ON = partes(KB, KV)
print("O CANAL 4 DELE COMO 5o SINAL NA NOSSA MAQUINA")
print("=" * 104)
print(f"{'janela':>8}{'voto':>6}{'portao':>16} | {'banda':>7}{'inicio':>8}{'det':>7}"
      f"{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 104)
b0, i0, d0, fp0, h0, l0 = mede(pos(
    pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
    & (ON["sp"] | ON["vb"]), REFRAT_H, DUR_MIN))
print(f"{'--':>8}{2:6d}{'sp|vb':>16} | {b0:5d}/8{i0:7d}/8{d0:6d}/8{fp0:10.3f}{h0:9.1f}"
      f"{l0:8.1f}h   <<< sem o canal 4")
print("-" * 104)
melhor = None
for jan_h in (6, 12, 24, 48):
    ca = canal_alarme(TAGS_DELE, jan_h)
    duty = 100*float(ca.sum())/float(mask.sum())
    for nv in (2, 3):
        for pn, pf in (("sp|vb", ON["sp"] | ON["vb"]),
                       ("sp|vb E alarme", (ON["sp"] | ON["vb"]) & ca)):
            ns = sum(ON[c].astype(int) for c in SIN) + ca.astype(int)
            v = pd.Series(ns >= nv, index=idx) & mask & pf
            b, i, d, fp, h, lm = mede(pos(v, REFRAT_H, DUR_MIN))
            if d == 8 and (melhor is None or (b, -fp, -h) > melhor[0]):
                melhor = ((b, -fp, -h), jan_h, nv, pn, b, i, d, fp, h, lm)
            if b >= b0 and d >= 7:
                print(f"{jan_h:7d}h{nv:6d}{pn:>16} | {b:5d}/8{i:7d}/8{d:6d}/8"
                      f"{fp:10.3f}{h:9.1f}{(lm if np.isfinite(lm) else 0):8.1f}h"
                      f"   (duty {duty:.0f}%)")
print("-" * 104)
if melhor:
    _, jh, nv, pn, b, i, d, fp, h, lm = melhor
    print(f"  melhor com det=8/8: janela {jh}h, voto>={nv}, portao={pn}")
    print(f"     -> banda {b}/8, inicio {i}/8, {fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
    print(f"  sem o canal 4     : banda {b0}/8, inicio {i0}/8, {fp0:.3f} FP/mes, {h0:.1f} h/mes")
else:
    print("  nenhuma configuracao com o canal 4 mantem det = 8/8")
