#!/usr/bin/env python3
"""Varredura completa com o RELIGAMENTO como eixo novo.

Pergunta: 0,861 FP/mes e o piso do conserto, ou so o primeiro valor que
funcionou? O ponto medido a mao (frac=0,02 sobre a config publicada) mantem 8/8
e leva a regua de inicio de 4/8 a 5/8, custando FP/mes 0,517 -> 0,861. Aqui os
outros seis eixos podem se reajustar em volta do corte.

Pontua nas DUAS reguas: `det` (a nossa, alarme de pe na janela) e `det_ini`
(a dos outros dois times, inicio do episodio na janela).
"""
from __future__ import annotations
import itertools, numpy as np, pandas as pd
import avalia as AV
from pos_processamento import (partes, pos, EW, mask, idx, alvo,
                               KB, KV, REFRAT, DURMIN)
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio

FRAC = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]

# forca e voto base dependem so de (kb, kv); o portao entra depois
P, F = {}, {}
for kb in KB:
    for kv in KV:
        P[(kb, kv)] = partes(kb, kv)
        K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
        F[(kb, kv)] = pd.concat([EW[c].where(mask) / (BASE[c] * K[c]) for c in SIN],
                                axis=1).max(axis=1).to_numpy()
print(f"canais pre-calculados para {len(P)} pares (kb, kv)", flush=True)

# voto ja cortado, por (kb, kv, mancal, frac)
V = {}
for (kb, kv), ON in P.items():
    ns = sum(ON[c].astype(int) for c in SIN)
    for mg in (False, True):
        v0 = pd.Series(ns >= 2, index=idx) & mask
        if mg:
            v0 = v0 & (ON["sp"] | ON["vb"])
        arr = v0.to_numpy()
        for fr in FRAC:
            V[(kb, kv, mg, fr)] = (
                v0 if fr == 0 else
                pd.Series(corta_rearma(arr, F[(kb, kv)], fr), index=idx))
print(f"votos cortados: {len(V)}", flush=True)

lin = []
for kb, kv in P:
    ns = sum(P[(kb, kv)][c].astype(int) for c in SIN)
    for mg, fr, rf, dm, esc in itertools.product((False, True), FRAC, REFRAT,
                                                 DURMIN, (False, True)):
        al = pos(V[(kb, kv, mg, fr)], ns, rf, dm, esc)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        lin.append(dict(kb=kb, kv=kv, mancal=mg, frac=fr, refrat=rf, dur_min=dm,
                        escalada=esc, det=m["det"], det_ini=mi["det"],
                        eps=m["episodios"], fp_mes=round(m["fp_mes"], 3),
                        h_fp_mes=round(m["h_fp_mes"], 1),
                        lead=round(m["lead_med"], 2) if m["det"] else np.nan,
                        fp_ini=round(mi["fp_mes"], 3),
                        h_ini=round(mi["h_fp_mes"], 1),
                        lead_ini=round(mi["lead_med"], 2) if mi["det"] else np.nan))
    print(f"  kb={kb} kv={kv} ok ({len(lin)} pontos)", flush=True)

d = pd.DataFrame(lin)
d.to_csv("varredura_religamento.csv", index=False)
print(f"\n{len(d)} pontos -> varredura_religamento.csv")

base = d[(d.kb == 1.7) & (d.kv == 2.2) & d.mancal & (d.frac == 0) &
         (d.refrat == REFRAT_H) & (d.dur_min == DUR_MIN) & ~d.escalada].iloc[0]
print(f"\ncontrole (ponto publicado): {base.det}/8 · {base.fp_mes:.3f} FP/mes · "
      f"{base.h_fp_mes:.1f} h/mes · det_ini {base.det_ini}/8   "
      f"(esperado 8/8, 0,517, 7,1, 4/8)")

print("\n" + "=" * 104)
print("MANTENDO 8/8 NA NOSSA REGUA -- qual o menor custo por nivel de det_ini?")
print("=" * 104)
oito = d[d.det == 8]
print(f"{'det_ini':>8} {'n':>6} {'FP/mes':>9} {'h/mes':>8} {'lead':>8} {'lead_ini':>9}  configuracao")
print("-" * 104)
for k in sorted(oito.det_ini.unique(), reverse=True):
    s = oito[oito.det_ini == k].sort_values(["fp_mes", "h_fp_mes"])
    b = s.iloc[0]
    print(f"{int(k):6d}/8 {len(s):6d} {b.fp_mes:9.3f} {b.h_fp_mes:8.1f} {b.lead:7.1f}h "
          f"{b.lead_ini:8.1f}h  frac={b.frac} refrat={int(b.refrat)}h dur={int(b.dur_min)}min "
          f"k={b.kb}/{b.kv} esc={b.escalada} portao={b.mancal}")

print("\n" + "=" * 104)
print("O PISO DE FP PARA CADA (det, det_ini)")
print("=" * 104)
piv = d.groupby(["det", "det_ini"])["fp_mes"].min().unstack()
print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
