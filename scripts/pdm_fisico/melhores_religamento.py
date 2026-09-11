#!/usr/bin/env python3
"""Os candidatos com 8/8 e det_ini 5/8, pontuados na REGRA C -- a mesma unidade
do 0,517 publicado. A varredura mede FP bruto; aqui converte."""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, EW, mask, idx, alvo
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

d = pd.read_csv("varredura_religamento.csv")
paradas = paradas_reais_2h(); meses = float(mask.sum()) * 2 / 60.0 / 730.0
CACHE = {}


def regra_c(kb, kv, mg, fr, rf, dm, esc):
    if (kb, kv) not in CACHE:
        ON = partes(kb, kv)
        K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
        F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
        CACHE[(kb, kv)] = (ON, sum(ON[c].astype(int) for c in SIN), F)
    ON, ns, F = CACHE[(kb, kv)]
    v = pd.Series(ns >= 2, index=idx) & mask
    if mg:
        v = v & (ON["sp"] | ON["vb"])
    if fr > 0:
        v = pd.Series(corta_rearma(v.to_numpy(), F, fr), index=idx)
    al = pos(v, ns, rf, dm, esc)
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    nnt = sum(1 for _, _, k, _ in cls if k == "NEUTRO")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    return dict(fp=nfp, neutro=nnt, fp_mes=nfp/meses, h_mes=h/meses,
                det=m["det"], det_ini=mi["det"], lead=m["lead_med"],
                lead_ini=mi["lead_med"], eps=len(eps))


alvo_cfg = d[(d.det == 8) & (d.det_ini == 5)].sort_values(["fp_mes", "h_fp_mes"])
print(f"{len(alvo_cfg)} configuracoes fazem 8/8 com det_ini 5/8. As 8 mais baratas,")
print("pontuadas na REGRA C (mesma unidade do 0,517 publicado):\n")
print("=" * 108)
print(f"{'frac':>5} {'refrat':>7} {'dur':>6} {'k':>9} {'esc':>5} {'portao':>7} | "
      f"{'det':>4} {'ini':>4} {'FP':>3} {'NEU':>4} {'FP/mes':>8} {'h/mes':>7} "
      f"{'lead':>7} {'lead_ini':>9}")
print("-" * 108)
pub = regra_c(1.7, 2.2, True, 0.0, REFRAT_H, DUR_MIN, False)
print(f"{'--':>5} {REFRAT_H:6d}h {DUR_MIN:5d}m {'1.7/2.2':>9} {'nao':>5} {'sim':>7} | "
      f"{pub['det']:3d}/8 {pub['det_ini']:3d}/8 {pub['fp']:3d} {pub['neutro']:4d} "
      f"{pub['fp_mes']:8.3f} {pub['h_mes']:7.1f} {pub['lead']:6.1f}h {pub['lead_ini']:8.1f}h"
      f"   <<< publicado")
print("-" * 108)
vistos = set()
for _, r in alvo_cfg.iterrows():
    key = (r.frac, r.refrat, r.dur_min, r.kb, r.kv, r.escalada, r.mancal)
    if key in vistos:
        continue
    vistos.add(key)
    m = regra_c(r.kb, r.kv, bool(r.mancal), r.frac, int(r.refrat), int(r.dur_min),
                bool(r.escalada))
    print(f"{r.frac:5.2f} {int(r.refrat):6d}h {int(r.dur_min):5d}m "
          f"{f'{r.kb}/{r.kv}':>9} {('sim' if r.escalada else 'nao'):>5} "
          f"{('sim' if r.mancal else 'nao'):>7} | {m['det']:3d}/8 {m['det_ini']:3d}/8 "
          f"{m['fp']:3d} {m['neutro']:4d} {m['fp_mes']:8.3f} {m['h_mes']:7.1f} "
          f"{m['lead']:6.1f}h {m['lead_ini']:8.1f}h")
    if len(vistos) >= 8:
        break
