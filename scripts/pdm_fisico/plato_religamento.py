#!/usr/bin/env python3
"""O platô em volta do ponto escolhido -- margem, nao argmax.

Mesma disciplina que o Diego usou para escolher os 45 min: mapear a faixa em que
o resultado se mantem e ficar no meio dela, em vez de pegar o melhor ponto e
descobrir depois que ele estava na borda de uma quebra.

Varre frac fino no refratario escolhido, e tambem varre o refratario no frac
escolhido -- as duas direcoes que importam.
"""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, EW, mask, idx, alvo
from publica_clearml import SIN, BASE
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

KBv, KVv, MG, DM = 1.7, 2.2, True, 120
paradas = paradas_reais_2h(); meses = float(mask.sum()) * 2 / 60.0 / 730.0
ON = partes(KBv, KVv); ns = sum(ON[c].astype(int) for c in SIN)
K = {"t": KBv, "p": KBv, "sp": KBv, "vb": KVv}
F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
v0 = (pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])).to_numpy()


def mede(fr, rf):
    v = pd.Series(corta_rearma(v0, F, fr) if fr > 0 else v0, index=idx)
    al = pos(v, ns, rf, DM, False)
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    return m["det"], mi["det"], nfp/meses, h/meses


print("PLATO EM frac  (refratario fixo em 72 h)")
print("=" * 76)
print(f"{'frac':>7} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8}   ")
print("-" * 76)
for fr in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
    det, di, fp, h = mede(fr, 72)
    marca = "  <<< quebra" if det < 8 else ("  ok" if di >= 5 else "")
    print(f"{fr:7.3f} {det:5d}/8 {di:7d}/8 {fp:9.3f} {h:8.1f}{marca}")

print("\nPLATO EM refratario  (frac fixo em 0,02)")
print("=" * 76)
print(f"{'refrat':>7} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8}")
print("-" * 76)
for rf in [24, 36, 48, 60, 72, 84, 96, 120, 144]:
    det, di, fp, h = mede(0.02, rf)
    marca = "  <<< quebra" if det < 8 else ("  ok" if di >= 5 else "")
    print(f"{rf:6d}h {det:5d}/8 {di:7d}/8 {fp:9.3f} {h:8.1f}{marca}")
