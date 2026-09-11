#!/usr/bin/env python3
"""Duas checagens antes de aceitar o ponto de idade=120h, ABS=6.

1. 120 h e INTERIOR do plato ou BORDA? A grade parava ali. Estender ate 240 h.
2. QUAIS episodios a regra realmente quebra? A hipotese e que so os muito longos
   sao quebrados, e que nesta base os muito longos sao TP ou NEUTRO -- por isso o
   ganho de 2 deteccoes sai de graca. Se for isso, e interpretavel; se estiver
   quebrando FP tambem e dando sorte, e ajuste.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
from escalada_por_idade import quebra_idade, pos_idade, F, v_base

paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
JAN = pd.Timedelta(hours=48)

print("1. O PLATO CONTINUA ACIMA DE 120 h?")
print("=" * 80)
print(f"{'idade':>7} {'ABS':>5} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8}")
print("-" * 80)
for ida in [96, 120, 144, 168, 192, 216, 240]:
    for ab in [5, 6, 8]:
        v = quebra_idade(corta_rearma(v_base, F, 0.03), F, ab, ida)
        al = pos_idade(pd.Series(v, index=idx), 72, DUR_MIN, F, ab, ida)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        cls = classifica_regra_c(AV.episodios(al), paradas)
        nfp = sum(1 for _, _, k, _ in cls if k == "FP")
        h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
        print(f"{ida:6d}h {ab:5.0f} {m['det']:5d}/8 {mi['det']:7d}/8 "
              f"{nfp/meses:9.3f} {h/meses:8.1f}")
    print("-" * 80)

print("\n2. QUAIS EPISODIOS A REGRA QUEBRA?  (idade=144h, ABS=6)")
print("=" * 92)
IDA, AB = 144, 6
v_sem = corta_rearma(v_base, F, 0.03)
v_com = quebra_idade(v_sem, F, AB, IDA)
al_sem = pos_idade(pd.Series(v_sem, index=idx), 72, DUR_MIN, F, np.inf, IDA)
al_com = pos_idade(pd.Series(v_com, index=idx), 72, DUR_MIN, F, AB, IDA)
cls_sem = classifica_regra_c(AV.episodios(al_sem), paradas)
cls_com = classifica_regra_c(AV.episodios(al_com), paradas)
print(f"  episodios antes: {len(cls_sem)}   depois: {len(cls_com)}")
print(f"\n{'episodio ANTES':>34} {'classe':>8} {'dur':>8}  ->  vira")
print("-" * 92)
for a, b, k, _ in cls_sem:
    filhos = [(x, y, kk) for x, y, kk, _ in cls_com if x >= a - pd.Timedelta("3h")
              and y <= b + pd.Timedelta("3h")]
    if len(filhos) > 1:
        det = ", ".join(f"{kk} {(y-x).total_seconds()/3600:.0f}h" for x, y, kk in filhos)
        print(f"{a:%d/%m/%Y %H:%M} a {b:%d/%m %H:%M} {k:>8} "
              f"{(b-a).total_seconds()/3600:7.0f}h  ->  {len(filhos)}x [{det}]")
print("-" * 92)
print("  so os episodios longos sao quebrados; se todos forem TP/NEUTRO,")
print("  o ganho de deteccao nao cria FP -- e o mecanismo, nao sorte.")
