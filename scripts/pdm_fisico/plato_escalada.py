#!/usr/bin/env python3
"""Quais niveis da fronteira tem PLATO e quais sao ponto unico?

A barra ABS esta sendo escolhida olhando os 8 eventos -- exatamente a critica que
fizemos ao filtro de 45 min do Diego. Com 8 rotulos o grau de liberdade se esgota
rapido. O que se pode defender e um nivel com plato LARGO; um nivel que so
aparece num valor de ABS e ajuste, nao resultado.
"""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, mask, idx, alvo
from publica_clearml import SIN, BASE, DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c
from escalada_absoluta import quebra_absoluta, pos_furo_abs, F, v_base, ns

paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
print(f"{'ABS':>7} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8}")
print("-" * 52)
GRADE = [200, 150, 100, 70, 50, 35, 25, 20, 15, 12, 10, 9, 8, 7, 6, 5,
         4.5, 4, 3.5, 3, 2.8, 2.5, 2.2, 2.0]
res = []
for ABS in GRADE:
    v = quebra_absoluta(corta_rearma(v_base, F, 0.03), F, ABS)
    al = pos_furo_abs(pd.Series(v, index=idx), 72, DUR_MIN, F, ABS, True)
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    cls = classifica_regra_c(AV.episodios(al), paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    res.append((ABS, m["det"], mi["det"], nfp/meses, h/meses))
    print(f"{ABS:7.1f} {m['det']:5d}/8 {mi['det']:7d}/8 {nfp/meses:9.3f} {h/meses:8.1f}")

print("\nLARGURA DO PLATO POR NIVEL DE det_ini")
print("=" * 62)
d = pd.DataFrame(res, columns=["ABS", "det", "det_ini", "fp", "h"])
for k in sorted(d.det_ini.unique(), reverse=True):
    s = d[(d.det_ini == k) & (d.det == 8)]
    if not len(s):
        continue
    lo, hi = s.ABS.min(), s.ABS.max()
    n = len(s)
    veredito = ("plato LARGO -- defensavel" if n >= 5 else
                "plato estreito -- frageil" if n >= 3 else
                "PONTO UNICO -- ajuste, nao resultado")
    print(f"  det_ini {k}/8: ABS de {lo:g} a {hi:g}  ({n} valores da grade)  {veredito}")
    b = s.sort_values("fp").iloc[0]
    print(f"      mais barato: ABS={b.ABS:g} -> {b.fp:.3f} FP/mes, {b.h:.1f} h/mes")
