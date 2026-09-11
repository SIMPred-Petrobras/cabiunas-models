#!/usr/bin/env python3
"""Ponto de operacao com piso de duracao honesto no episodio escalonado.

Sem piso, o 7/8 vinha de um alarme de 0 min com lead 0,0 h no 17/03 -- disparo no
instante do trip. Nao e deteccao. Com piso >= 10 min some, e sobra 6/8.

Fixa dur_esc = 60 min (metade do dur_min normal: um episodio aberto por excursao
de ordem de grandeza pode ser mais curto, mas nao pode ser uma piscada) e mapeia
o plato do que resta.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import DUR_MIN, REFRAT_H
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
from escalada_por_idade import quebra_idade, F, v_base
from checa_degenerado import pos_dur_esc

DUR_ESC = 60
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
JAN = pd.Timedelta(hours=48)
IDADES = [48, 72, 96, 120, 144, 168]
ABSS = [200, 50, 20, 10, 8, 6, 5, 4, 3]

print(f"MAPA COM PISO DE {DUR_ESC} min NO EPISODIO ESCALONADO")
print("=" * 92)
print("det_ini | " + "".join(f"{a:>6g}" for a in ABSS))
res = []
for ida in IDADES:
    linha = ""
    for ab in ABSS:
        v = quebra_idade(corta_rearma(v_base, F, 0.03), F, ab, ida)
        al = pos_dur_esc(pd.Series(v, index=idx), 72, F, ab, ida, DUR_ESC)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        cls = classifica_regra_c(AV.episodios(al), paradas)
        nfp = sum(1 for _, _, k, _ in cls if k == "FP")
        h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
        res.append(dict(idade=ida, ABS=ab, det=m["det"], det_ini=mi["det"],
                        fp=nfp/meses, h=h/meses))
        linha += "   X  " if m["det"] < 8 else f"{mi['det']:>6d}"
    print(f"  {ida:4d}h | {linha}")
d = pd.DataFrame(res)

print("\nCUSTO (FP/mes) onde det_ini = 6/8 e det = 8/8")
print("=" * 92)
s = d[(d.det == 8) & (d.det_ini == 6)]
print(s.pivot(index="idade", columns="ABS", values="fp").to_string(float_format=lambda x: f"{x:.3f}"))
print(f"\n  celulas: {len(s)}   custo minimo: {s.fp.min():.3f} FP/mes")

b = s.sort_values(["fp", "h"]).iloc[0]
print(f"\n\nO PONTO: religamento 0,03 + escalada (idade={int(b.idade)}h, ABS={b.ABS:g}, "
      f"piso {DUR_ESC} min) + refrat 72h")
print("=" * 92)
v = quebra_idade(corta_rearma(v_base, F, 0.03), F, b.ABS, b.idade)
al = pos_dur_esc(pd.Series(v, index=idx), 72, F, b.ABS, b.idade, DUR_ESC)
m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
cls = classifica_regra_c(AV.episodios(al), paradas)
nfp = sum(1 for _, _, k, _ in cls if k == "FP")
print(f"{'':<26}{'publicado':>12}{'so religa':>12}{'com escalada':>14}")
print("-" * 92)
for rot, pub, rel, novo in [
        ("deteccao (nossa)", "8/8", "8/8", f"{m['det']}/8"),
        ("deteccao (inicio)", "4/8", "5/8", f"{mi['det']}/8"),
        ("FP/mes (regra C)", "0.517", "0.775", f"{nfp/meses:.3f}"),
        ("h/mes", "7.1", "11.3", f"{sum((y-x).total_seconds()/3600 for x,y,k,_ in cls if k=='FP')/meses:.1f}")]:
    print(f"{rot:<26}{pub:>12}{rel:>12}{novo:>14}")

print(f"\nPOR EVENTO")
print("-" * 92)
eps = AV.episodios(al)
for t in alvo:
    t0 = t - JAN
    nasc = [(x, y) for x, y in eps if t0 <= x <= t]
    if nasc:
        x, y = max(nasc, key=lambda ab: ab[0])
        print(f"  {t:%d/%m/%Y}  NASCE  lead {(t-x).total_seconds()/3600:5.1f}h  "
              f"dura {(y-x).total_seconds()/60:6.0f} min")
    else:
        dep = [(x, y) for x, y in eps if x <= t and y >= t0]
        print(f"  {t:%d/%m/%Y}  de pe  (ep de {(dep[0][1]-dep[0][0]).total_seconds()/3600:.0f}h)"
              if dep else f"  {t:%d/%m/%Y}  nao detecta")
