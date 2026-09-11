#!/usr/bin/env python3
"""Superficie fina (idade x ABS) -- separar plato de ponto de sorte.

Na grade grossa apareceu nao-monotonia: idade=72h dava 7/8 em ABS=6, 6/8 em
ABS=4 e 7/8 em ABS=3. Um resultado que oscila quando o parametro varia
monotonicamente esta preso a alinhamento de borda de episodio, nao a forca de
sinal -- e a mesma critica que fizemos ao filtro de 45 min do Diego.

So aceita um nivel se ele tiver PLATO conexo em ambos os eixos.
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
IDADES = [24, 36, 48, 60, 72, 84, 96, 120]
ABSS = [300, 200, 120, 80, 50, 30, 20, 14, 10, 8, 6, 5, 4, 3.5, 3, 2.5]

res = []
for ida in IDADES:
    for ab in ABSS:
        v = quebra_idade(corta_rearma(v_base, F, 0.03), F, ab, ida)
        al = pos_idade(pd.Series(v, index=idx), 72, DUR_MIN, F, ab, ida)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        cls = classifica_regra_c(AV.episodios(al), paradas)
        nfp = sum(1 for _, _, k, _ in cls if k == "FP")
        h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
        res.append(dict(idade=ida, ABS=ab, det=m["det"], det_ini=mi["det"],
                        fp=nfp/meses, h=h/meses))
    print(f"  idade={ida}h ok", flush=True)
d = pd.DataFrame(res); d.to_csv("superficie_idade_abs.csv", index=False)

print("\nMAPA DE det_ini  (linhas = idade, colunas = ABS; X = perdeu det 8/8)")
print("=" * 100)
piv = d.pivot(index="idade", columns="ABS", values="det_ini")
det8 = d.pivot(index="idade", columns="ABS", values="det")
print("idade  " + "".join(f"{a:>6g}" for a in ABSS))
for ida in IDADES:
    linha = "".join(f"{'  X   ' if det8.loc[ida,a] < 8 else f'{int(piv.loc[ida,a]):>6d}'}"
                    for a in ABSS)
    print(f"{ida:4d}h  {linha}")

print("\n\nPLATOS CONEXOS (det = 8/8), por nivel de det_ini")
print("=" * 100)
ok = d[d.det == 8]
for k in sorted(ok.det_ini.unique(), reverse=True):
    s = ok[ok.det_ini == k]
    # maior retangulo conexo: conta celulas e verifica se ha bloco >= 2x2
    m = piv.copy()
    bloco = 0
    for i, ida in enumerate(IDADES[:-1]):
        for j, ab in enumerate(ABSS[:-1]):
            quad = [(IDADES[i], ABSS[j]), (IDADES[i], ABSS[j+1]),
                    (IDADES[i+1], ABSS[j]), (IDADES[i+1], ABSS[j+1])]
            if all(piv.loc[x, y] == k and det8.loc[x, y] == 8 for x, y in quad):
                bloco += 1
    b = s.sort_values(["fp", "h"]).iloc[0]
    ver = ("PLATO 2x2 -- defensavel" if bloco >= 2 else
           "bloco unico 2x2 -- limitrofe" if bloco == 1 else
           "SEM bloco 2x2 -- ponto de sorte")
    print(f"  det_ini {k}/8: {len(s):2d} celulas, {bloco} blocos 2x2   {ver}")
    print(f"      mais barato: idade={int(b.idade)}h ABS={b.ABS:g} -> "
          f"{b.fp:.3f} FP/mes, {b.h:.1f} h/mes")
