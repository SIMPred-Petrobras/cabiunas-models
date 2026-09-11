#!/usr/bin/env python3
"""Os episodios criados pela escalada sao alarmes de verdade ou piscadas?

Em idade=144h o unico filho tinha 0 h -- uma amostra de 2 min. Isento do dur_min,
ele conta como deteccao pela regua de inicio. Operacionalmente isso e inutil: o
operador nao ve um alarme de 2 minutos. Se o ganho vier dai, e gaming da metrica.

Checa a duracao dos episodios que dao a deteccao, e testa exigir um piso
(DUR_ESC) para o episodio escalonado -- menor que os 120 min do normal, mas nao
zero.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from avalia import GAP_EP_H
from pos_processamento import mask, idx, alvo, sel
from publica_clearml import DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
from escalada_por_idade import quebra_idade, F, v_base

paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
JAN = pd.Timedelta(hours=48)


def pos_dur_esc(voto, refrat_h, forca, ABS, idade_h, dur_esc):
    """Como pos_idade, mas o episodio escalonado precisa durar >= dur_esc min."""
    fs = pd.Series(np.where(np.isfinite(forca), forca, 0.0), index=idx)
    al = pd.Series(False, index=idx); bloq = None; ini_bloq = None; fortes = []
    for a, b in AV.episodios(voto):
        forte = bool(fs.loc[a:b].max() > ABS)
        velho = ini_bloq is not None and (a - ini_bloq).total_seconds()/3600 >= idade_h
        if bloq is not None and a <= bloq and not (forte and velho):
            continue
        al.loc[a:b] = True
        bloq = b + pd.Timedelta(hours=refrat_h); ini_bloq = a
        if forte:
            fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        dur = (b - a).total_seconds()/60 + 2
        forte = any(x >= a and y <= b for x, y in fortes)
        if (forte and dur >= dur_esc) or dur >= DUR_MIN:
            fin.loc[a:b] = True
    return fin & sel


print("A DURACAO DO EPISODIO QUE ENTREGA CADA DETECCAO  (idade=96h, ABS=6)")
print("=" * 84)
v = quebra_idade(corta_rearma(v_base, F, 0.03), F, 6, 96)
al = pos_dur_esc(pd.Series(v, index=idx), 72, F, 6, 96, 0)
eps = AV.episodios(al)
for t in alvo:
    t0 = t - JAN
    nasc = [(a, b) for a, b in eps if t0 <= a <= t]
    if nasc:
        a, b = max(nasc, key=lambda ab: ab[0])
        d = (b - a).total_seconds()/60
        flag = "  <<< PISCADA" if d < 30 else ("  curto" if d < 120 else "")
        print(f"  {t:%d/%m/%Y}  nasce {a:%d/%m %H:%M}  lead "
              f"{(t-a).total_seconds()/3600:5.1f}h  dura {d:7.1f} min{flag}")
    else:
        dep = [(a, b) for a, b in eps if a <= t and b >= t0]
        print(f"  {t:%d/%m/%Y}  de pe (ep de "
              f"{(dep[0][1]-dep[0][0]).total_seconds()/3600:.0f}h)" if dep else
              f"  {t:%d/%m/%Y}  nao detecta")

print("\n\nEXIGINDO PISO DE DURACAO NO EPISODIO ESCALONADO")
print("=" * 92)
print(f"{'dur_esc':>8} {'idade':>7} {'ABS':>5} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8}")
print("-" * 92)
for dur_esc in [0, 10, 20, 30, 60, 120]:
    for ida, ab in [(96, 6), (120, 6), (72, 6)]:
        v = quebra_idade(corta_rearma(v_base, F, 0.03), F, ab, ida)
        al = pos_dur_esc(pd.Series(v, index=idx), 72, F, ab, ida, dur_esc)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        cls = classifica_regra_c(AV.episodios(al), paradas)
        nfp = sum(1 for _, _, k, _ in cls if k == "FP")
        h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
        print(f"{dur_esc:7d}m {ida:6d}h {ab:5.0f} {m['det']:5d}/8 {mi['det']:7d}/8 "
              f"{nfp/meses:9.3f} {h/meses:8.1f}")
    print("-" * 92)
