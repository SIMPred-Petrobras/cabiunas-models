#!/usr/bin/env python3
"""Reescalonamento por barra ABSOLUTA de intensidade.

O criterio relativo (M x o minimo do episodio) fragmenta episodio normal demais:
a M=1,5 traz 27/02 mas perde outro e triplica o FP. Os dados apontam para uma
barra absoluta -- as excursoes que queremos capturar sao grandes:
  27/02  p a 268x o limiar
  17/03  t a 8,38x
  29/04  t a 3,20x  (este e caso de refratario, precisa do furo)

Regra: um instante com forca acima de ABS abre alarme NOVO, custe o que custar --
episodio aberto e furado, refratario e furado. E a leitura operacional obvia:
uma excursao de uma ordem de grandeza nao e continuacao de nada.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from avalia import GAP_EP_H
from pos_processamento import partes, EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE, DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c

N_GAP = int(pd.Timedelta(hours=GAP_EP_H) / (idx[1] - idx[0])) + 1
JAN = pd.Timedelta(hours=48)


def quebra_absoluta(voto, forca, ABS):
    """Abre episodio novo na primeira amostra em que a forca cruza ABS de baixo
    para cima dentro de um episodio ja aberto."""
    v = np.asarray(voto, dtype=bool).copy()
    if not np.isfinite(ABS):
        return v
    f = np.where(np.isfinite(forca), forca, 0.0)
    acima = f > ABS
    dentro = False; ini = 0; ja = False
    for i in range(len(v)):
        if not v[i]:
            dentro = False; ja = False
            continue
        if not dentro:
            dentro = True; ini = i; ja = acima[i]; continue
        if acima[i] and not ja:                 # cruzou para cima dentro do ep
            j = max(ini + 1, i - N_GAP)
            v[j:i] = False
            ini = i
        ja = acima[i]
    return v


def pos_furo_abs(voto, refrat_h, dur_min, forca, ABS, isenta=False):
    """isenta: episodio aberto por excursao acima de ABS nao precisa cumprir
    dur_min. Um pico de 268x por 30 min diz mais que uma deriva de 1,1x por 2 h,
    e era o dur_min que estava matando o episodio novo do 27/02."""
    fs = pd.Series(np.where(np.isfinite(forca), forca, 0.0), index=idx)
    al = pd.Series(False, index=idx); bloq = None; fortes = []
    for a, b in AV.episodios(voto):
        forte = bool(fs.loc[a:b].max() > ABS)
        if bloq is not None and a <= bloq and not forte:
            continue
        al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=refrat_h)
        if forte:
            fortes.append((a, b))
    fin = pd.Series(False, index=idx)
    for a, b in AV.episodios(al):
        forte = isenta and any(x >= a and y <= b for x, y in fortes)
        if forte or (b - a).total_seconds()/60 + 2 >= dur_min:
            fin.loc[a:b] = True
    return fin & sel


K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
ON = partes(KB, KV); ns = sum(ON[c].astype(int) for c in SIN)
v_base = (pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0

print("BARRA ABSOLUTA + ISENCAO DE DURACAO  (sobre religamento frac=0,03, refrat=72h)")
print("=" * 108)
print(f"{'ABS':>7} {'isenta':>7} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8} {'eps':>5}  "
      f"{'ganha':>14} {'perde':>14}")
print("-" * 108)
ref = None
for ABS, isenta in ([(np.inf, False)] + [(a, i) for a in (50, 20, 10, 8, 5, 3)
                                         for i in (True,)]):
    v = corta_rearma(v_base, F, 0.03)
    v = quebra_absoluta(v, F, ABS)
    al = pos_furo_abs(pd.Series(v, index=idx), 72, DUR_MIN, F, ABS, isenta)
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    nasce = {t.strftime("%d/%m") for t in alvo if any(t - JAN <= x <= t for x, _ in eps)}
    if ref is None:
        ref = nasce
    g = sorted(nasce - ref); pd_ = sorted(ref - nasce)
    print(f"{ABS:7.0f} {('sim' if isenta else 'nao'):>7} {m['det']:5d}/8 {mi['det']:7d}/8 {nfp/meses:9.3f} {h/meses:8.1f} "
          f"{len(eps):5d}  {', '.join(g) if g else '--':>14} {', '.join(pd_) if pd_ else '--':>14}")
