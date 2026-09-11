#!/usr/bin/env python3
"""IDEIA 3 -- escalada condicionada a IDADE do episodio (reanuncio).

POR QUE. A escalada por barra absoluta (escalada_absoluta.py) leva a regua de
inicio a 6/8 e 7/8, mas cara: ABS baixo fragmenta episodio curto e normal, e cada
fragmento vira FP. Isso e fisicamente errado -- reintensificacao dentro de um
episodio de 2 h e ruido; dentro de um de 100 h e evento novo. E a ISA-18.2 trata
exatamente esse caso: alarme permanente deve ser REANUNCIADO, nao mantido calado.

REGRA. So permite abrir episodio novo por intensidade se o episodio corrente ja
esta de pe ha mais de IDADE horas. Episodio curto nunca e fragmentado -> o custo
da escalada some, e o ganho (27/02, 17/03, 29/04, todos dentro de episodios de
144 h, 195 h e refratario) fica.

HIPOTESE: mesmo det_ini da escalada pura, com FP muito menor.
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

PASSO_H = (idx[1] - idx[0]).total_seconds() / 3600
N_GAP = int(pd.Timedelta(hours=GAP_EP_H) / (idx[1] - idx[0])) + 1
JAN = pd.Timedelta(hours=48)


def quebra_idade(voto, forca, ABS, idade_h):
    """Abre episodio novo quando a forca cruza ABS para cima -- SO se o episodio
    corrente ja tem mais de idade_h de vida."""
    v = np.asarray(voto, dtype=bool).copy()
    if not np.isfinite(ABS):
        return v
    f = np.where(np.isfinite(forca), forca, 0.0)
    acima = f > ABS
    n_idade = int(idade_h / PASSO_H)
    dentro = False; ini = 0; ja = False
    for i in range(len(v)):
        if not v[i]:
            dentro = False; ja = False
            continue
        if not dentro:
            dentro = True; ini = i; ja = acima[i]; continue
        if acima[i] and not ja and (i - ini) >= n_idade:
            j = max(ini + 1, i - N_GAP)
            v[j:i] = False
            ini = i
        ja = acima[i]
    return v


def pos_idade(voto, refrat_h, dur_min, forca, ABS, idade_h):
    """Refratario com furo por intensidade -- tambem so apos idade_h do bloqueio."""
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
        if any(x >= a and y <= b for x, y in fortes) or \
           (b - a).total_seconds()/60 + 2 >= dur_min:
            fin.loc[a:b] = True
    return fin & sel


K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
ON = partes(KB, KV); ns = sum(ON[c].astype(int) for c in SIN)
v_base = (pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0

print("IDEIA 3 -- ESCALADA CONDICIONADA A IDADE DO EPISODIO")
print("=" * 96)
print(f"{'idade':>7} {'ABS':>6} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8} {'eps':>5}")
print("-" * 96)
melhor = {}
for idade in [0, 6, 12, 24, 48, 72]:
    for ABS in [200, 50, 20, 10, 6, 4, 3]:
        v = quebra_idade(corta_rearma(v_base, F, 0.03), F, ABS, idade)
        al = pos_idade(pd.Series(v, index=idx), 72, DUR_MIN, F, ABS, idade)
        m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
        eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
        nfp = sum(1 for _, _, k, _ in cls if k == "FP")
        h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
        if m["det"] < 8:
            continue
        key = mi["det"]
        cand = (nfp/meses, h/meses, idade, ABS, len(eps))
        if key not in melhor or cand < melhor[key]:
            melhor[key] = cand
        print(f"{idade:6d}h {ABS:6.0f} {m['det']:5d}/8 {mi['det']:7d}/8 "
              f"{nfp/meses:9.3f} {h/meses:8.1f} {len(eps):5d}")
    print("-" * 96)

print("\nMELHOR POR NIVEL (mantendo det = 8/8)")
print("=" * 96)
print(f"{'det_ini':>8} {'FP/mes':>9} {'h/mes':>8} {'idade':>7} {'ABS':>6} {'eps':>5}   contra o que temos")
print("-" * 96)
REF = {4: (0.517, 7.1), 5: (0.775, 11.3), 6: (1.550, 18.9), 7: (2.067, 28.8), 8: (2.928, 97.7)}
for k in sorted(melhor, reverse=True):
    fp, h, ida, ab, ne = melhor[k]
    r = REF.get(k)
    cmp = (f"antes {r[0]:.3f}/{r[1]:.1f}  ->  "
           + ("MELHOR" if fp < r[0] - 1e-9 else "igual" if abs(fp-r[0]) < 1e-9 else "pior")) if r else ""
    print(f"{k:6d}/8 {fp:9.3f} {h:8.1f} {ida:6d}h {ab:6.0f} {ne:5d}   {cmp}")
