#!/usr/bin/env python3
"""EIXO NUNCA VARRIDO -- trocar limiar por confirmacao.

O voto esta cravado em >= 2 em TODO o codigo (pos_processamento, busca_conjunta,
publica_clearml). Varremos k exaustivamente com o voto fixo, nunca a combinacao.

POR QUE ISSO PODE MELHORAR. A analise de banda mostra que o nosso problema nao e
FP -- e o INSTANTE do alarme: leads de 1,3 h a 194,9 h, contra 3,8-43,2 h do
Diego. Lead curto vem de limiar alto (o sinal so cruza perto da falha); lead
longo vem de limiar baixo num canal (o vb em ~p69 acende na deriva e fica de pe).

Baixar o limiar aumenta o lead dos curtos, mas explode o FP -- A NAO SER que se
compense exigindo mais canais em concordancia. Essa troca desloca o ponto num
eixo diferente do que ja exploramos.

METRICA ALVO: deteccoes na BANDA ACIONAVEL [4 h, 48 h], nao a contagem crua.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, pos, mask, idx, alvo
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, DUR_MIN
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

reset = (~mask).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
TMIN, TMAX = 4.0, 48.0


def canais(kb, kv):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    out = {}
    for c in SIN:
        thr = BASE[c]*K[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out


def avalia_banda(al):
    """Deteccoes cujo INICIO de episodio cai em [t-TMAX, t-TMIN]."""
    eps = AV.episodios(al)
    det, leads = 0, []
    for t in alvo:
        c = [a for a, _ in eps
             if t - pd.Timedelta(hours=TMAX) <= a <= t - pd.Timedelta(hours=TMIN)]
        if c:
            det += 1; leads.append((t - max(c)).total_seconds()/3600)
    return det, leads


KB = [0.8, 1.0, 1.2, 1.4, 1.7, 2.0]
KV = [1.2, 1.5, 1.8, 2.2, 2.8]
print(f"BANDA ACIONAVEL [{TMIN:.0f} h, {TMAX:.0f} h] -- referencia: Diego faz 7/8 nela")
print("=" * 96)
print(f"{'voto':>5} {'portao':>7} {'kb':>5} {'kv':>5} {'banda':>7} {'det':>6} "
      f"{'FP/mes':>9} {'h/mes':>8} {'lead med':>9}")
print("-" * 96)
best = []
for nv, mg, kb, kv in itertools.product((2, 3), (True, False), KB, KV):
    ON = canais(kb, kv); ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= nv, index=idx) & mask
    if mg:
        v = v & (ON["sp"] | ON["vb"])
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
    al = pos(pd.Series(corta_rearma(v.to_numpy(), F, 0.03), index=idx), ns, 72, DUR_MIN, False)
    nb, leads = avalia_banda(al)
    m = AV.avalia(al, alvo, mask)
    cls = classifica_regra_c(AV.episodios(al), paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    best.append((nb, -nfp/meses, nv, mg, kb, kv, m["det"], nfp/meses, h/meses,
                 np.mean(leads) if leads else np.nan))
best.sort(reverse=True)
for nb, _, nv, mg, kb, kv, det, fp, h, lm in best[:16]:
    print(f"{nv:5d} {('sim' if mg else 'nao'):>7} {kb:5.1f} {kv:5.1f} {nb:5d}/8 "
          f"{det:5d}/8 {fp:9.3f} {h:8.1f} {lm:8.1f}h")
print("-" * 96)
print("  ponto atual (voto>=2, portao, kb=1,7 kv=2,2): banda 4/8, 0,775 FP/mes")
