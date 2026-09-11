#!/usr/bin/env python3
"""Diagnostico dos falsos positivos NO PONTO DE PRODUCAO -- as duas analises que
o relatorio do Diego (04/09/2026) tem e nos nao.

1. FP por COMBINACAO DE CANAL (a Tabela 4 dele, para o nosso detector). Diz qual
   canal carrega o custo, e portanto onde uma supressao teria efeito.

2. Corroboracao por catalogo COM CONTROLE NEGATIVO (o Passo 3 dele). Ele mede
   6,9x de enriquecimento nos 42 FP dele. `fp_alarmes.py` ja fazia isso do nosso
   lado, mas numa configuracao antiga (79 FP); aqui e no ponto publicado.

O nulo e o mesmo de fp_alarmes.py: mantem os alarmes onde estao e RESSORTEIA os
episodios -- mesma quantidade, mesma duracao, inicio uniforme sobre os instantes
pontuaveis. Sem isso, "X% dos FP tem alarme" nao significa nada: sao ~2,4
alarmes/dia no catalogo e um episodio de 12 h contem um por acaso.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, mask, idx, alvo
from plota_estilo_francisco import (alarme, paradas_reais_2h, classifica_regra_c,
                                    KB, KV)
from publica_clearml import SIN
from fp_alarmes import catalogo, conta, nulo

# ---------------------------------------------------------------- classificacao
al = alarme()
eps = AV.episodios(al)
cls = classifica_regra_c(eps, paradas_reais_2h())
fp = [(a, b) for a, b, k, _ in cls if k == "FP"]
tp = [(a, b) for a, b, k, _ in cls if k == "TP"]
nt = [(a, b) for a, b, k, _ in cls if k == "NEUTRO"]
dur_fp = [(b - a).total_seconds() / 3600 for a, b in fp]

print("PONTO DE PRODUCAO -- classificacao dos episodios (regra C)")
print("=" * 78)
print(f"  episodios totais : {len(eps)}")
print(f"  TP  (deteccao)   : {len(tp)}")
print(f"  NEUTRO (parada)  : {len(nt)}")
print(f"  FP               : {len(fp)}   "
      f"mediana {np.median(dur_fp):.2f} h, max {max(dur_fp):.2f} h, "
      f"total {sum(dur_fp):.0f} h")

# ------------------------------------------------- 1. FP por combinacao de canal
ON = partes(KB, KV)
print(f"\n1. FP POR COMBINACAO DE CANAL  (a Tabela 4 do Diego, para nos)")
print("=" * 78)
combos = {}
for a, b in fp:
    ativos = tuple(c for c in SIN if bool(ON[c].loc[a:b].any()))
    combos.setdefault(ativos, []).append((b - a).total_seconds() / 3600)
print(f"{'combinacao':<34} {'episodios':>10} {'dur. mediana':>14} {'horas':>9}")
print("-" * 78)
for k in sorted(combos, key=lambda k: -len(combos[k])):
    v = combos[k]
    print(f"{'+'.join(k):<34} {len(v):10d} {np.median(v):11.2f} h {sum(v):8.0f}")
print("-" * 78)
# quem e ESSENCIAL: sem esse canal o voto>=2 nao fecha, ou o portao sp|vb cai
for c in SIN:
    ess = sum(1 for k, v in combos.items() for _ in v
              if c in k and (len(k) == 2 or (c in ("sp", "vb")
                                             and not ({"sp", "vb"} - {c}) & set(k))))
    part = sum(len(v) for k, v in combos.items() if c in k)
    print(f"  canal {c:<3}: participa de {part:2d}/{len(fp)}  "
          f"e e ESSENCIAL em {ess:2d}/{len(fp)}")

# ------------------------------- 2. corroboracao por catalogo + controle negativo
cat = catalogo(idx)
ts = cat["t"]
span_h = (idx[-1] - idx[0]).total_seconds() / 3600
print(f"\n2. CORROBORACAO POR CATALOGO  (o Passo 3 do Diego, para nos)")
print("=" * 78)
print(f"  catalogo: {len(cat):,} onsets ativos, {cat['Tag Alarme'].nunique()} tags, "
      f"{len(cat)/((idx[-1]-idx[0]).days):.2f}/dia")

pontuaveis = pd.Series(idx[mask.to_numpy()])
com, tot = conta(fp, ts)
coms, tots = nulo([pd.Timedelta(hours=d) for d in dur_fp], pontuaveis, ts)
p = float((coms >= com).mean())
print(f"\n  FP com ao menos um alarme dentro : {com}/{len(fp)} = "
      f"{100*com/len(fp):.1f}%")
print(f"  nulo (mesma duracao, sorteado)   : {coms.mean():.1f} +- {coms.std():.1f}"
      f"  ({100*coms.mean()/len(fp):.1f}%)")
print(f"  ENRIQUECIMENTO                   : {com/coms.mean():.2f}x    p = {p:.4f}")
print(f"     [ele, nos 42 FP dele: 23,8% contra 3,4% = 6,9x]")

# a mesma medida nos TP, como controle positivo
comt, _ = conta(tp, ts)
print(f"\n  controle positivo -- TP com alarme dentro: {comt}/{len(tp)} = "
      f"{100*comt/len(tp):.0f}%")


# ------------------------------------------------------- 3. auditoria individual
# Com 6 episodios nao se faz estatistica: para o nulo de 28,2% seriam precisos
# 5 de 6 para chegar a p < 0,05 (P(X>=4) = 0,057 sob binomial). Entao o unico
# caminho e abrir um por um.
print(f"\n3. OS {len(fp)} FP, UM POR UM  (com {len(fp)} nao ha poder estatistico)")
print("=" * 100)
from math import comb
pn = coms.mean() / len(fp)
for k in range(len(fp), 0, -1):
    pk = sum(comb(len(fp), j) * pn**j * (1-pn)**(len(fp)-j) for j in range(k, len(fp)+1))
    if pk >= 0.05:
        print(f"  poder: seriam precisos {k+1} de {len(fp)} para p < 0,05 "
              f"(temos {com}). P(X>={k}) = {pk:.3f}")
        break

paradas = paradas_reais_2h()
print(f"\n{'inicio':>17} {'dur':>7} {'canais':>12} {'alarmes dentro':>15}  tags / contexto")
print("-" * 100)
for a, b in fp:
    h = (b - a).total_seconds() / 3600
    ativos = "+".join(c for c in SIN if bool(ON[c].loc[a:b].any()))
    dentro = cat[(cat.t >= a) & (cat.t <= b)]
    tags = ", ".join(f"{t}x{n}" if n > 1 else t
                     for t, n in dentro["Tag Alarme"].value_counts().items()) or "--"
    # o episodio nasce na borda de um blackout de 6 h depois de religar?
    ini_op = paradas[(paradas.fim >= a - pd.Timedelta("24h")) & (paradas.fim <= a)]
    borda = ""
    if len(ini_op):
        dt = (a - ini_op.fim.iloc[-1]).total_seconds() / 3600
        borda = f"  [{dt:.2f} h apos religar]"
    print(f"{a:%d/%m/%Y %H:%M} {h:6.1f}h {ativos:>12} {len(dentro):15d}  {tags[:44]}{borda}")
