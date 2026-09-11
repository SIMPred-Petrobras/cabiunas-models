#!/usr/bin/env python3
"""IDEIA 2 -- a ORDEM em que os canais acendem separa TP de FP?

MOTIVACAO FISICA. Um modo de falha tem assinatura temporal, nao so amplitude:
  mancal      -> vibracao sobe primeiro, temperatura depois (calor e consequencia)
  oleo        -> pressao primeiro, depois temperatura, depois vibracao
  termico     -> temperatura primeiro
Coincidencia de ruido nao deve ter ordem consistente. Se os TP concentram uma
ordem e os FP se espalham, isso e um discriminante com quase nenhum parametro
livre -- que e o unico tipo que 14 rotulos suportam.

Nunca usamos essa informacao: os quatro canais entram numa CONTAGEM (voto >= 2),
que e invariante a permutacao. Jogamos fora a ordem inteira.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, pos, mask, idx, alvo
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
from corte_com_rearme import corta_rearma
from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
ON = partes(KB, KV); ns = sum(ON[c].astype(int) for c in SIN)
v0 = (pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])).to_numpy()
al = pos(pd.Series(corta_rearma(v0, F, 0.03), index=idx), ns, 72, DUR_MIN, False)
eps = AV.episodios(al)
cls = classifica_regra_c(eps, paradas_reais_2h())

# ---- para cada episodio: ordem de acendimento e atraso entre canais ----------
def assinatura(a, b):
    """Primeiro instante em que cada canal acende dentro do episodio, e a ordem."""
    prim = {}
    for c in SIN:
        s = ON[c].loc[a:b]
        s = s[s.fillna(False)]
        if len(s):
            prim[c] = s.index[0]
    if not prim:
        return None, {}, None
    t0 = min(prim.values())
    atraso = {c: (t - t0).total_seconds()/3600 for c, t in prim.items()}
    ordem = tuple(sorted(prim, key=lambda c: prim[c]))
    return ordem, atraso, len(prim)

print("ASSINATURA TEMPORAL POR EPISODIO")
print("=" * 100)
print(f"{'inicio':>17} {'classe':>8} {'dur':>7} {'ordem de acendimento':>26}  atrasos (h)")
print("-" * 100)
linhas = []
for a, b, k, _ in cls:
    ordem, atraso, n = assinatura(a, b)
    if ordem is None:
        continue
    at = " ".join(f"{c}+{atraso[c]:.1f}" for c in ordem)
    print(f"{a:%d/%m/%Y %H:%M} {k:>8} {(b-a).total_seconds()/3600:6.1f}h "
          f"{'->'.join(ordem):>26}  {at}")
    linhas.append(dict(ini=a, classe=k, ordem=ordem, primeiro=ordem[0], n=n,
                       dur=(b-a).total_seconds()/3600, **{f"at_{c}": atraso.get(c, np.nan) for c in SIN}))

d = pd.DataFrame(linhas)
print("\n\nPRIMEIRO CANAL A ACENDER, POR CLASSE")
print("=" * 60)
tab = pd.crosstab(d.primeiro, d.classe)
print(tab.to_string())
print("\nORDEM COMPLETA, POR CLASSE")
print("=" * 60)
for k in ["TP", "NEUTRO", "FP"]:
    s = d[d.classe == k]
    print(f"  {k} ({len(s)}):")
    for o, n in s.ordem.value_counts().items():
        print(f"      {'->'.join(o):<26} {n}")
