#!/usr/bin/env python3
"""PONTO DE DEPLOY -- escolher por ROBUSTEZ, nao por otimo.

POR QUE MUDAR O CRITERIO. A validacao temporal (validacao_temporal.py) mostrou
que a ESTRUTURA de dois niveis generaliza (3/3 nos eventos nunca vistos contra
2/3 do v1) mas os LIMIARES sao ajustados: uma selecao honesta, olhando so o
passado, escolhe outros valores e vai pior que o proprio v1 no futuro.

Em producao o detector so enfrenta evento futuro. Entao o ponto certo nao e o que
maximiza o numero nos 8 que temos -- e o que **nao desaba quando o dado muda**.

CRITERIO MINIMAX: para cada candidato, avalia o PIOR CASO na sua vizinhanca
(+-1 passo de grade em cada eixo) e escolhe o candidato cujo pior caso e melhor.
E pratica padrao quando a amostra e pequena, e ataca exatamente o que a validacao
temporal apontou: o `lo.t = 1,10` com margem de +-5%.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import K_LO, REFRAT_V2
from validacao_temporal import detecta, mede

TMIN, TMAX = 4.0, 48.0
GR = {"t": [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3],
      "p": [0.6, 0.7, 0.8, 1.0, 1.2],
      "sp": [0.7, 0.9, 1.1, 1.3],
      "vb": [1.6, 1.8, 2.0, 2.2]}

# avalia todo o grid uma vez
print(f"avaliando {np.prod([len(v) for v in GR.values()])} configuracoes ...", flush=True)
R = {}
for kt, kp, ks, kv in itertools.product(*[GR[c] for c in ("t","p","sp","vb")]):
    klo = {"t": kt, "p": kp, "sp": ks, "vb": kv}
    al = detecta(klo, 1.7, 2.2, REFRAT_V2)
    b, i, fp, fpm, _ = mede(al, list(alvo))
    det = AV.avalia(al, alvo, mask)["det"]
    R[(kt, kp, ks, kv)] = (b, i, det, fpm)

def viz(key):
    """os vizinhos a +-1 passo de grade em cada eixo (inclui o proprio)."""
    out = []
    for j, c in enumerate(("t","p","sp","vb")):
        g = GR[c]; k = g.index(key[j])
        for d in (-1, 0, 1):
            if 0 <= k+d < len(g):
                nk = list(key); nk[j] = g[k+d]; out.append(tuple(nk))
    return set(out)

print("\nCRITERIO MINIMAX -- melhor PIOR CASO na vizinhanca")
print("=" * 104)
lin = []
for key, (b, i, det, fpm) in R.items():
    V = [R[v] for v in viz(key) if v in R]
    pior_b = min(x[0] for x in V); pior_det = min(x[2] for x in V)
    pior_fp = max(x[3] for x in V)
    lin.append((pior_b, pior_det, -pior_fp, key, b, i, det, fpm, pior_b, pior_det, pior_fp))
lin.sort(reverse=True)

print(f"{'k_lo t/p/sp/vb':>22} | {'--- no ponto ---':^26} | {'--- PIOR da vizinhanca ---':^30}")
print(f"{'':>22} | {'banda':>7}{'inicio':>8}{'det':>5}{'FP/mes':>8} | "
      f"{'banda':>7}{'det':>6}{'FP/mes':>9}")
print("-" * 104)
for row in lin[:10]:
    _, _, _, key, b, i, det, fpm, pb, pd_, pf = row
    mk = "  <<< o publicado" if key == tuple(K_LO[c] for c in ("t","p","sp","vb")) else ""
    print(f"{'/'.join(str(x) for x in key):>22} | {b:5d}/8{i:7d}/8{det:4d}/8{fpm:8.3f} | "
          f"{pb:5d}/8{pd_:5d}/8{pf:9.3f}{mk}")
print("-" * 104)

pub = tuple(K_LO[c] for c in ("t","p","sp","vb"))
if pub in R:
    V = [R[v] for v in viz(pub) if v in R]
    print(f"\n  O PUBLICADO {pub}:")
    print(f"    no ponto        : banda {R[pub][0]}/8, det {R[pub][2]}/8, {R[pub][3]:.3f} FP/mes")
    print(f"    pior vizinho    : banda {min(x[0] for x in V)}/8, "
          f"det {min(x[2] for x in V)}/8, {max(x[3] for x in V):.3f} FP/mes")
melhor = lin[0]
print(f"\n  MAIS ROBUSTO {melhor[3]}:")
print(f"    no ponto        : banda {melhor[4]}/8, det {melhor[6]}/8, {melhor[7]:.3f} FP/mes")
print(f"    pior vizinho    : banda {melhor[8]}/8, det {melhor[9]}/8, {melhor[10]:.3f} FP/mes")
