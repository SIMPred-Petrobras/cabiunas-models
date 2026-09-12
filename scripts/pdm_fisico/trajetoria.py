#!/usr/bin/env python3
"""IDEIA A -- casamento de TRAJETORIA multivariada.

POR QUE E DIFERENTE DE TUDO QUE JA FOI TESTADO. Os quatro canais construidos
nesta sessao (sp_vib, modo comum, margem, inovacao) sao todos MELHORES
individualmente e nenhum melhora o detector -- porque canais esparsos nao
coincidem e nao conseguem corroborar. Este desenho nao exige coincidencia: usa a
correlacao temporal como informacao em vez de descarta-la.

Todos os detectores dos quatro times pontuam cada instante ISOLADAMENTE e depois
olham persistencia. Ninguem modelou a FORMA no tempo. A pericia mostrou que em
8/8 eventos ha 24 a 44 h com >=2 canais simultaneos -- isso e uma forma, e
estamos reduzindo a uma contagem.

DESENHO
  vetor  : 4 canais x 24 pontos (a cada 2 h nas ultimas 48 h) = 96 dims,
           em log10(E/limiar) -- ja comparavel entre canais (0 = no limiar)
  score  : -min distancia euclidiana aos templates dos trips
  LOEO   : ao avaliar o evento i, o template i SAI do conjunto. Por construcao,
           nao ha como o resultado ser auto-casamento.
  grade  : avalia a cada 30 min (nao a cada 2 min) -- 24x menos calculo, e a
           trajetoria de 48 h nao muda em 2 min
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import EW, mask, idx, alvo, op
from publica_clearml import SIN, BASE
from plota_estilo_francisco import KB, KV

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
JAN_H, N_PT, PASSO_AVAL = 48.0, 24, 15        # 24 pontos a cada 2 h; avalia a cada 30 min
RNG = np.random.default_rng(20260911)

# razao E/limiar em log10, ja comparavel entre canais (0 = exatamente no limiar)
R = np.column_stack([np.log10(np.clip(
        (EW[c].where(mask)/(BASE[c]*K[c])).to_numpy(), 1e-2, 1e2)) for c in SIN])
n_2min = len(idx)
off = np.linspace(0, JAN_H*30, N_PT).astype(int)     # 48h/2min = 1440; 24 pontos

def vetor(i):
    """trajetoria de 48 h terminando em i, 96 dims. NaN onde a janela esta
    mascarada (maquina parada) -- a distancia trata isso, nao descarta."""
    j = i - off[::-1]
    if j[0] < 0: return None
    return R[j].reshape(-1)

# instantes avaliaveis
val = np.flatnonzero(mask.to_numpy() & op.to_numpy())
val = val[(val >= 1440) & (val % PASSO_AVAL == 0)]
V, IDXV = [], []
for i in val:
    v = vetor(i)
    # exige ao menos metade da trajetoria observada
    if v is not None and np.isfinite(v).mean() >= 0.5:
        V.append(v); IDXV.append(i)
V = np.asarray(V, dtype="float32"); IDXV = np.asarray(IDXV)
print(f"instantes avaliaveis: {len(V):,}  (grade de {PASSO_AVAL*2} min)")
print(f"dimensao do vetor: {V.shape[1]}  ({len(SIN)} canais x {N_PT} pontos)")

pos_alvo = np.array([idx.get_indexer([t], method="nearest")[0] for t in alvo])
T = np.asarray([vetor(p) for p in pos_alvo], dtype="float32")
cob = np.isfinite(T).mean(axis=1)
print(f"cobertura das trajetorias dos templates: "
      + ", ".join(f"{100*c:.0f}%" for c in cob))

def dist(A, B):
    """distancia euclidiana MEDIA sobre as dimensoes finitas nos dois -- tolera
    janela parcialmente mascarada sem descartar o template."""
    out = np.empty((len(A), len(B)), dtype="float32")
    for j in range(len(B)):
        d = A - B[j]
        ok = np.isfinite(d)
        n = ok.sum(axis=1)
        sq = np.where(ok, d*d, 0.0).sum(axis=1)
        out[:, j] = np.where(n > 0, np.sqrt(sq/np.maximum(n, 1))*np.sqrt(A.shape[1]), np.inf)
    return out

D = dist(V, T)
print(f"matriz de distancia: {D.shape}")
print(f"  distancia mediana ao template mais proximo: {np.median(D.min(axis=1)):.2f}")

np.savez_compressed("trajetoria_cache.npz", D=D, IDXV=IDXV, pos_alvo=pos_alvo)
print("-> trajetoria_cache.npz")

# quao parecidos sao os templates ENTRE SI? se forem muito diferentes, nao ha forma comum
DT = dist(T, T)
np.fill_diagonal(DT, np.nan)
print(f"\nDISTANCIA ENTRE OS PROPRIOS TEMPLATES")
print(f"  mediana {np.nanmedian(DT):.2f} | min {np.nanmin(DT):.2f} | max {np.nanmax(DT):.2f}")
print(f"  distancia tipica de um instante qualquer ao template mais proximo: "
      f"{np.median(D.min(axis=1)):.2f}")
if np.nanmedian(DT) > np.median(D.min(axis=1)):
    print("  -> os templates sao MAIS distantes entre si do que um ponto qualquer")
    print("     esta de um template: NAO HA FORMA COMUM. O metodo nao tem base.")
else:
    print("  -> os templates se parecem mais entre si do que o ruido: ha forma comum")
