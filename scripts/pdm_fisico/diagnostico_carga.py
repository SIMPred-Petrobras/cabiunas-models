#!/usr/bin/env python3
"""Os nossos sinais sao contaminados por PONTO DE OPERACAO?

VERIFICADO EM 10/09/2026: nao ha condicionamento a carga em lugar nenhum. O
`pos_processamento.py` usa so um portao binario (`T5_AVG_A > 300`), e a PCA e
ajustada sobre os sensores CRUS -- aprende UM modelo de normalidade para todos os
pontos de operacao. O Diego tem portao de degrau e de rampa de carga por canal;
nos nao temos equivalente.

Num turbocompressor, temperatura de mancal e vibracao escalam com carga, rotacao
e pressao de succao. Se o residuo nao e condicionado, entao:
  - mudanca de carga PARECE anomalia          -> falso positivo
  - degradacao em carga baixa fica MASCARADA  -> deteccao perdida
  - e o instante do disparo passa a depender do ponto de operacao, nao da
    degradacao -- que e exatamente o nosso defeito medido (lead de 1,3 a 194,9 h)

Este script mede se a suspeita procede, ANTES de construir qualquer coisa.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, g
from publica_clearml import SIN, BASE
from plota_estilo_francisco import alarme, KB, KV, paradas_reais_2h, classifica_regra_c

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
RAZ = {c: EW[c].where(mask) / (BASE[c]*K[c]) for c in SIN}
carga = g["T5_AVG_A"].where(mask)
dcarga = carga.diff().abs().rolling(30, min_periods=5).mean()   # 1 h de variacao

print("1. CORRELACAO DOS SINAIS COM A CARGA (T5_AVG_A), em operacao")
print("=" * 76)
print(f"{'canal':>6} {'corr com carga':>16} {'corr com |dcarga|':>19}")
print("-" * 76)
for c in SIN:
    s = RAZ[c]
    v = s.notna() & carga.notna()
    r1 = float(np.corrcoef(s[v], carga[v])[0, 1])
    v2 = s.notna() & dcarga.notna()
    r2 = float(np.corrcoef(s[v2], dcarga[v2])[0, 1])
    flag = "  <<< forte" if abs(r1) > 0.3 or abs(r2) > 0.3 else ""
    print(f"{c:>6} {r1:16.3f} {r2:19.3f}{flag}")

print("\n\n2. O SINAL SOBE QUANDO A CARGA MUDA?  (por quintil de |dcarga|)")
print("=" * 76)
q = pd.qcut(dcarga.dropna(), 5, labels=["muito baixa", "baixa", "media", "alta", "muito alta"])
print(f"{'|dcarga|':>13} " + "".join(f"{c:>10}" for c in SIN))
print("-" * 76)
for lab in ["muito baixa", "baixa", "media", "alta", "muito alta"]:
    m = q[q == lab].index
    print(f"{lab:>13} " + "".join(f"{float(RAZ[c].reindex(m).mean()):10.3f}" for c in SIN))

print("\n\n3. OS NOSSOS FP COINCIDEM COM TRANSICAO DE CARGA?")
print("=" * 90)
al = alarme(); cls = classifica_regra_c(AV.episodios(al), paradas_reais_2h())
lim = float(dcarga.quantile(0.90))
print(f"   limiar de 'carga mudando': p90 de |dcarga| = {lim:.3f} C/2min")
print(f"\n{'episodio':>17} {'classe':>8} {'carga media':>13} {'|dcarga| med':>14} {'% em transicao':>16}")
print("-" * 90)
res = {}
for a, b, k, _ in cls:
    cm = float(carga.loc[a:b].mean()); dm = float(dcarga.loc[a:b].mean())
    pt = float((dcarga.loc[a:b] > lim).mean())
    res.setdefault(k, []).append(pt)
    print(f"{a:%d/%m/%Y %H:%M} {k:>8} {cm:13.1f} {dm:14.3f} {100*pt:15.1f}%")
print("-" * 90)
for k in ("TP", "NEUTRO", "FP"):
    if k in res:
        print(f"  {k:>7}: fracao media do episodio em transicao de carga = "
              f"{100*np.mean(res[k]):.1f}%  (n={len(res[k])})")
base = float((dcarga > lim).mean())
print(f"  {'base':>7}: {100*base:.1f}% do tempo em operacao")
