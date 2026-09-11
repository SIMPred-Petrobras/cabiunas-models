#!/usr/bin/env python3
"""IDEIA 6 -- portao de corroboracao por TEMPERATURA SUB-LIMIAR.

ACHADO (features_episodios.py): `pico_t` -- o pico da razao E_t/limiar_t dentro
do episodio -- separa TP de FP melhor que qualquer outra feature: mediana 0,98
contra 0,31, AUC 0,78. Note que 0,98 esta LOGO ABAIXO do limiar de alarme: a
temperatura sobe nas falhas reais sem necessariamente disparar o proprio canal.

Hoje o canal `t` so participa por cima do limiar (binario). Toda a informacao
sub-limiar e jogada fora. A regra a testar: exigir que o episodio tenha
`pico_t >= X` com X < 1 para ser creditado -- corroboracao termica fraca.

Fisicamente: falha de mancal ou de oleo dissipa energia e aquece. Coincidencia
de ruido em pressao e vibracao nao tem por que aquecer.

VALIDACAO: LOEO. Para cada episodio, o limiar X e escolhido nos OUTROS e
aplicado ao retirado. Sem isso, com 17 amostras, e so ajuste.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo
from publica_clearml import SIN, BASE
from combina_final import constroi, canais, mede
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

LO, HI, KV, IDA, AB = 1.0, 1.7, 2.2, 96, 20
al = constroi(LO, HI, KV, IDA, AB)
paradas = paradas_reais_2h()
cls = classifica_regra_c(AV.episodios(al), paradas)
K = {"t": HI, "p": HI, "sp": HI, "vb": KV}
RAZ_t = EW["t"].where(mask) / (BASE["t"]*K["t"])

reg = [(a, b, k, float(RAZ_t.loc[a:b].max())) for a, b, k, _ in cls]
print("pico_t POR EPISODIO, ORDENADO")
print("=" * 72)
print(f"{'inicio':>17} {'classe':>8} {'pico_t':>9}")
print("-" * 72)
for a, b, k, v in sorted(reg, key=lambda r: -r[3]):
    print(f"{a:%d/%m/%Y %H:%M} {k:>8} {v:9.3f}")

tp = [v for *_, k, v in reg if k == "TP"]
fp = [v for *_, k, v in reg if k == "FP"]
nt = [v for *_, k, v in reg if k == "NEUTRO"]
print(f"\n  TP     min {min(tp):.3f}  mediana {np.median(tp):.3f}")
print(f"  NEUTRO min {min(nt):.3f}  mediana {np.median(nt):.3f}")
print(f"  FP     min {min(fp):.3f}  mediana {np.median(fp):.3f}  max {max(fp):.3f}")

print("\n\nEFEITO DO PORTAO (in-sample)")
print("=" * 92)
print(f"{'X':>7} {'TP mantidos':>13} {'NEUTRO':>8} {'FP mantidos':>13} {'FP cortados':>13}")
print("-" * 92)
GRADE = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
for X in GRADE:
    ntp = sum(1 for v in tp if v >= X); nfp = sum(1 for v in fp if v >= X)
    nnt = sum(1 for v in nt if v >= X)
    print(f"{X:7.2f} {ntp:8d}/{len(tp)} {nnt:6d}/{len(nt)} {nfp:8d}/{len(fp)} "
          f"{len(fp)-nfp:13d}")

print("\n\nLOEO -- o limiar X escolhido SEM ver o episodio avaliado")
print("=" * 92)
print("  regra de escolha: maior X que nao corta nenhum TP do conjunto de treino")
print("-" * 92)
acertos = erros = cortes_fp = mantidos_fp = 0
for i, (a, b, k, v) in enumerate(reg):
    if k == "NEUTRO":
        continue
    treino = [(kk, vv) for j, (_, _, kk, vv) in enumerate(reg) if j != i and kk in ("TP", "FP")]
    tp_tr = [vv for kk, vv in treino if kk == "TP"]
    X = min(tp_tr)                       # maior X que preserva todo TP de treino
    passa = v >= X
    if k == "TP":
        acertos += passa; erros += (not passa)
    else:
        cortes_fp += (not passa); mantidos_fp += passa
    print(f"  {a:%d/%m/%Y} {k:>3}  pico_t {v:6.3f}  X(treino) {X:6.3f}  -> "
          f"{'mantido' if passa else 'CORTADO'}")
print("-" * 92)
print(f"  TP preservados : {acertos}/{len(tp)}   TP perdidos: {erros}")
print(f"  FP cortados    : {cortes_fp}/{len(fp)}   FP mantidos: {mantidos_fp}")
