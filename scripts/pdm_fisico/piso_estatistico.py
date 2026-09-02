#!/usr/bin/env python3
"""Quanta melhoria este conjunto de dados AINDA CONSEGUE MEDIR.

A pergunta que fecha a sessao. Quatro ataques independentes ao ponto de operacao bateram
no mesmo piso (8/8 a 1,033 FP/mes): geometria do transiente (nao_decaimento.py, 1152
pontos), pos-processamento (pos_processamento.py, 720 pontos), autocalibracao por
percentil (autocalibra.py) e ordenacao por gravidade (severidade.py). Quando quatro
caminhos que nao se falam param no mesmo lugar, ha duas leituras possiveis e elas pedem
acoes opostas:

  (a) o detector esta no seu otimo -- pare de mexer nele;
  (b) a REGUA nao enxerga mais nada -- pare de acreditar em qualquer diferenca medida.

Este script decide entre as duas pelo unico caminho valido: o tamanho da barra de erro.
Com 8 eventos e 12 falsos positivos em 11,6 meses, a incerteza de amostragem sozinha ja
pode ser maior que tudo que estamos disputando. Se for, entao "melhorar o resultado"
nao e um problema de modelagem -- e um problema de quantidade de evidencia, e a resposta
muda de "ajuste o detector" para "colete mais eventos ou mude o alvo".

Nada aqui e ajustado. So intervalo de confianca e poder.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

DET, N_EV = 8, 8
N_FP, MESES = 12, 11.6
RUIDO_RETREINO_PP = 20.7      # medido: dois treinos de config identica (memoria do projeto)


def wilson(k, n, alfa=0.05):
    z = stats.norm.ppf(1 - alfa / 2)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def poisson_ic(k, expo, alfa=0.05):
    lo = stats.chi2.ppf(alfa / 2, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(1 - alfa / 2, 2 * k + 2) / 2
    return lo / expo, hi / expo


def n_para_distinguir(l0, l1, alfa=0.05, poder=0.80):
    """Meses de operacao necessarios para separar duas taxas de FP (Poisson, dois lados)."""
    za, zb = stats.norm.ppf(1 - alfa / 2), stats.norm.ppf(poder)
    return ((za * np.sqrt(l0) + zb * np.sqrt(l1)) / (l0 - l1)) ** 2


if __name__ == "__main__":
    print("=" * 92)
    print("1. O QUE 8 DE 8 REALMENTE DIZ SOBRE O RECALL")
    print("=" * 92)
    lo, hi = wilson(DET, N_EV)
    print(f"  recall pontual      : {DET}/{N_EV} = 100%")
    print(f"  IC 95% (Wilson)     : [{lo*100:.0f}%, {hi*100:.0f}%]")
    print(f"  -> um detector cujo recall VERDADEIRO fosse {lo*100:.0f}% acertaria 8 de 8")
    print(f"     nesta amostra com probabilidade nao desprezivel. O '8/8' e compativel")
    print(f"     com qualquer recall acima de {lo*100:.0f}%.")
    print(f"  cada evento vale {100/N_EV:.1f} pontos percentuais de recall.")
    print(f"  o ruido de retreino ja medido no projeto e de {RUIDO_RETREINO_PP:.1f} pp,")
    print(f"  ou seja, {RUIDO_RETREINO_PP/(100/N_EV):.1f} eventos: MAIOR que qualquer diferenca")
    print(f"  de deteccao que esta amostra consegue resolver.")

    print("\n" + "=" * 92)
    print("2. O QUE 1,03 FP/MES REALMENTE DIZ SOBRE O CUSTO")
    print("=" * 92)
    tx = N_FP / MESES
    lo, hi = poisson_ic(N_FP, MESES)
    print(f"  taxa pontual        : {N_FP} FP em {MESES:.1f} meses = {tx:.2f} FP/mes")
    print(f"  IC 95% (Poisson)    : [{lo:.2f}, {hi:.2f}] FP/mes")
    print(f"  desvio-padrao da taxa: {np.sqrt(N_FP)/MESES:.2f} FP/mes")
    print()
    print(f"  a porta de mancal moveu o custo de 1,12 para 1,03 FP/mes.")
    print(f"  isso e UM episodio a menos em 11,6 meses, ou {1/MESES/(np.sqrt(N_FP)/MESES):.2f}")
    print(f"  desvio-padrao. Nao e um efeito: e ruido de contagem.")

    print("\n" + "=" * 92)
    print("3. QUANTOS MESES DE OPERACAO PARA MEDIR UMA MELHORIA DE VERDADE")
    print("=" * 92)
    print(f"  {'reduzir 1,03 FP/mes para':>26} {'meses necessarios':>18} {'~eventos esperados':>20}")
    taxa_ev = N_EV / MESES
    for alvo in (0.90, 0.75, 0.60, 0.50, 0.35, 0.25):
        m = n_para_distinguir(tx, alvo)
        print(f"  {alvo:>24.2f}   {m:>16.0f}   {m*taxa_ev:>18.0f}")
    print()
    print(f"  hoje temos {MESES:.1f} meses. Para provar que uma mudanca levou o custo de")
    print(f"  1,03 a 0,75 FP/mes seriam necessarios {n_para_distinguir(tx, 0.75):.0f} meses -- "
          f"{n_para_distinguir(tx, 0.75)/MESES:.0f}x o que existe.")

    print("\n" + "=" * 92)
    print("4. VEREDITO")
    print("=" * 92)
    print("  A regua chegou ao fim antes do detector. As diferencas que restam disputar")
    print("  (uma decima de FP/mes, um evento de recall) sao MENORES que a barra de erro")
    print("  da propria amostra. Qualquer ganho medido daqui em diante nesta base e")
    print("  indistinguivel de sorte, e adotar o que 'ganhou' e sobreajuste com outro nome.")
    print()
    print("  Consequencia pratica: parar de otimizar contra estes 8 eventos. O caminho")
    print("  para um resultado melhor deixa de ser modelagem e passa a ser evidencia --")
    print("  mais eventos (tempo de operacao ou outras maquinas da familia) ou um alvo")
    print("  com mais positivos por unidade de tempo.")
