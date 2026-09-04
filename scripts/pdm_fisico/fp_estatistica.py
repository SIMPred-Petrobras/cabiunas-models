#!/usr/bin/env python3
"""O recall (7/8) tem amostra pequena (n=9 trips) -- ja tratado. O FP/mes vem
de uma amostra MUITO maior (13 mil horas de operacao pontuavel), entao a
pergunta estatistica aqui e diferente: nao e "sera que a taxa e essa mesmo",
e sim "essa taxa e ESTAVEL, ou os episodios se aglomeram de um jeito que faz
'X FP/mes' enganar como expectativa de regime permanente".

Tres checagens:
  1. dispersao mes-a-mes: variancia >> media = aglomeracao (Poisson simples
     nao serve, a taxa nao e constante no tempo).
  2. IC de Poisson exato na taxa global.
  3. taxa de treino (pre-corte) e estatisticamente igual a de teste
     (pos-corte)? -- teste binomial exato sobre a divisao dos episodios,
     ponderado pela exposicao (horas pontuaveis) de cada lado.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO

K_BASE, K_VIB = 1.3, 5.5


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)

    out = roda(BRACO, df, falhas)
    alerta = alerta_2k(out, mask, K_BASE, K_VIB)
    eps = A.episodios(alerta)

    # episodios que NAO batem em nenhuma janela de deteccao de 48h = falso positivo
    jan = [(t - pd.Timedelta(hours=48), t) for t in falhas]
    fp_eps = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan)]
    print(f"episodios totais: {len(eps)}  |  falsos positivos: {len(fp_eps)}")

    # -------- 1. dispersao mes-a-mes
    meses_idx = mask.index.to_period("M")
    horas_mes = mask.groupby(meses_idx).sum() * 2 / 60.0
    horas_mes = horas_mes[horas_mes > 0]
    contagem = pd.Series(0, index=horas_mes.index)
    for a, _ in fp_eps:
        p = pd.Period(a, freq="M")
        if p in contagem.index:
            contagem[p] += 1
    print("\nepisodios de FP por mes (so meses com operacao pontuavel):")
    print(contagem.to_string())
    media, var = contagem.mean(), contagem.var(ddof=1)
    print(f"\nmedia={media:.2f}  variancia={var:.2f}  indice de dispersao (var/media)={var/media:.2f}")
    print("(Poisson 'bem comportado' teria var/media ~= 1; >>1 = aglomeracao real)")
    # teste formal: estatistica de dispersao ~ qui-quadrado(n-1) sob H0 Poisson
    n_meses = len(contagem)
    chi2_disp = (n_meses - 1) * var / media
    p_disp = 1 - stats.chi2.cdf(chi2_disp, df=n_meses - 1)
    print(f"teste de dispersao: chi2={chi2_disp:.1f}, df={n_meses-1}, p={p_disp:.4f} "
          f"(p pequeno = rejeita Poisson simples, confirma aglomeracao)")

    # -------- 2. IC de Poisson exato na taxa global
    total_fp = len(fp_eps)
    total_horas = horas_mes.sum()
    total_meses_op = total_horas / (365.25 * 24 / 12)   # mes-operacao ~ 730h
    lo = stats.chi2.ppf(0.025, 2 * total_fp) / 2 if total_fp > 0 else 0
    hi = stats.chi2.ppf(0.975, 2 * (total_fp + 1)) / 2
    print(f"\ntotal: {total_fp} FP em {total_meses_op:.1f} meses-operacao "
          f"({total_horas:.0f}h pontuaveis)")
    print(f"taxa observada: {total_fp/total_meses_op:.2f} FP/mes")
    print(f"IC 95% de Poisson exato: [{lo/total_meses_op:.2f}, {hi/total_meses_op:.2f}] FP/mes")

    # -------- 3. taxa treino vs teste (binomial ponderado por exposicao)
    tr = mask.index < CORTE
    horas_tr = mask[tr].sum() * 2 / 60.0
    horas_te = mask[~tr].sum() * 2 / 60.0
    fp_tr = sum(1 for a, _ in fp_eps if a < CORTE)
    fp_te = len(fp_eps) - fp_tr
    p_exposicao_te = horas_te / (horas_tr + horas_te)
    bt = stats.binomtest(fp_te, fp_tr + fp_te, p_exposicao_te)
    print(f"\nFP treino: {fp_tr} em {horas_tr:.0f}h ({fp_tr/(horas_tr/730):.2f}/mes)")
    print(f"FP teste : {fp_te} em {horas_te:.0f}h ({fp_te/(horas_te/730):.2f}/mes)")
    print(f"H0: mesma taxa dos dois lados (esperado {p_exposicao_te:.1%} dos FP no teste, "
          f"proporcional a exposicao)")
    print(f"teste binomial exato: p={bt.pvalue:.4f} "
          f"{'(rejeita H0 -- taxas diferentes)' if bt.pvalue<0.05 else '(nao rejeita H0)'}")


main()
