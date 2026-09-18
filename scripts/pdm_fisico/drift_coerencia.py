#!/usr/bin/env python3
"""O FIT_POINTS e o TEMPO DE COERENCIA DO REGIME, nao um hiperparametro.

A HIPOTESE, depois de sete tecnicas refutadas. Todas as alternativas ao baseline
atual quebram, e agora por um motivo so:

  janela MENOR  -> amostra pequena demais, o estimador treme
  janela MAIOR  -> atravessa mudanca de regime, e a escala e dominada por
                   residuos de drift (medido: amplitude de 2,45x para 1279x)

Entao 20.000 pontos (~28 dias estaveis) nao e um numero escolhido: e
aproximadamente a DURACAO DE UM REGIME HOMOGENEO nesta maquina. O valor certo e
uma propriedade fisica do processo, nao do algoritmo -- e e por isso que a
superficie tem um pico estreito em vez de um plato.

TESTE DA HIPOTESE. Se o baseline de 20.000 pontos e homogeneo, o erro do PCA na
primeira metade dele deve ser parecido com o da segunda. Se uma janela maior
atravessa regime, a diferenca entre as metades cresce com o tamanho.

E a consequencia pratica e melhor do que monitorar o p99: o teste de homogeneidade
roda no PROPRIO baseline, no momento do retreino, sem esperar o mes seguinte.

RESULTADO: HIPOTESE NAO CONFIRMADA.

    FIT_POINTS   ~dias   t: p90   p: p90   p: max
       8.000       11     4,90     4,16     12,11
      14.000       19     4,06     2,44      6,02
      20.000       28     3,49     2,47     10,48   <- atual
      30.000       42     2,27     2,64     28,28
      45.000       62     2,93     5,41    135,57
      70.000       97     2,87    27,37    160,13

Nenhum baseline e homogeneo em tamanho nenhum -- a razao mediana entre as metades
fica em ~1,4 em toda a faixa. E NAO HA MINIMO EM 20.000: os minimos caem em
14.000 e 30.000, dependendo do canal e da estatistica.

Confirma-se so a metade fraca: acima de ~45.000 pontos (62 dias) o baseline
atravessa mudancas de regime severas (razao de 135 a 160 entre as metades), o que
explica por que janelas grandes quebram. Mas o pico em 20.000 segue sem
explicacao estrutural -- ver `drift_vizinhanca_fit.py`, que testa a terceira
hipotese (coincidencia) e a confirma.

Uso:  PYTHONPATH=. python drift_coerencia.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from cabiunas_pdm import config as C, detector as DET
from ablacao import ScorerMax
from drift_baseline import DF, STABLE, IX

FAM = {"t": C.TEMPERATURE_TAGS, "p": C.PRESSURE_TAGS}


def homogeneidade(fit_points: int):
    """Para cada mes: razao entre o p99 do erro na 2a e na 1a metade do baseline.

    1,0 = baseline homogeneo. Longe de 1 = a janela atravessa mudanca de regime,
    e as duas metades nao descrevem a mesma maquina."""
    meses = pd.date_range(IX[0].normalize().replace(day=1), IX[-1], freq="MS", tz="UTC")
    razoes = {"t": [], "p": []}
    for m0 in meses:
        fit = DF.loc[STABLE & (IX < m0), C.SENSOR_TAGS].dropna().tail(fit_points)
        if len(fit) < fit_points // 2:
            continue
        meio = len(fit) // 2
        for c, tags in FAM.items():
            sc = ScorerMax().fit(fit[tags])
            r1 = sc._raw_scores(fit[tags].iloc[:meio])[0]
            r2 = sc._raw_scores(fit[tags].iloc[meio:])[0]
            r1 = r1[np.isfinite(r1)]; r2 = r2[np.isfinite(r2)]
            if len(r1) < 100 or len(r2) < 100:
                continue
            razoes[c].append(float(np.percentile(r2, 99) / max(np.percentile(r1, 99), 1e-9)))
    return razoes


if __name__ == "__main__":
    print("HOMOGENEIDADE DO BASELINE -- p99 do erro na 2a metade / 1a metade")
    print("Valor longe de 1,0 significa que a janela atravessa mudanca de regime.\n")
    print(f"{'FIT_POINTS':>12}{'~dias':>8}   {'canal t: mediana':>18}{'p90':>9}"
          f"{'max':>9}   {'canal p: mediana':>18}{'p90':>9}{'max':>9}")
    print("-" * 100)
    for fp in (8_000, 14_000, 20_000, 30_000, 45_000, 70_000):
        r = homogeneidade(fp)
        if not r["t"]:
            continue
        def q(v):
            a = np.abs(np.log(np.array(v)))   # distancia simetrica de 1,0
            return (np.exp(np.median(a)), np.exp(np.percentile(a, 90)), np.exp(a.max()))
        mt, p9t, mxt = q(r["t"]); mp, p9p, mxp = q(r["p"])
        marca = "  <- atual" if fp == 20_000 else ""
        print(f"{fp:>12,}{fp*2/60/24:>8.0f}   {mt:>18.2f}{p9t:>9.2f}{mxt:>9.2f}"
              f"   {mp:>18.2f}{p9p:>9.2f}{mxp:>9.2f}{marca}".replace(",", "."))
