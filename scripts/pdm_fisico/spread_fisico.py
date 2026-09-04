#!/usr/bin/env python3
"""O spread do mancal cresceu em GRAUS, ou so o z cresceu?

deriva_origem.py achou o unico sinal com tendencia clara: a mediana do z do spread dos
termopares de mancal vai de 1,08 -> 1,48 -> 0,75 -> 3,06 -> 4,25 ao longo dos cinco
semestres, e o quantil implicado do limiar fixo cai de 100 para 70,5. Os outros tres
sinais nao tem tendencia. Isso separa duas historias com consequencias opostas:

  DEGRADACAO REAL -- o spread em GRAUS CELSIUS cresceu. Entao parte do que a regua chama
    de falso positivo e o detector reportando corretamente uma maquina que esta piorando,
    e "estabilizar o custo" seria calibrar para nao ver a degradacao. Consequencia: o
    numero de FP nao deve ser estabilizado, deve ser explicado.

  ARTEFATO DE NORMALIZACAO -- o spread em graus esta igual, mas o MAD da referencia
    encolheu, entao o mesmo desvio fisico vira um z maior. Consequencia: o normalizador
    esta quebrado e o custo cresce por construcao.

O z e (spread - mediana) / MAD, as duas da referencia rolante de 400 h. Basta olhar as
tres pecas separadamente por semestre. Tambem se olha termopar a termopar, porque a
autopsia de 2025-11-04 ja mostrou o TI_0305 subindo a 119,6 degC com os tres irmaos em
~63 degC -- se a degradacao for real, deve ter dono.

Nota sobre o CFAR: a correcao que eu recomendei (limiar por alvo de duty) foi medida em
deriva_origem.py e e MUITO pior -- deteccao cai de 8/9 para 1-5/9 e o duty nem estabiliza.
O motivo aparece aqui: se o escore sobe porque a maquina degrada, um limiar ancorado no
quantil recente sobe junto e calibra para fora exatamente o que deveria detectar.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
from ablacao import canonico, mascara_pontuacao

MANC = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0305", "954005_624_TI_0307"]


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    idx = df.index
    mask = mascara_pontuacao(df)
    sems = [x.index[0] for _, x in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS"))
            if len(x) and (mask & (idx >= x.index[0]) & (idx <= x.index[-1])).sum() * 2 / 60 >= 300]
    jan = [((idx >= a) & (idx <= (sems[i+1] if i+1 < len(sems) else idx[-1] + pd.Timedelta("2min"))))
           for i, a in enumerate(sems)]

    sp = DET._spread_mancal(df)          # a grandeza fisica, em graus
    print("=== SPREAD DO MANCAL EM GRAUS CELSIUS (so operacao pontuavel) ===")
    print(f"{'semestre':>11} {'mediana':>9} {'p90':>8} {'p99':>8} {'MAD':>8}")
    for i, a in enumerate(sems):
        v = sp[mask & jan[i]].dropna()
        mad = float((v - v.median()).abs().median() * 1.4826)
        print(f"{a.date()!s:>11} {v.median():9.3f} {v.quantile(.90):8.3f} "
              f"{v.quantile(.99):8.3f} {mad:8.3f}")

    print("\n=== TERMOPARES DO MANCAL: mediana em graus, por semestre ===")
    print(f"{'semestre':>11} " + " ".join(f"{t.split('_')[-2]+'_'+t.split('_')[-1]:>9}" for t in MANC)
          + f" {'amplitude':>10}")
    for i, a in enumerate(sems):
        meds = [g[t][mask & jan[i]].median() for t in MANC]
        print(f"{a.date()!s:>11} " + " ".join(f"{m:9.2f}" for m in meds)
              + f" {max(meds)-min(meds):10.2f}")

    print("\n=== decomposicao do z: numerador (desvio em graus) x denominador (MAD) ===")
    print("    z = (spread - mediana_ref) / MAD_ref, ambas da referencia rolante de 400 h")
    print(f"{'semestre':>11} {'spread med':>11} {'MAD (graus)':>12} {'z implicito':>12}")
    for i, a in enumerate(sems):
        v = sp[mask & jan[i]].dropna()
        mad = float((v - v.median()).abs().median() * 1.4826)
        print(f"{a.date()!s:>11} {v.median():11.3f} {mad:12.4f} {v.median()/max(mad,1e-9):12.2f}")

    print("\n  leitura: se 'spread med' cresce e o MAD fica estavel -> degradacao fisica real.")
    print("           se 'spread med' fica estavel e o MAD encolhe   -> artefato do normalizador.")


main()
