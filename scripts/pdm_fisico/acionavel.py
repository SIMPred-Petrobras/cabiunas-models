#!/usr/bin/env python3
"""Duas perguntas que nao dependem de nenhum evento novo.

1. AS DETECCOES SAO ACIONAVEIS? Duas das antecedencias obtidas estao abaixo de
   3 h. Se a operacao precisa de N horas para agir (parada controlada em vez de
   trip), a recall efetiva e menor que 7/8. Aqui a recall vira funcao do tempo
   minimo util, em vez de um numero so.

2. O NIVEL DE ATENCAO ANTECEDE? O detector tem dois patamares por construcao --
   >=1 sinal (atencao) e >=2 (confirmado) -- mas so o segundo e usado. Se
   atencao aparece muito antes, existe aviso precoce sendo descartado, e e
   exatamente a materia-prima de um indice continuo. O preco e o quanto o
   patamar de atencao fica ligado.

Configuracao: a que venceu 7 dos 8 folds do leave-one-out (k_base=1.7,
k_vib=2.2), nao a escolhida com a amostra inteira -- as antecedencias aqui
devem ser as honestas.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO

K_BASE, K_VIB = 1.7, 2.2      # venceu 7 dos 8 folds do LOEO
JANELA_H = 48.0               # a regua do projeto; leads iguais a 48 sao truncados


def main():
    df = canonico()
    falhas_todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    falhas = falhas_todas[falhas_todas >= "2025-01-01"].reset_index(drop=True)
    mask = mascara_pontuacao(df)
    idx = mask.index
    meses = mask.sum() * 2 / 60 / 730
    cal_meses = (idx[-1] - idx[0]).total_seconds() / 3600 / 730
    jan48 = [(t - pd.Timedelta(hours=JANELA_H), t) for t in falhas_todas]

    out = roda(BRACO, df, falhas_todas)

    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    S = {"t": DET._sustained(ew("t", "1h"), DET.THR_FAM * K_BASE),
         "p": DET._sustained(ew("p", "1h"), DET.THR_FAM * K_BASE),
         "sp": DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * K_BASE),
         "vb": DET._sustained(ew("vb", "30min"), 3.0 * K_VIB)}
    n = sum(s.astype(int) for s in S.values())

    niveis = {"atencao (>=1 sinal)": (n >= 1) & mask,
              "CONFIRMADO (>=2 sinais)": (n >= 2) & mask}

    def lead(al, ev):
        w = al[(al.index >= ev - pd.Timedelta(hours=JANELA_H)) & (al.index < ev)]
        w = w[w]
        return (ev - w.index[0]).total_seconds() / 3600 if len(w) else np.nan

    print("=== antecedencia por evento e por patamar (horas) ===")
    print(f"{'evento':>12} " + "".join(f"{k:>26}" for k in niveis))
    leads = {}
    for nome, al in niveis.items():
        leads[nome] = [lead(al, ev) for ev in falhas]
    for i, ev in enumerate(falhas):
        cel = []
        for nome in niveis:
            v = leads[nome][i]
            cel.append("       nao visto" if np.isnan(v)
                       else (f"  >= {v:.0f} (truncado)" if v >= JANELA_H - .05 else f"  {v:.1f}"))
        print(f"{ev:%Y-%m-%d} " + "".join(f"{c:>26}" for c in cel))

    print("\n=== custo de cada patamar ===")
    for nome, al in niveis.items():
        eps = A.episodios(al)
        fp = [(a, b) for a, b in eps
              if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
        h = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
        duty = float(al.mean())
        print(f"  {nome:26s} {len(fp):3d} episodios de FP | "
              f"{len(fp)/cal_meses:5.2f}/mes calendario | {h/cal_meses:6.0f} h/mes | "
              f"ligado {100*duty:.0f}% do tempo")

    print("\n=== recall EFETIVA em funcao do tempo minimo para agir ===")
    print(f"{'precisa de':>12} " + "".join(f"{k:>26}" for k in niveis))
    for lim in [0.5, 1, 2, 3, 4, 6, 8, 12, 24]:
        cel = []
        for nome in niveis:
            v = np.array(leads[nome], dtype=float)
            k = int(np.nansum(v >= lim))
            cel.append(f"  {k}/{len(falhas)}  ({100*k/len(falhas):.0f}%)")
        print(f"{lim:9.1f} h " + "".join(f"{c:>26}" for c in cel))

    R = pd.DataFrame({"evento": falhas, **{k: leads[k] for k in niveis}})
    R.to_csv("acionavel.csv", index=False)


main()
