#!/usr/bin/env python3
"""A mascara daqui melhora o detector dele, ou so o dessensibiliza?

melhora_diego.py mediu, sobre a reproducao do stack do EXP7 no alvo dele:
  mascara dele  (on + piso 150 degC)          96,9% hit, 26 preditivos, NAR 0,45%
  mascara nossa (quente T5>300 + blackout 6h) 100,0% hit, 29 preditivos, NAR 0,82%

O ganho parece grande -- 26 -> 29 preditivos e zero sem-deteccao, que sao exatamente os
numeros publicados do EXP10c. Mas o limiar e recalculado como percentil do escore DENTRO
de cada mascara, e trocar a mascara troca a distribuicao de referencia: com so dado
quente, o p99,0 cai em valor absoluto e o detector fica mais sensivel. Ou seja, os dois
pontos nao estao no mesmo custo, e comparar deteccao a custo diferente e a armadilha de
Pareto em que ja caimos tres vezes nesta investigacao.

Este script varre percentis mais baixos NA MASCARA DELE ate alcancar o NAR da mascara
nossa, e compara ali. Se a mascara dele alcancar 29 preditivos ao chegar em 0,82%, o
ganho e sensibilidade e nao mascara.

Reporta tambem HORAS ABSOLUTAS de alarme, porque NAR tem denominador diferente em cada
mascara (a nossa exclui as 6 h pos-partida e tudo abaixo de 300 degC), e fracao com
denominador movel nao e moeda comparavel.
"""
from __future__ import annotations
import sys, gc
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C
import avalia as A
from ablacao import canonico, mascara_pontuacao
from diego_stack import monta_features
from quadrante import alvos_diego, regua_diego, JAN_H
from melhora_diego import (SENSORES, OOS0, OOS1, mascara_dele, treino_dele,
                           escore_com_piso, alerta, nar)

PERC = [95.0, 96.0, 97.0, 97.5, 98.0, 98.5, 99.0, 99.3, 99.5]
DEB = 2


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    idx = df.index
    reais, under = alvos_diego()
    todos_ts = pd.concat([reais["t"], under["t"]])
    longe = pd.Series(True, index=idx)
    for t in todos_ts:
        longe &= ~((idx >= t - pd.Timedelta(hours=JAN_H)) & (idx <= t + pd.Timedelta(hours=JAN_H)))

    m_dele = mascara_dele(df, g)
    m_nossa = mascara_pontuacao(df)
    fit_idx = df.index[treino_dele(df, g, todos_ts)]
    oos = (idx >= OOS0) & (idx <= OOS1)
    print(f"horas pontuaveis no OOS -- mascara dele: {(m_dele & oos).sum()*2/60:.0f} h | "
          f"mascara nossa: {(m_nossa & oos).sum()*2/60:.0f} h", flush=True)

    print("montando features ...", flush=True)
    F = monta_features(g, SENSORES)
    del g; gc.collect()
    s = escore_com_piso(F, fit_idx, 0.0)
    del F; gc.collect()

    print(f"\n{'mascara':>8} {'perc':>7} | {'hit':>7} {'pred':>5} {'reat':>5} {'nada':>5} "
          f"{'lead':>7} | {'NAR':>7} {'h alarme OOS':>13}")
    linhas = []
    for rot, mk in [("dele", m_dele), ("nossa", m_nossa)]:
        ref = s[mk & (idx < OOS0)].dropna()
        for p in PERC:
            lim = float(np.percentile(ref, p))
            al = alerta(s, mk, lim, DEB)
            x = regua_diego(al, reais, mk)
            n = nar(al, mk, longe)
            h = (al.fillna(False) & mk & oos).sum() * 2 / 60
            print(f"{rot:>8} {p:7.2f} | {x['hit']:6.1f}% {x['pred']:5d} {x['reat']:5d} "
                  f"{x['nada']:5d} {x['lead_med']:6.1f}h | {n:6.2f}% {h:12.1f} h")
            linhas.append(dict(mascara=rot, percentil=p, nar=n, h_alarme=h,
                               **{k: v for k, v in x.items() if k != "sem_det"}))
    t = pd.DataFrame(linhas)
    t.to_csv("melhora_diego2.csv", index=False)

    print(f"\n=== COMPARACAO A CUSTO IGUALADO ===")
    for col, rot in [("nar", "NAR (%)"), ("h_alarme", "horas de alarme no OOS")]:
        print(f"\n  pareando por {rot}:")
        dn = t[t.mascara == "nossa"]
        dd = t[t.mascara == "dele"]
        for _, r in dn.iterrows():
            j = (dd[col] - r[col]).abs().idxmin()
            q = dd.loc[j]
            print(f"    nossa p{r.percentil:<5} {r.pred:2.0f} pred / {r.hit:5.1f}% a {r[col]:7.2f}"
                  f"   vs   dele p{q.percentil:<5} {q.pred:2.0f} pred / {q.hit:5.1f}% a {q[col]:7.2f}"
                  f"   -> {r.pred - q.pred:+.0f} preditivos")


main()
