#!/usr/bin/env python3
"""O que causa as rajadas de FP (set/2024, mar/2025, jan/2026, cada uma com
9-10 episodios no mesmo mes)? Duas hipoteses testaveis com o que ja temos:
  1. reinicios em sequencia -- cada partida gera um episodio curto proprio
     (a rampa ja mostrou que o pos-partida e uma zona de risco).
  2. um unico sinal dominando (ex.: vibracao sozinha oscilando) em vez dos
     4 sinais contribuindo de forma distribuida.
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

K_BASE, K_VIB = 1.3, 5.5
MESES = ["2024-09", "2025-03", "2026-01"]


def horas_desde_partida(df: pd.DataFrame) -> pd.Series:
    op = df["in_operation"].astype(bool)
    starts = op & ~op.shift(fill_value=False)
    tempo = pd.Series(df.index, index=df.index)
    marca = tempo.where(starts).ffill()
    h = (tempo - marca).dt.total_seconds() / 3600.0
    return h.fillna(1e6)


def conta_por_sinal(out, mask, k_base, k_vib):
    idx = out.index
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    return {
        "t": DET._sustained(ew("t", "1h"), DET.THR_FAM * k_base),
        "p": DET._sustained(ew("p", "1h"), DET.THR_FAM * k_base),
        "sp": DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * k_base),
        "vb": DET._sustained(ew("vb", "30min"), 3.0 * k_vib),
    }


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    h_partida = horas_desde_partida(df)

    out = roda(BRACO, df, falhas)
    sinais = conta_por_sinal(out, mask, K_BASE, K_VIB)
    n = sum(s.astype(int) for s in sinais.values())
    alerta = (n >= 2) & mask
    eps = A.episodios(alerta)

    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]
    fp_eps = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]

    for mes in MESES:
        p = pd.Period(mes, freq="M")
        do_mes = [(a, b) for a, b in fp_eps if pd.Period(a, freq="M") == p]
        print(f"\n=== {mes}: {len(do_mes)} episodios de FP ===")
        for a, b in do_mes:
            dur_h = (b - a).total_seconds() / 3600 + 2 / 60
            h_desde = h_partida.asof(a)
            quais = [c for c, s in sinais.items() if s.loc[a:b].any()]
            print(f"  {a.strftime('%Y-%m-%d %H:%M')} -> {b.strftime('%m-%d %H:%M')}  "
                  f"({dur_h:5.1f}h)  h_desde_partida={h_desde:6.1f}h  sinais=[{','.join(quais)}]")

    print("\n--- resumo: quantos episodios (de TODOS os FP, nao so os 3 meses) "
          "comecam com <30h desde a ultima partida? ---")
    inicios = pd.DatetimeIndex([a for a, _ in fp_eps])
    h_ini = pd.Series([h_partida.asof(t) for t in inicios])
    print(f"{(h_ini < 30).sum()} de {len(fp_eps)} ({(h_ini<30).mean():.0%})")

    print("\n--- resumo: distribuicao de qual(is) sinal(is) aparece em cada episodio de FP ---")
    contagem_sinal = {}
    for a, b in fp_eps:
        quais = tuple(sorted(c for c, s in sinais.items() if s.loc[a:b].any()))
        contagem_sinal[quais] = contagem_sinal.get(quais, 0) + 1
    for k, v in sorted(contagem_sinal.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")


main()
