#!/usr/bin/env python3
"""Escape por magnitude: um sinal sozinho, muito acima do limiar, confirma sem o voto.

De onde veio. A autopsia dos 4 casos que o stack do Diego pega e nos nao (autopsia4.py)
mostrou o MESMO ponto de falha nos quatro: um unico sinal sustentou acima do limiar --
em dois deles a 5,97x e a 22,02x -- e o alerta nao saiu porque o voto exige 2. O voto
trata 1,05x e 22x como equivalentes, o que joga informacao fora.

A regra:  alerta = (votos >= 2) OU (algum sinal sustentado acima de M x seu limiar)

E o mesmo mecanismo do GATE_ESCAPE_MULTIPLIER do CNN1D-AE do Diego, que nasceu de um caso
analogo (dois portoes bloqueando uma deteccao real).

CETICISMO OBRIGATORIO. Os 4 casos foram escolhidos por serem os que perdemos -- e a
regra foi desenhada olhando para eles. Vinte ideias ja morreram nesta investigacao quando
medidas a custo igualado. Entao:
  1. varredura de M com k_base reajustado para IGUALAR EPISODIOS (nao horas: horas e a
     moeda barata e ja nos enganou uma vez neste projeto);
  2. LOEO com (M, k) reescolhidos fora do evento em teste;
  3. diagnostico de ONDE o escape dispara -- a hipotese concorrente e que `p` estoura o
     limiar em transiente de partida, e as 4 janelas todas tem um ciclo liga/desliga
     dentro. Se o escape so acende perto de partida, e ruido de transiente, nao deteccao.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

MS = [np.inf, 10.0, 7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5]
KS = [1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.8, 3.2, 3.8, 4.5]
JAN = pd.Timedelta(hours=48)
CASOS = [pd.Timestamp(x, tz="UTC") for x in
         ["2025-11-05 02:42", "2025-11-24 22:49", "2025-11-30 20:50", "2026-01-23 21:20"]]


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    op = df["in_operation"].astype(bool)
    partidas = op & ~op.shift(fill_value=False)
    out = roda(BRACO, df, falhas)

    E = {c: out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
         for c, hl in [("t", "1h"), ("p", "1h"), ("sp", "30min"), ("vb", "30min")]}

    def thrs(kb, kv):
        return {"t": DET.THR_FAM * kb, "p": DET.THR_FAM * kb,
                "sp": DET.THR_SPREAD * kb, "vb": 3.0 * kv}

    def alerta(kb, kv, M):
        T = thrs(kb, kv)
        s = {c: DET._sustained(E[c], T[c]) for c in E}
        n = sum(s[c].astype(int) for c in E)
        base = (n >= 2)
        if np.isfinite(M):
            esc = np.logical_or.reduce([DET._sustained(E[c], T[c] * M).values for c in E])
            base = base | pd.Series(esc, index=idx)
        return base & mask

    linhas = []
    for M in MS:
        for kb in KS:
            al = alerta(kb, K_VIB, M)
            x = A.avalia(al, falhas, mask)
            xt = A.avalia(trunca(al, 12), falhas, mask)
            pega4 = sum(bool(al.loc[t - pd.Timedelta(hours=24): t + pd.Timedelta(hours=24)].any())
                        for t in CASOS)
            linhas.append(dict(M=M, k=kb, det=x["det"], eps=x["episodios"], fp=x["fp_mes"],
                               h=x["h_fp_mes"], lead=x["lead_med"],
                               t12_det=xt["det"], t12_h=xt["h_fp_mes"], pega4=pega4,
                               quais=",".join(x["detectados"])))
        print(f"  M={M} varrido", flush=True)
    T = pd.DataFrame(linhas); T.to_csv("escape.csv", index=False)

    base = T[(T.M == np.inf) & (T.k == K_BASE)].iloc[0]
    print("\n" + "=" * 96)
    print(f"A EPISODIOS IGUALADOS ({base.fp:.2f} FP/mes, o do ponto atual)")
    print("=" * 96)
    print(f"{'M':>6} {'k':>5} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} "
          f"{'+teto12h':>9} {'pega os 4':>10}")
    for M in MS:
        g = T[T.M == M].assign(d=(T[T.M == M].fp - base.fp).abs()).sort_values("d")
        r = g.iloc[0]
        print(f"{r.M:6.1f} {r.k:5.2f} {int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{r.lead:6.1f} {int(r.t12_det):7d}/9 {int(r.pega4):8d}/4")

    print("\n" + "=" * 96)
    print("ONDE O ESCAPE ACENDE: e transiente de partida?")
    print("=" * 96)
    for M in [5.0, 3.0, 2.0]:
        al_s = alerta(K_BASE, K_VIB, M)
        al_n = alerta(K_BASE, K_VIB, np.inf)
        so_esc = al_s & ~al_n
        eps = A.episodios(so_esc)
        if not eps:
            print(f"  M={M}: nenhum episodio novo"); continue
        tp = [p for p in partidas[partidas].index]
        dt = [min((abs((a - x).total_seconds()) for x in tp), default=np.inf) / 3600 for a, b in eps]
        dt = np.array(dt)
        print(f"  M={M:4.1f}: {len(eps):3d} episodios so-escape   "
              f"mediana da distancia a partida mais proxima: {np.median(dt):6.1f} h   "
              f"dentro de 12 h de uma partida: {(dt <= 12).sum()}/{len(eps)} "
              f"({100*(dt<=12).mean():.0f}%)")


if __name__ == "__main__":
    main()
