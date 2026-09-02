#!/usr/bin/env python3
"""Fecha o ponto de operacao combinando tudo que se mostrou gratis, com alvo FP < 1/mes.

Ingredientes ja medidos isoladamente:
  PISO      piso absoluto de 1,6 degC no MAD do spread (piso_fisico.csv): a episodios
            igualados, MESMOS eventos, -17% de horas, +3,4 h de lead.
  REFRAT    periodo refratario apos cada episodio (reduz_fp.csv): R=24h da 8/9 com metade
            do custo, lead identico, p melhor (0,0003 -> 0,0001), LOEO 6/9 -> 7/9.
  TETO      teto de 12 h por episodio: -72% de horas por uma deteccao.
  DUR_MIN   descarta episodio curto: -12% de episodios de graca.
  k         a curva classica -- reduz FP custando deteccao. So entra se o resto nao bastar.

ALVO: FP < 1,00 por mes de operacao. Hoje o melhor ponto validado esta em 3,12.

Nota sobre o que "FP<1" significa em cada mecanismo -- os tres nao sao equivalentes:
  - subir k reduz FP porque o detector VE MENOS. Perde deteccao.
  - o refratario reduz FP porque REPORTA MENOS VEZES o que ja viu. Nao perde o primeiro
    alerta (lead perdido medido: 0,00 h nos nove). Mas se uma degradacao NOVA comecar
    dentro da janela refrataria, ela fica muda -- risco que n=9 nao consegue medir.
  - o piso reduz custo porque corrige um denominador que colapsou. Nao perde nada.
Reportamos os tres separados para que a escolha final seja informada, nao so barata.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, mascara_pontuacao
from portoes import K_VIB
from auto_reset import trunca
import piso_fisico as PF
import reduz_fp as RF

PISOS = [0.0, 1.6]
KS = [1.7, 2.0, 2.3, 2.6, 3.0, 3.5]
RS = [0, 24, 48, 72, 120, 168, 240]
DS = [0, 60]
TETOS = [None, 12]
ALVO_FP = 1.00


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    d = PF.pre(df, falhas)

    L = []
    for piso in PISOS:
        out = PF.sinais(d, idx, piso, 0.0)
        E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
             for c, h in [("t", "1h"), ("p", "1h"), ("sp", "30min"), ("vb", "30min")]}
        for kb in KS:
            T = {"t": DET.THR_FAM*kb, "p": DET.THR_FAM*kb,
                 "sp": DET.THR_SPREAD*kb, "vb": 3.0*K_VIB}
            n = sum(DET._sustained(E[c], T[c]).astype(int) for c in E)
            al0 = (n >= 2) & mask
            for Rh in RS:
                alR = RF.refratario(al0, Rh)
                for D in DS:
                    alD = RF.dur_min(alR, D)
                    for teto in TETOS:
                        al = trunca(alD, teto) if teto else alD
                        x = A.avalia(al, falhas, mask)
                        L.append(dict(piso=piso, k=kb, R=Rh, D=D, teto=teto or 0,
                                      det=x["det"], eps=x["episodios"], fp=x["fp_mes"],
                                      h=x["h_fp_mes"], lead=x["lead_med"],
                                      quais=",".join(x["detectados"])))
        print(f"  piso={piso} varrido ({len(L)} configs)", flush=True)
    T = pd.DataFrame(L); T.to_csv("ponto_final.csv", index=False)

    print("\n" + "=" * 104)
    print("1) O PONTO FECHADO: refratario + piso, sem tocar em k")
    print("=" * 104)
    print(f"{'config':>44} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'duty':>7} {'lead':>6}")
    for piso, Rh, D, teto in [(0.0,0,0,None),(0.0,24,0,None),(1.6,24,0,None),(1.6,24,60,None),
                              (0.0,0,0,12),(0.0,24,0,12),(1.6,24,0,12),(1.6,24,60,12),
                              (1.6,48,60,12),(1.6,72,60,12)]:
        r = T[(T.piso==piso)&(T.k==1.7)&(T.R==Rh)&(T.D==D)&(T.teto==(teto or 0))].iloc[0]
        rot = (f"piso={piso} R={Rh}h D={D}min" + (f" teto={teto}h" if teto else ""))
        print(f"{rot:>44} {int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{100*r.h/730:6.2f}% {r.lead:6.1f}")

    print("\n" + "=" * 104)
    print(f"2) A FRONTEIRA EM FP < {ALVO_FP:.2f}/mes -- melhor deteccao para cada custo")
    print("=" * 104)
    S = T[T.fp < ALVO_FP].sort_values(["det", "lead"], ascending=[False, False])
    if not len(S):
        print("  NENHUMA configuracao da grade chega a FP < 1,00/mes.")
    else:
        print(f"  configuracoes abaixo de {ALVO_FP:.2f} FP/mes: {len(S)}   "
              f"melhor deteccao alcancada: {int(S.det.max())}/9\n")
        print(f"{'piso':>5} {'k':>5} {'R':>5} {'D':>5} {'teto':>5} {'det':>6} {'eps':>5} "
              f"{'FP/mes':>7} {'h/mes':>7} {'duty':>7} {'lead':>6}  eventos")
        for _, r in S.head(14).iterrows():
            print(f"{r.piso:5.1f} {r.k:5.2f} {int(r.R):4d}h {int(r.D):4d}m {int(r.teto):4d}h "
                  f"{int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
                  f"{100*r.h/730:6.2f}% {r.lead:6.1f}  {r.quais[:60]}")

    print("\n" + "=" * 104)
    print("3) O PRECO DE CADA PATAMAR DE FP (melhor deteccao possivel em cada faixa)")
    print("=" * 104)
    print(f"{'FP/mes <=':>10} {'melhor det':>11} {'como':>52} {'h/mes':>7} {'lead':>6}")
    for lim in [4.5, 3.5, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5]:
        S = T[T.fp <= lim]
        if not len(S): continue
        b = S.sort_values(["det", "h"], ascending=[False, True]).iloc[0]
        rot = f"piso={b.piso} k={b.k} R={int(b.R)}h D={int(b.D)}min teto={int(b.teto)}h"
        print(f"{lim:9.1f} {int(b.det):9d}/9 {rot:>52} {b.h:7.1f} {b.lead:6.1f}")


if __name__ == "__main__":
    main()
