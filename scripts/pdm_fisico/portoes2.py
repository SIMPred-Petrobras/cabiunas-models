#!/usr/bin/env python3
"""Aprofunda os dois portoes: por que a rampa e nula, e ate onde a volatilidade vai.

portoes.py mostrou rampa nula em 40..500 degC/h e volatilidade dando -5,9% de horas
de alarme a 0,15 um. Faltam tres coisas:

  1. POR QUE a rampa e nula. Hipotese: a mascara de pontuacao daqui (operacao quente
     T5>300 + blackout de 6 h pos-partida) ja remove o que o portao de rampa do Diego
     remove. La a mascara e so state=="on", entao manobra de carga continua sendo
     pontuada. Se for isso, o portao nao tem o que suprimir -- e um resultado sobre a
     diferenca entre as duas pipelines, nao sobre o mecanismo.
  2. a curva completa do portao de volatilidade abaixo de 0,15.
  3. o efeito no NUMERO de episodios, nao so nas horas. Supressao fragmenta episodio;
     ja medimos que contagem de episodio e manipulavel por suavizacao. Sem as duas
     colunas juntas o ganho pode ser contabil.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import proxy_rampa, indice_volatilidade, K_BASE, K_VIB

VOL = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.30, 0.39, 0.60, 1.00]


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    op = df["in_operation"].astype(bool)

    rate = proxy_rampa(g).reindex(idx)
    vol = indice_volatilidade(g).reindex(idx)

    # ---- 1. por que a rampa e nula: onde estao as rampas de verdade
    print("=== onde vivem as rampas de |dT5/dt| > 100 degC/h ===")
    alta = (rate > 100).fillna(False)
    print(f"total de amostras com rampa alta          : {alta.sum():6d}  ({alta.sum()*2/60:7.1f} h)")
    print(f"  ... com a maquina em operacao (op)      : {(alta&op).sum():6d}  ({(alta&op).sum()*2/60:7.1f} h)")
    print(f"  ... dentro da mascara de pontuacao daqui: {(alta&mask).sum():6d}  ({(alta&mask).sum()*2/60:7.1f} h)")
    print(f"a mascara daqui ja descarta {100*(1-(alta&mask).sum()/max((alta&op).sum(),1)):.1f}% das rampas altas\n")
    for lim in (40, 60, 100, 150):
        a = (rate > lim).fillna(False)
        print(f"  |dT5/dt|>{lim:>3} degC/h: {(a&op).sum()*2/60:7.1f} h operando -> "
              f"{(a&mask).sum()*2/60:6.1f} h pontuaveis  ({100*(a&mask).mean():.3f}% da serie)")

    print("\nmontando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)

    def linha(al, nome):
        x = A.avalia(al, falhas, mask)
        x.update(A.permuta(al, mask, x["det"], len(falhas)))
        return dict(portao=nome, det=x["det"], eps=x["episodios"], fp=x["fp"],
                    fp_mes=x["fp_mes"], h_mes=x["h_fp_mes"], lead=x["lead_med"],
                    duty=100*x["duty"], p=x["p"], quais=",".join(x["detectados"]))

    print("\n=== PORTAO DE VOLATILIDADE: curva completa (serie inteira, 9 eventos) ===")
    print(f"{'limiar':>7} {'supr%':>6} {'det':>5} {'episod':>7} {'FP':>5} {'FP/mes':>7} "
          f"{'h/mes':>7} {'duty%':>6} {'lead':>6} {'p':>7}")
    linhas = [linha(base, "nenhum")]
    r = linhas[0]
    print(f"{'-':>7} {0.0:6.1f} {r['det']:5d} {r['eps']:7d} {r['fp']:5d} {r['fp_mes']:7.2f} "
          f"{r['h_mes']:7.1f} {r['duty']:6.2f} {r['lead']:6.1f} {r['p']:7.4f}")
    for v in VOL:
        blq = (vol > v).fillna(False)
        al = base & ~blq
        d = linha(al, f"vol>{v}")
        d["supr"] = 100 * (blq & mask).sum() / mask.sum()
        linhas.append(d)
        print(f"{v:7.2f} {d['supr']:6.1f} {d['det']:5d} {d['eps']:7d} {d['fp']:5d} {d['fp_mes']:7.2f} "
              f"{d['h_mes']:7.1f} {d['duty']:6.2f} {d['lead']:6.1f} {d['p']:7.4f}")

    pd.DataFrame(linhas).to_csv("portoes2.csv", index=False)
    print("\nleitura: 'supr%' e quanto do tempo pontuavel o portao bloqueia. Um portao "
          "util derruba h/mes MUITO mais rapido do que supr% -- se os dois andam juntos, "
          "ele so esta cortando tempo no atacado, nao mirando falso positivo.")


main()
