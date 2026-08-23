#!/usr/bin/env python3
"""A deriva do custo e do detector ou da operacao da maquina?

futuro.py achou o que mais importa para a pergunta 'vai continuar valendo': com o teto
de 12 h, a duracao media do episodio de falso positivo cresce de forma perfeitamente
monotona ao longo dos 5 semestres (4,9 -> 7,8 h, rho=+1,00), e as horas de alarme por
mes de operacao vao de 12 a 59. Nada disso foi causado por mexer no detector -- os
parametros sao fixos.

Duas explicacoes concorrentes, com consequencias opostas:

  (a) o DETECTOR esta derivando -- a referencia rolante de 400 h nao acompanha mais a
      maquina, ou os limiares fixos (THR_FAM, k) ficaram apertados para o novo normal.
      Consequencia: precisa recalibrar, e vai precisar de novo.
  (b) a OPERACAO mudou -- campanhas mais curtas geram mais religamentos, e ja medimos
      que 48% dos falsos positivos comecam nas primeiras 30 h apos religamento
      (fp_rajadas.py). Mais partidas por mes => mais FP, sem nada de errado no detector.
      Consequencia: o numero de FP e uma funcao do regime de operacao, e reportar
      FP/mes sem normalizar por partidas e enganoso.

Distinguir e barato: contar partidas por mes de operacao e a duracao das campanhas por
semestre, e ver se acompanham a curva de FP.

O p do Spearman com n=5 tambem e recalculado por permutacao exata -- a aproximacao
assintotica do scipy reporta p=0,000 para rho=1,0, o que nao existe com 5 pontos
(o minimo possivel e 2/120 = 0,0167).
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd
from scipy import stats

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca


def p_exato(y):
    """p bilateral por permutacao exata do Spearman contra a ordem temporal."""
    n = len(y); r0 = stats.spearmanr(np.arange(n), y).statistic
    todos = [abs(stats.spearmanr(np.arange(n), [y[i] for i in perm]).statistic)
             for perm in itertools.permutations(range(n))]
    return float(np.mean(np.array(todos) >= abs(r0) - 1e-12)), r0


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    op = df["in_operation"].astype(bool)
    partidas = op & ~op.shift(fill_value=False)

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    teto = trunca(alerta_2k(out, mask, K_BASE, K_VIB), 12)
    jan = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    print("\n=== regime de operacao x custo do detector, por semestre ===")
    print(f"{'semestre':>10} {'h oper':>8} {'partidas':>9} {'part/mes':>9} "
          f"{'campanha med':>13} | {'FP/mes':>7} {'h alarme/mes':>13} {'h/episodio':>11}")
    L = []
    for _, gser in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS")):
        if len(gser) == 0:
            continue
        sel = (idx >= gser.index[0]) & (idx <= gser.index[-1])
        ho = (mask & sel).sum() * 2 / 60
        if ho < 300:
            continue
        np_ = int((partidas & sel).sum())
        h_op = (op & sel).sum() * 2 / 60
        camp = h_op / max(np_, 1)
        eps = A.episodios(teto & sel)
        fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jan)]
        hfp = sum((b - a).total_seconds() / 3600 + 2 / 60 for a, b in fp)
        mes = ho / 730
        print(f"{str(gser.index[0].date()):>10} {ho:8.0f} {np_:9d} {np_/mes:9.2f} "
              f"{camp:11.1f} h | {len(fp)/mes:7.2f} {hfp/mes:13.1f} {hfp/max(len(fp),1):11.1f}")
        L.append(dict(sem=gser.index[0].date(), h_op=ho, partidas=np_, part_mes=np_/mes,
                      campanha_h=camp, fp_mes=len(fp)/mes, h_mes=hfp/mes,
                      h_ep=hfp/max(len(fp), 1)))
    t = pd.DataFrame(L); t.to_csv("deriva.csv", index=False)

    print("\n=== tendencia temporal (Spearman, p por permutacao exata, n=5) ===")
    for col, rot in [("part_mes", "partidas por mes de operacao"),
                     ("campanha_h", "duracao media da campanha"),
                     ("fp_mes", "FP por mes de operacao"),
                     ("h_mes", "horas de alarme por mes"),
                     ("h_ep", "horas por episodio de FP")]:
        p, r = p_exato(list(t[col]))
        print(f"  {rot:32s} rho={r:+.2f}  p={p:.4f}")

    print("\n=== correlacao entre regime de operacao e custo (Spearman) ===")
    for a, b in [("part_mes", "fp_mes"), ("part_mes", "h_mes"),
                 ("campanha_h", "fp_mes"), ("campanha_h", "h_mes")]:
        r = stats.spearmanr(t[a], t[b])
        print(f"  {a:12s} x {b:8s}: rho={r.statistic:+.2f}  p={r.pvalue:.3f}")

    print("\n=== duty cycle: fracao do tempo pontuavel com alarme ativo ===")
    for _, r in t.iterrows():
        print(f"  {r['sem']}: {100*r.h_mes/730:5.2f}% do tempo de operacao com alarme ativo")


main()
