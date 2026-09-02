#!/usr/bin/env python3
"""O spread do mancal (`sp`) paga a deriva inteira e contribui pouco? Ablacao.

A pista veio de modos.py: `sp` sustentou em apenas 4 dos 9 eventos -- o mais fraco dos
quatro sinais (vb 8/9, p 7/9, t 5/9). E e exatamente o sinal cujo denominador colapsa:
o MAD do spread cai de 1,22 para 0,32 degC entre 2024-H1 e 2026-H1 enquanto o spread
FISICO em graus tambem cai (12,2 -> 9,7 degC). Foi esse colapso que diagnosticamos como
origem da deriva do custo (rho=+0,351, p=0,0074 sobre 57 campanhas), e para o qual nenhum
piso de escala funcionou.

Se `sp` for o portador da deriva e contribuir pouco para a deteccao, tira-lo resolve de
uma vez o unico problema estrutural que sobrou aberto.

Bracos:
  A  t,p,sp,vb   voto>=2   o detector como esta
  B  t,p,vb      voto>=2   sem o spread
  C  t,p,sp,vb   voto>=2   com limiar proprio de `sp` endurecido (k_sp = m x k_base)

Protocolo: k_base varrido para IGUALAR EPISODIOS (a moeda do operador), e para cada braco
mede-se deteccao, custo, lead, o p de permutacao E A TENDENCIA DO DUTY POR CAMPANHA
(n=57, a unica agregacao com poder -- por semestre n=5 e nao resolve nada). Tudo com o
ponto de operacao novo (refratario 48 h + duracao minima 60 min).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from portoes import K_BASE, K_VIB
import reduz_fp as RF

HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
KS = [1.2, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0]
R_REFRAT, D_MIN = 48, 60


def campanhas(df, mask, idx):
    op = df["in_operation"].astype(bool)
    gid = (op & ~op.shift(fill_value=False)).cumsum().where(op)
    out = []
    for _, sub in df[op].groupby(gid[op]):
        a, b = sub.index[0], sub.index[-1]
        h = (mask & (idx >= a) & (idx <= b)).sum() * 2 / 60
        if h >= 24:
            out.append((a, b, h))
    return out


def main():
    df = canonico(); idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)
    out = roda("max+vib_rol", df, falhas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask) for c, h in HL.items()}
    camps = campanhas(df, mask, idx)
    jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]
    meses_op = mask.sum() * 2 / 60 / 730.0
    print(f"campanhas com >=24 h pontuaveis: {len(camps)}\n", flush=True)

    def duty_camp(al):
        y = []
        for a, b, h in camps:
            sel = (idx >= a) & (idx <= b)
            eps = A.episodios(al & sel)
            fp = [(x, z) for x, z in eps if not any(x <= t1 and z >= t0 for t0, t1 in jw)]
            y.append(100 * sum((z - x).total_seconds()/3600 + 2/60 for x, z in fp) / h)
        return np.array(y)

    def constroi(kb, sinais, m_sp=1.0):
        T = {"t": DET.THR_FAM*kb, "p": DET.THR_FAM*kb,
             "sp": DET.THR_SPREAD*kb*m_sp, "vb": 3.0*K_VIB}
        n = sum(DET._sustained(E[c], T[c]).astype(int) for c in sinais)
        al = (n >= 2) & mask
        return RF.dur_min(RF.refratario(al, R_REFRAT), D_MIN)

    BRACOS = [("A  4 sinais (atual)", ["t","p","sp","vb"], 1.0),
              ("B  sem o spread",     ["t","p","vb"],      1.0),
              ("C  sp com limiar 1,5x", ["t","p","sp","vb"], 1.5),
              ("C  sp com limiar 2,0x", ["t","p","sp","vb"], 2.0),
              ("C  sp com limiar 3,0x", ["t","p","sp","vb"], 3.0)]

    L = []
    for rot, sinais, m in BRACOS:
        for kb in KS:
            al = constroi(kb, sinais, m)
            x = A.avalia(al, falhas, mask)
            y = duty_camp(al)
            r = stats.spearmanr(np.arange(len(y)), y)
            mid = len(y)//2
            L.append(dict(braco=rot, k=kb, det=x["det"], eps=x["episodios"], fp=x["fp_mes"],
                          h=x["h_fp_mes"], lead=x["lead_med"], rho=r.statistic, p_rho=r.pvalue,
                          duty1=y[:mid].mean(), duty2=y[mid:].mean(),
                          quais=",".join(x["detectados"])))
        print(f"  {rot} varrido", flush=True)
    T = pd.DataFrame(L); T.to_csv("ablacao_sp.csv", index=False)

    ref = T[(T.braco.str.startswith("A")) & (T.k == K_BASE)].iloc[0]
    print("\n" + "=" * 108)
    print(f"A EPISODIOS IGUALADOS ({ref.fp:.2f} FP/mes) -- e a deriva por campanha (n={len(camps)})")
    print("=" * 108)
    print(f"{'braco':22s} {'k':>5} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} "
          f"{'p':>8} | {'rho deriva':>10} {'p_rho':>8} {'duty 1a->2a metade':>20}")
    for rot, _, _ in BRACOS:
        g = T[T.braco == rot].assign(d=(T[T.braco == rot].fp - ref.fp).abs()).sort_values("d")
        r = g.iloc[0]
        al = constroi(r.k, dict(BRACOS_ := {b[0]: b[1] for b in BRACOS})[rot],
                      {b[0]: b[2] for b in BRACOS}[rot])
        x = A.avalia(al, falhas, mask); perm = A.permuta(al, mask, x["det"], len(falhas))["p"]
        print(f"{rot:22s} {r.k:5.2f} {int(r.det):4d}/9 {int(r.eps):5d} {r.fp:7.2f} {r.h:7.1f} "
              f"{r.lead:6.1f} {perm:8.4f} | {r.rho:+10.3f} {r.p_rho:8.4f} "
              f"{r.duty1:7.2f}% ->{r.duty2:7.2f}%")

    print("\n" + "=" * 108)
    print("quais eventos cada braco pega, no ponto de custo igualado")
    print("=" * 108)
    base_q = set(ref.quais.split(","))
    for rot, _, _ in BRACOS:
        g = T[T.braco == rot].assign(d=(T[T.braco == rot].fp - ref.fp).abs()).sort_values("d")
        q = set(str(g.iloc[0].quais).split(","))
        print(f"  {rot:22s} {int(g.iloc[0].det)}/9   perdeu={sorted(base_q-q)}  ganhou={sorted(q-base_q)}")


if __name__ == "__main__":
    main()
