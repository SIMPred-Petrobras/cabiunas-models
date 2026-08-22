#!/usr/bin/env python3
"""Ablacao a FALSO POSITIVO IGUALADO.

A primeira rodada comparou os bracos nos limiares nativos do detector e nao
significa nada: acrescentar um sinal a um voto '2 ou mais de N' com limiar fixo
so deixa o detector mais sensivel, entao ele detecta mais E alarma mais. Os
numeros sobem juntos e nenhuma comparacao e possivel.

Aqui cada braco ganha um unico botao de sensibilidade -- todos os limiares
multiplicados por k -- e a curva inteira e reportada. A comparacao acontece na
mesma coluna de falso positivo, e o ponto de operacao de cada braco e escolhido
NO TREINO pelo orcamento de FP, nunca olhando o teste.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE

BRACOS = ["base", "max", "vib_mensal", "vib_rol", "max+vib_rol"]
KS = [0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5, 7.5, 10.0]
ORCAMENTO = 1.0     # FP/mes no treino -- o teto do relatorio do Francisco


def alerta_k(out, mask, braco, k):
    idx = out.index
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    n = (DET._sustained(ew("t", "1h"), DET.THR_FAM * k).astype(int)
         + DET._sustained(ew("p", "1h"), DET.THR_FAM * k).astype(int)
         + DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * k).astype(int))
    if "vib" in braco:
        n = n + DET._sustained(ew("vb", "30min"), 3.0 * k).astype(int)
    return (n >= 2) & mask


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    tr = pd.Series(idx < CORTE, index=idx); te = ~tr
    ev_tr, ev_te = falhas[falhas < CORTE], falhas[falhas >= CORTE]
    mask = mascara_pontuacao(df)

    linhas = []
    for braco in BRACOS:
        out = roda(braco, df, falhas)
        for k in KS:
            al = alerta_k(out, mask, braco, k)
            d = {"braco": braco, "k": k}
            for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
                am, qm = al[m], (mask & m)[m]
                x = A.avalia(am, ev, qm); x.update(A.permuta(am, qm, x["det"], len(ev)))
                d.update({f"{tag}_n": x["det"], f"{tag}_det": f"{x['det']}/{x['n_ev']}",
                          f"{tag}_lead": x["lead_med"], f"{tag}_fp": x["fp_mes"],
                          f"{tag}_h": x["h_fp_mes"], f"{tag}_p": x["p"],
                          f"{tag}_quais": ",".join(x["detectados"])})
            linhas.append(d)
            print(f"{braco:12s} k={k:<5} tr {d['tr_det']} FP={d['tr_fp']:6.2f} {d['tr_h']:6.1f}h "
                  f"p={d['tr_p']:.3f} | te {d['te_det']} FP={d['te_fp']:6.2f} {d['te_h']:6.1f}h "
                  f"p={d['te_p']:.3f}", flush=True)
        pd.DataFrame(linhas).to_csv("ablacao2.csv", index=False)

    t = pd.DataFrame(linhas)
    print("\n\n=== ponto de cada braco escolhido NO TREINO: menor k com FP treino <= "
          f"{ORCAMENTO}/mes (empate -> maior deteccao)")
    print(f"{'braco':13s} {'k':>5} | {'treino':>7} {'FP':>6} {'h/mes':>6} {'p':>6} "
          f"| {'TESTE':>7} {'FP':>6} {'h/mes':>6} {'p':>6}  eventos de teste")
    for b in BRACOS:
        s = t[(t.braco == b) & (t.tr_fp <= ORCAMENTO)]
        if s.empty:
            print(f"{b:13s}   -- nenhum k respeita o orcamento")
            continue
        r = s.sort_values(["tr_n", "k"], ascending=[False, True]).iloc[0]
        print(f"{b:13s} {r.k:>5} | {r.tr_det:>7} {r.tr_fp:6.2f} {r.tr_h:6.1f} {r.tr_p:6.3f} "
              f"| {r.te_det:>7} {r.te_fp:6.2f} {r.te_h:6.1f} {r.te_p:6.3f}  {r.te_quais}")


main()
