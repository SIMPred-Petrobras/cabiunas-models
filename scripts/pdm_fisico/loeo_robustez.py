#!/usr/bin/env python3
"""O 7/8 depende da regra de desempate escolhida -- ja vimos isso na pratica
com 2025-11-04. Aqui isso vira numero: mesma grade (k_base,k_vib) por fold,
4 regras de selecao diferentes e igualmente defensaveis a priori, aplicadas
DEPOIS de calcular a grade inteira (ninguem espia o evento de fora em nenhuma
delas -- a diferenca e so como cada uma escolhe o ponto dentro do treino).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO

KS = [0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5, 7.5, 10.0]
ALVO = 3.400823


def regra_A_teto15_maxdet(s):   # a que usamos ate agora
    c = s[s.tr_fp <= 1.5 * ALVO]
    if c.empty:
        c = s.copy(); c["d"] = (c.tr_fp - ALVO).abs(); return c.sort_values("d").iloc[0]
    return c.sort_values(["tr_n", "tr_fp"], ascending=[False, True]).iloc[0]

def regra_B_fp_mais_proximo(s):   # regra original da ablacao3, sem teto
    c = s.copy(); c["d"] = (c.tr_fp - ALVO).abs()
    return c.sort_values(["d", "tr_n"], ascending=[True, False]).iloc[0]

def regra_C_max_deteccao_sem_teto(s):   # so maximiza deteccao de treino, ignora custo
    return s.sort_values(["tr_n", "tr_fp"], ascending=[False, True]).iloc[0]

def regra_D_teto_nativo_estrito(s):   # teto = 1.0x o nativo, sem folga
    c = s[s.tr_fp <= ALVO]
    if c.empty:
        c = s.copy(); c["d"] = (c.tr_fp - ALVO).abs(); return c.sort_values("d").iloc[0]
    return c.sort_values(["tr_n", "tr_fp"], ascending=[False, True]).iloc[0]

REGRAS = {
    "A: teto 1.5x, max deteccao": regra_A_teto15_maxdet,
    "B: FP mais proximo do nativo": regra_B_fp_mais_proximo,
    "C: max deteccao, sem teto de FP": regra_C_max_deteccao_sem_teto,
    "D: teto 1.0x nativo (estrito)": regra_D_teto_nativo_estrito,
}


def main():
    df = canonico()
    falhas_todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    falhas = falhas_todas[falhas_todas >= "2025-01-01"].reset_index(drop=True)
    idx = df.index
    mask = mascara_pontuacao(df)

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas_todas)

    grades = {}
    for ev in falhas:
        excl = (idx >= ev - pd.Timedelta(hours=48)) & (idx < ev + pd.Timedelta(hours=2))
        train_m = mask & ~excl
        ev_train = falhas[falhas != ev]
        grid = []
        for kb in KS:
            for kv in KS:
                al = alerta_2k(out, train_m, kb, kv)
                x = A.avalia(al[train_m], ev_train, train_m[train_m])
                grid.append(dict(k_base=kb, k_vib=kv, tr_n=x["det"], tr_fp=x["fp_mes"]))
        grades[ev] = pd.DataFrame(grid)
        print(f"  grade pronta: {ev.strftime('%Y-%m-%d')}", flush=True)

    resultado = {}
    detalhe = []
    for nome_regra, fn in REGRAS.items():
        acertos = 0
        linha_regra = []
        for ev in falhas:
            r = fn(grades[ev])
            kb, kv = r.k_base, r.k_vib
            hold = (idx >= ev - pd.Timedelta(days=7)) & (idx < ev + pd.Timedelta(days=1))
            al_full = alerta_2k(out, mask, kb, kv)
            xh = A.avalia(al_full[hold], pd.Series([ev]), mask[hold])
            ok = xh["det"] >= 1
            acertos += int(ok)
            linha_regra.append(dict(regra=nome_regra, evento=ev, k_base=kb, k_vib=kv,
                                     detectado=ok))
        resultado[nome_regra] = acertos
        detalhe.extend(linha_regra)
        print(f"{nome_regra:32s} -> {acertos}/{len(falhas)}", flush=True)

    D = pd.DataFrame(detalhe)
    D.to_csv("loeo_robustez.csv", index=False)
    vals = list(resultado.values())
    print(f"\nfaixa de resultados sob as 4 regras: {min(vals)}/{len(falhas)} a {max(vals)}/{len(falhas)}")
    print("\nquem discorda entre regras (eventos que mudam de detectado/perdido):")
    piv = D.pivot(index="evento", columns="regra", values="detectado")
    incons = piv[piv.nunique(axis=1) > 1]
    print(incons.to_string() if not incons.empty else "  nenhum -- todas as regras concordam em todo evento")


main()
