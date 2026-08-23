#!/usr/bin/env python3
"""Uniao do detector de 4 sinais (sem teto) com o stack do EXP7 -- os erros sao disjuntos.

diego_stack_valida.py so testou a uniao com a versao COM teto de 12 h, que perde
2024-01-16 e 2025-03-17. A versao SEM teto perde apenas 2024-01-16; o stack perde
apenas 2025-03-17. Os conjuntos de erro sao disjuntos, entao a uniao deveria dar 9/9 --
o primeiro 9/9 legitimo da investigacao, se sobreviver ao preco e ao nulo.

Tres coisas decidem se vale:
  - o preco em horas e em EPISODIOS (o stack sozinho ja custa 329 episodios/17,3 por mes;
    somar isso a 88 do detector daqui pode entregar um alarme que ninguem atende);
  - o teste de permutacao (a uniao tem cobertura maior, entao acerta mais por acaso);
  - LOEO, com o ponto do stack reescolhido fora do evento em teste.

Tambem se aplica o teto de 12 h SO no braco de 4 sinais (o stack ja produz episodios
curtos, truncar nao faz sentido nele).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
from diego_stack import alerta_de
from diego_stack_valida import seleciona


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    tr = pd.Series(idx < CORTE, index=idx)
    ev_tr = falhas[falhas < CORTE]

    print("montando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    nosso = alerta_2k(out, mask, K_BASE, K_VIB)

    linhas = []
    for tag in ["diego_iforest_estatico", "nosso_iforest_estatico"]:
        s = pd.read_parquet(f"escore_{tag}.parquet")["escore"].reindex(idx)
        mj = mask & s.notna()
        (_, _), p, sm, lim = seleciona(s, mj, tr, ev_tr, mj & tr)
        stack = alerta_de(s, mj, lim, sm)

        combos = [
            ("4 sinais (sem teto)", nosso),
            ("4 sinais + teto 12h", trunca(nosso, 12)),
            (f"stack {tag}", stack),
            ("UNIAO: 4 sinais + stack", (nosso.fillna(False) | stack.fillna(False)) & mask),
            ("UNIAO: 4 sinais(teto 12h) + stack",
             (trunca(nosso, 12).fillna(False) | stack.fillna(False)) & mask),
        ]
        print(f"\n{'='*82}\n{tag}  (percentil={p}, sustentacao={sm} min)")
        print(f"{'configuracao':36s} {'det':>4} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} "
              f"{'lead':>6} {'p':>8}  perdidos")
        for nome, al in combos:
            x = A.avalia(al, falhas, mask)
            x.update(A.permuta(al, mask, x["det"], len(falhas)))
            perd = sorted(set(t.strftime("%Y-%m-%d") for t in falhas) - set(x["detectados"]))
            print(f"{nome:36s} {x['det']:>2}/9 {x['episodios']:5d} {x['fp_mes']:7.2f} "
                  f"{x['h_fp_mes']:7.1f} {x['lead_med']:6.1f} {x['p']:8.4f}  {','.join(perd) or '-'}")
            linhas.append(dict(stack=tag, config=nome, det=x["det"], eps=x["episodios"],
                               fp_mes=x["fp_mes"], h_mes=x["h_fp_mes"], lead=x["lead_med"],
                               p=x["p"], perdidos=",".join(perd)))

        # LOEO da uniao: ponto do stack reescolhido sem o evento em teste
        pega = []
        for ev in falhas:
            outros = falhas[falhas != ev]
            sel = seleciona(s, mj, tr, outros[outros < CORTE], mj & tr)
            if sel is None:
                pega.append(False); continue
            (_, _), p2, sm2, lim2 = sel
            u = (nosso.fillna(False) | alerta_de(s, mj, lim2, sm2).fillna(False)) & mask
            pega.append(A.avalia(u, pd.Series([ev]), mask)["det"] == 1)
        print(f"  LOEO da uniao (4 sinais sem teto + stack): {sum(pega)}/9  "
              f"[perde: {','.join(t.strftime('%Y-%m-%d') for t, k in zip(falhas, pega) if not k) or 'nenhum'}]")

    pd.DataFrame(linhas).to_csv("ensemble.csv", index=False)


main()
