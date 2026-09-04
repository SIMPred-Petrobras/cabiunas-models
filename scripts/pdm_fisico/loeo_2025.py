#!/usr/bin/env python3
"""Mesmo leave-one-evento-out, mas so nos 8 eventos de 2025-2026 -- 2024-01-16
sai da lista de eventos avaliados (e nunca entra em ev_train de ninguem),
porque e estruturalmente indetectavel (159h de historico total antes dele,
contra as 400h que a referencia precisa) e nao e questao de limiar.

A referencia dos sinais (roda()) continua recebendo a lista COMPLETA de 9
falhas -- e so higiene de dado (nulificar a janela ao redor de qualquer falha
conhecida na referencia "normal"), nao afeta quais eventos contam como alvo.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO

KS = [0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5, 7.5, 10.0]
ALVO = 3.400823
TETO = 1.5 * ALVO


def escolhe(t: pd.DataFrame) -> pd.Series:
    s = t[t.tr_fp <= TETO]
    if s.empty:
        s = t.copy(); s["d"] = (s.tr_fp - ALVO).abs()
        return s.sort_values("d").iloc[0]
    return s.sort_values(["tr_n", "tr_fp"], ascending=[False, True]).iloc[0]


def main():
    df = canonico()
    falhas_ref = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    falhas = falhas_ref[falhas_ref >= "2025-01-01"].reset_index(drop=True)
    print(f"eventos avaliados: {len(falhas)} (removido 2024-01-16, estruturalmente indetectavel)")
    idx = df.index
    mask = mascara_pontuacao(df)

    print("montando 'out' (walk-forward mensal, com a referencia dos 9 eventos) ...", flush=True)
    out = roda(BRACO, df, falhas_ref)

    linhas = []
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
        t = pd.DataFrame(grid)
        r = escolhe(t)
        kb, kv = r.k_base, r.k_vib

        hold = (idx >= ev - pd.Timedelta(days=7)) & (idx < ev + pd.Timedelta(days=1))
        al_full = alerta_2k(out, mask, kb, kv)
        xh = A.avalia(al_full[hold], pd.Series([ev]), mask[hold])

        linhas.append(dict(evento=ev, k_base=kb, k_vib=kv, tr_n=r.tr_n, tr_fp=r.tr_fp,
                            detectado=xh["det"] >= 1, lead_h=xh["lead_med"]))
        print(f"{ev.strftime('%Y-%m-%d')}  k_base={kb} k_vib={kv}  (treino-nos-outros-7: "
              f"{r.tr_n}/{len(ev_train)} FP={r.tr_fp:.2f})  -> "
              f"{'DETECTADO' if xh['det'] >= 1 else 'perdido'}"
              + (f", lead={xh['lead_med']:.1f}h" if xh["det"] >= 1 else ""), flush=True)

    R = pd.DataFrame(linhas)
    R.to_csv("loeo_2025.csv", index=False)
    print(f"\n=== leave-one-out, so 2025-2026: {int(R['detectado'].sum())}/{len(falhas)} "
          f"({100*R['detectado'].mean():.0f}%) ===")


main()
