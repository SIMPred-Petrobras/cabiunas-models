#!/usr/bin/env python3
"""Validacao leave-one-evento-out do braco 'max+vib_rol' desacoplado.

O split unico treino/teste (corte 2025-07-01) e mais uma fonte de instabilidade:
o ponto recomendado (k_base=1.3, k_vib=5.5) foi escolhido olhando SO os 6
eventos de treino. Aqui, cada um dos 9 eventos sai uma vez -- o (k_base,k_vib)
e escolhido pelos outros 8, e o evento fora e avaliado sem ter influenciado a
escolha. E o teste mais duro que da pra fazer com 9 rotulos sem esperar
trip novo.

Exclusao por fold: so a janela [evento-48h, evento+2h] do proprio evento sai
do calculo de FP/deteccao de treino daquele fold -- exclusao ampla (por mes,
como no split original) contaria alarme legitimo antes de um evento vizinho
no mesmo mes como falso positivo, e enviesaria a escolha.
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
    return s.sort_values(["tr_n", "tr_fp", "tr_p"], ascending=[False, True, True]).iloc[0]


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)

    print("montando 'out' (walk-forward mensal, uma vez) ...", flush=True)
    out = roda(BRACO, df, falhas)

    linhas = []
    for ev in falhas:
        excl = (idx >= ev - pd.Timedelta(hours=48)) & (idx < ev + pd.Timedelta(hours=2))
        train_m = mask & ~excl
        ev_train = falhas[falhas != ev]

        grid = []
        for kb in KS:
            for kv in KS:
                al = alerta_2k(out, train_m, kb, kv)
                am, qm = al[train_m], train_m[train_m]
                x = A.avalia(am, ev_train, qm)
                grid.append(dict(k_base=kb, k_vib=kv, tr_n=x["det"], tr_fp=x["fp_mes"]))
        t = pd.DataFrame(grid)
        t["tr_p"] = np.nan  # p de permutacao caro demais p/ 900 pontos; usa so p/ o ponto escolhido
        r = escolhe(t)
        kb, kv = r.k_base, r.k_vib

        hold = (idx >= ev - pd.Timedelta(days=7)) & (idx < ev + pd.Timedelta(days=1))
        al_full = alerta_2k(out, mask, kb, kv)
        am_h, qm_h = al_full[hold], mask[hold]
        xh = A.avalia(am_h, pd.Series([ev]), qm_h)

        # p de treino do ponto escolhido (permutacao, so aqui, 1x por fold)
        al_tr = alerta_2k(out, train_m, kb, kv)
        xt = A.avalia(al_tr[train_m], ev_train, train_m[train_m])
        xt.update(A.permuta(al_tr[train_m], train_m[train_m], xt["det"], len(ev_train)))

        linhas.append(dict(
            evento=ev, k_base=kb, k_vib=kv,
            tr_det=f"{xt['det']}/{xt['n_ev']}", tr_fp=xt["fp_mes"], tr_p=xt["p"],
            detectado=xh["det"] >= 1, lead_h=xh["lead_med"],
            fp_local_7d=xh["fp"],
        ))
        print(f"{ev.strftime('%Y-%m-%d')}  escolhido k_base={kb} k_vib={kv}  "
              f"(treino-nos-outros-8: {xt['det']}/{xt['n_ev']} FP={xt['fp_mes']:.2f} p={xt['p']:.3f})  "
              f"-> {'DETECTADO' if xh['det'] >= 1 else 'perdido'}"
              + (f", lead={xh['lead_med']:.1f}h" if xh["det"] >= 1 else ""), flush=True)

    R = pd.DataFrame(linhas)
    R.to_csv("loeo.csv", index=False)
    n_det = int(R["detectado"].sum())
    print(f"\n=== leave-one-evento-out: {n_det}/9 detectados quando o proprio evento "
          "NAO participa da escolha do limiar ===")
    print(R[["evento", "k_base", "k_vib", "tr_det", "tr_fp", "detectado", "lead_h"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))


main()
