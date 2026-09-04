#!/usr/bin/env python3
"""O nosso detector de 4 sinais contra os 20 episodios de TRIP da Secao 18 do EXP10c.

Completa a linha que faltava da matriz de cobertura. Ja tinhamos:

    detector EXP10c  x  32 alarmes de temperatura  -> 96,9% hit (nossa reproducao)
    detector EXP10c  x  20 episodios de TRIP       -> 5/20 (dele), 2/2 no denominador certo
    4 sinais         x  32 alarmes de temperatura  -> 65,6% hit, 16 preditivos
    4 sinais         x  20 episodios de TRIP       -> ESTE SCRIPT

Aviso de construcao, declarado antes do numero. 18 dos 20 episodios ocorrem com a
maquina JA PARADA (T5 entre 27 e 34 degC, ferro frio) -- ver cruza_diego_trip.py. A
nossa mascara de pontuacao exige operacao quente (RUNNING_A e T5>300 degC), entao o
detector NAO PRODUZ ESCORE nesses instantes. Para esses 18 so existe uma via de acerto:
ter alarmado antes da maquina parar, dentro da janela de +-24 h da regua dele. Um
resultado baixo no denominador de 20 nao mede qualidade do detector -- mede que 18
daqueles alvos nao sao eventos de maquina em operacao. Por isso reportamos as duas
contagens, /20 e /2, lado a lado.

REGUA: a dele (+-24 h, preditivo / reativo / sem deteccao), importada de quadrante.py,
para ser comparavel linha a linha com a Tabela 13 do relatorio.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
from quadrante import regua_diego, JAN_H

TRIPS = "../../eval_predictive_out/cruza_diego_trip.csv"


def alvos_trip():
    d = pd.read_csv(TRIPS, parse_dates=["inicio", "fim"])
    d["t"] = d["inicio"].dt.tz_localize("UTC")
    return d


def detalha(al, alvos, mask, rot):
    """Episodio a episodio: o que o nosso detector viu em +-24 h."""
    print(f"\n--- {rot} ---")
    print(f"{'#':>3} {'episodio':>17} {'estado':>10} {'parada':>7} {'Diego':>13} "
          f"{'nos':>13} {'lead/atraso':>12}  mascara ativa em +-24h")
    n_pred = n_reat = n_nada = 0
    for i, (_, r) in enumerate(alvos.iterrows(), 1):
        t = r["t"]
        jan = al.loc[t - pd.Timedelta(hours=JAN_H): t + pd.Timedelta(hours=JAN_H)]
        mj = mask.loc[t - pd.Timedelta(hours=JAN_H): t + pd.Timedelta(hours=JAN_H)]
        on = jan[jan.fillna(False)]
        h_mask = mj.sum() * 2 / 60
        if not len(on):
            cat, dt_ = "sem deteccao", ""
            n_nada += 1
        elif (on.index < t).any():
            a0 = on.index[on.index < t][0]
            cat, dt_ = "PREDITIVO", f"{(t - a0).total_seconds()/3600:+6.1f} h antes"
            n_pred += 1
        else:
            cat, dt_ = "reativo", f"{(on.index[0] - t).total_seconds()/3600:6.1f} h depois"
            n_reat += 1
        print(f"{i:3d} {t:%d/%m/%Y %H:%M} {r['estado']:>10} "
              f"{'sim' if r['parada_real'] else '-':>7} {r['diego']:>13} {cat:>13} "
              f"{dt_:>12}  {h_mask:5.1f} h de {2*JAN_H:.0f} h")
    return n_pred, n_reat, n_nada


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    alvos = alvos_trip()

    dentro = (alvos["t"] >= idx[0]) & (alvos["t"] <= idx[-1])
    print(f"episodios de TRIP: {len(alvos)}  dentro do nosso cache: {int(dentro.sum())}")
    print(f"  com a maquina ja parada: {int((alvos['estado'] == 'parada').sum())}")
    print(f"  com parada real >=2 h  : {int(alvos['parada_real'].sum())}")

    print("\nmontando 'out' (walk-forward mensal) ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)
    teto = trunca(base, 12)

    for nome, al in [("4 sinais (k=1,7/2,2)", base), ("4 sinais + teto de 12 h", teto)]:
        p, r_, n_ = detalha(al, alvos, mask, nome)
        real = alvos[alvos["parada_real"]]
        xr = regua_diego(al, real, mask)
        print(f"\n  denominador de 20 (o do relatorio): {p} preditivos, {r_} reativos, "
              f"{n_} sem deteccao  -> {100*(p+r_)/len(alvos):.1f}% de cobertura")
        print(f"  denominador de 2 (paradas reais)  : {xr['pred']} preditivos, "
              f"{xr['reat']} reativos, {xr['nada']} sem deteccao  -> "
              f"{xr['hit']:.0f}%  lead mediano {xr['lead_med']:.1f} h")

    print("\n" + "=" * 78)
    print("COMPARACAO DIRETA no denominador honesto (2 paradas reais, regua dele)")
    print("=" * 78)
    real = alvos[alvos["parada_real"]]
    xb = regua_diego(base, real, mask)
    print(f"  candidato EXP10c (Tabela 13) : 2/2 preditivos, lead 11,4 e 19,6 h")
    print(f"  nosso 4 sinais               : {xb['pred']}/2 preditivos, "
          f"{xb['reat']} reativos, {xb['nada']} sem deteccao, lead mediano {xb['lead_med']:.1f} h")


if __name__ == "__main__":
    main()
