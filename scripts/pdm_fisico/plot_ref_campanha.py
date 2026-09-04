#!/usr/bin/env python3
"""Serie temporal comparando as duas referencias, mesmo eixo de tempo:
  painel de cima  = referencia mensal (base, 4.7 FP/mes)
  painel de baixo = referencia por campanha, boot 12h (10.4 FP/mes)
para ver ONDE a reconstrucao gera o excesso de alarme.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import BRACO
from referencia_campanha import alerta_de, K_BASE, K_VIB


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask_base = mascara_pontuacao(df)
    jan48 = [(t - pd.Timedelta(hours=48), t) for t in falhas]

    out_b = roda(BRACO, df, falhas)
    al_b = alerta_de(out_b, mask_base)

    out_c = pd.read_parquet("ref_campanha_out.parquet")
    pont = pd.read_parquet("ref_campanha_pont.parquet")["pontuavel"]
    mask_c = mask_base & pont
    al_c = alerta_de(out_c, mask_c)

    def sinais(out, mask):
        idx = out.index
        def ew(c, hl):
            return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
        return (DET._sustained(ew("t", "1h"), DET.THR_FAM * K_BASE).astype(int)
                + DET._sustained(ew("p", "1h"), DET.THR_FAM * K_BASE).astype(int)
                + DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * K_BASE).astype(int)
                + DET._sustained(ew("vb", "30min"), 3.0 * K_VIB).astype(int)).where(mask, 0)

    n_b, n_c = sinais(out_b, mask_base), sinais(out_c, mask_c)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for ax, n, al, titulo, cor in [
        (axes[0], n_b, al_b, "referência MENSAL (base) — 4,7 FP/mês", "#5b7d99"),
        (axes[1], n_c, al_c, "referência por CAMPANHA, boot 12 h — 10,4 FP/mês", "#a8501d"),
    ]:
        ax.fill_between(n.index, 0, n.to_numpy(), step="mid", color=cor, lw=0, alpha=0.8)
        ax.axhline(2, color="#333", lw=1, ls=":")
        eps = A.episodios(al)
        fp = [(a, b) for a, b in eps
              if not any((a <= t1) and (b >= t0) for t0, t1 in jan48)]
        for a, b in fp:
            ax.axvspan(a, b, color="#8a1c1c", alpha=0.22, lw=0)
        for ev in falhas:
            pego = al[(al.index >= ev - pd.Timedelta(hours=48)) & (al.index < ev)].any()
            ax.axvline(ev, color="#256b58" if pego else "#8a1c1c", lw=1.3,
                       ls="--", alpha=0.9)
        ax.set_ylim(0, 4.5); ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel("nº de sinais")
        ax.set_title(f"{titulo}   ({len(fp)} episódios de falso positivo)", fontsize=10.5)

    axes[0].plot([], [], color="#8a1c1c", alpha=0.22, lw=8, label="episódio de falso positivo")
    axes[0].plot([], [], color="#256b58", ls="--", lw=1.3, label="trip detectado")
    axes[0].plot([], [], color="#8a1c1c", ls="--", lw=1.3, label="trip perdido")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)

    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    axes[1].set_xlim(n_b.index[0], n_b.index[-1])
    fig.suptitle("TC-330.03A — reconstruir a referência a cada campanha dobra o falso positivo",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig("fig_ref_campanha.png", dpi=140)
    print("figura salva: fig_ref_campanha.png")


main()
