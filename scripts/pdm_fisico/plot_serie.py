#!/usr/bin/env python3
"""Serie temporal completa: quantos sinais simultaneos (de 4) estao acima do
limiar no ponto recomendado (k_base=1.3, k_vib=5.5), os 9 trips reais
marcados, e os episodios CONFIRMADOS (>=2 sinais) destacados.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import BRACO

K_BASE, K_VIB = 1.3, 5.5


def conta_sinais(out, mask, k_base, k_vib):
    idx = out.index
    def ew(c, hl):
        return out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
    s1 = DET._sustained(ew("t", "1h"), DET.THR_FAM * k_base)
    s2 = DET._sustained(ew("p", "1h"), DET.THR_FAM * k_base)
    s3 = DET._sustained(ew("sp", "30min"), DET.THR_SPREAD * k_base)
    s4 = DET._sustained(ew("vb", "30min"), 3.0 * k_vib)
    n = s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)
    return n.where(mask, 0)


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)

    out = roda(BRACO, df, falhas)
    n = conta_sinais(out, mask, K_BASE, K_VIB)
    alerta = (n >= 2) & mask
    eps = A.episodios(alerta)
    x = A.avalia(alerta[mask], falhas, mask[mask])
    print(f"deteccoes: {x['det']}/{x['n_ev']}  FP={x['fp_mes']:.2f}/mes  episodios={x['episodios']}")

    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.fill_between(n.index, 0, n.to_numpy(), step="mid", color="#5b7d99", lw=0, alpha=0.85,
                     label="sinais simultaneos acima do limiar (0-4)")
    ax.axhline(2, color="#333333", lw=1, ls=":", label="limiar CONFIRMADO (>=2)")

    for a, b in eps:
        ax.axvspan(a, b, color="#a8501d", alpha=0.35, lw=0)

    ymax = 4.3
    for i, ev in enumerate(falhas):
        cor = "#a8501d" if ev >= CORTE else "#256b58"
        ax.axvline(ev, color=cor, lw=1.3, ls="--", alpha=0.85)
        ax.annotate(ev.strftime("%Y-%m-%d"), (ev, ymax), rotation=90, va="top", ha="right",
                    fontsize=7.5, color=cor)

    ax.plot([], [], color="#256b58", ls="--", lw=1.3, label="trip real (treino)")
    ax.plot([], [], color="#a8501d", ls="--", lw=1.3, label="trip real (teste)")
    ax.plot([], [], color="#a8501d", alpha=0.35, lw=8, label="episodio CONFIRMADO do detector")

    ax.set_ylim(0, ymax + 0.3)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("nº de sinais\nsimultâneos")
    ax.set_title(f"TC-330.03A — detector (k_base={K_BASE}, k_vib={K_VIB}) vs. 9 trips reais  "
                 f"— {x['det']}/{x['n_ev']} detectados, {x['fp_mes']:.1f} FP/mês")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(n.index[0], n.index[-1])
    fig.tight_layout()
    fig.savefig("fig_serie_deteccoes.png", dpi=140)
    print("figura salva: fig_serie_deteccoes.png")


main()
