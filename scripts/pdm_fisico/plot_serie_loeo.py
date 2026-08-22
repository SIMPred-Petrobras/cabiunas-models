#!/usr/bin/env python3
"""Mesma figura, mas honesta: mostra o resultado leave-one-out (loeo_2025.csv),
nao o ponto unico escolhido com a amostra inteira.

Como o LOEO usa um (k_base,k_vib) DIFERENTE por evento (o proprio evento nunca
vota na escolha do seu), nao existe uma unica serie continua que represente
"o" detector -- a serie de fundo usa a config que venceu em 7 dos 8 folds
(k_base=1.7, k_vib=2.2); o marcador de cada evento reflete o resultado REAL do
seu proprio fold, la onde as duas configs divergem (2025-11-04, cujo fold
escolheu k_vib=5.5) isso fica anotado explicitamente -- a serie de fundo mostra
um episodio ali que o teste honesto NAO usou.
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

K_BASE_DOM, K_VIB_DOM = 1.7, 2.2   # venceu em 7 dos 8 folds do leave-one-out


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
    falhas_todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    R = pd.read_csv("loeo_2025.csv", parse_dates=["evento"])
    R["evento"] = R["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df)

    out = roda(BRACO, df, falhas_todas)
    n = conta_sinais(out, mask, K_BASE_DOM, K_VIB_DOM)
    alerta = (n >= 2) & mask
    eps = A.episodios(alerta)

    n_det = int(R["detectado"].sum())
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.fill_between(n.index, 0, n.to_numpy(), step="mid", color="#5b7d99", lw=0, alpha=0.75,
                     label=f"sinais simultâneos ({K_BASE_DOM=}, {K_VIB_DOM=} — venceu 7/8 folds)")
    ax.axhline(2, color="#333333", lw=1, ls=":", label="limiar CONFIRMADO (≥2)")
    for a, b in eps:
        ax.axvspan(a, b, color="#9a9a9a", alpha=0.30, lw=0)

    ymax = 4.3
    for _, row in R.iterrows():
        ev = row["evento"]
        ok = bool(row["detectado"])
        cor = "#256b58" if ok else "#a8501d"
        marca = "✓" if ok else "✗"
        ax.axvline(ev, color=cor, lw=1.4, ls="--", alpha=0.9)
        rotulo = ev.strftime("%Y-%m-%d")
        if abs(row["k_base"] - K_BASE_DOM) > 1e-6 or abs(row["k_vib"] - K_VIB_DOM) > 1e-6:
            rotulo += f"  (fold usou k_vib={row['k_vib']:.1f})"
        ax.annotate(f"{marca} {rotulo}", (ev, ymax), rotation=90, va="top", ha="right",
                    fontsize=7.5, color=cor, fontweight="bold")

    ev24 = falhas_todas[falhas_todas < "2025-01-01"].iloc[0]
    ax.axvline(ev24, color="#888888", lw=1.2, ls="--", alpha=0.7)
    ax.annotate("fora do escopo\n(sem histórico)  2024-01-16", (ev24, ymax), rotation=90,
                va="top", ha="right", fontsize=7, color="#888888")

    ax.plot([], [], color="#256b58", ls="--", lw=1.4, label="✓ detectado (leave-one-out)")
    ax.plot([], [], color="#a8501d", ls="--", lw=1.4, label="✗ perdido (leave-one-out)")
    ax.plot([], [], color="#9a9a9a", alpha=0.3, lw=8, label=f"episódio da config {K_BASE_DOM}/{K_VIB_DOM}")

    ax.set_ylim(0, ymax + 0.3)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("nº de sinais\nsimultâneos")
    ax.set_title(f"TC-330.03A — resultado HONESTO (leave-one-evento-out): "
                 f"{n_det}/{len(R)} detectados ({100*n_det/len(R):.0f}%), "
                 f"cada evento avaliado sem votar no próprio limiar")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_xlim(n.index[0], n.index[-1])
    fig.tight_layout()
    fig.savefig("fig_serie_loeo.png", dpi=140)
    print("figura salva: fig_serie_loeo.png")
    print(f"{n_det}/{len(R)} detectados no leave-one-out (fora 2024-01-16, sem historico)")


main()
