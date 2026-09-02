#!/usr/bin/env python3
"""Figura de abertura da apresentacao: O QUE O DETECTOR TEM QUE ACHAR.

Tres familias fisicas de sinal ao longo dos 16 meses, com os 8 trips catalogados
marcados e a mascara de avaliacao desenhada. Sem nenhuma deteccao -- e a figura
do ALVO, nao do resultado; o resultado vem depois em fig_nosso_estilo_francisco.

A mascara e o que separa "dado" de "dado julgavel": so conta o instante em que a
maquina esta rodando (RUNNING_A), quente (T5 > 300 degC) e fora das 6 h seguintes
a um religamento. Sao 353,2 dias julgaveis dentro de 485 de calendario.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from publica_clearml import GRID, BLACKOUT, VIBRATION_TAGS, T0

INK, INK2, MUTED = "#131a20", "#3d4a55", "#6b7885"
RULE, GROUND = "#dde2e6", "#f4f6f7"
AMP = "#b8792a"
STOP = "#c9cfd4"
BLK = "#e6d9c4"
COR = {"t5": "#2a78d6", "mancal": "#eb6834", "vib": "#1baf7a"}
PRE = "954005_624_"

g = pd.read_parquet("grade2min.parquet")
idx = g.index
op = (g["RUNNING_A"] > 0.5).fillna(False)
estavel = op & (g["T5_AVG_A"] > 300)
part = op & ~op.shift(fill_value=False)
n_bl = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID))
blk = part.rolling(n_bl, min_periods=1).max().astype(bool)
sel = idx >= T0
mask = (estavel & ~blk) & sel

fal = pd.read_csv("falhas.csv", parse_dates=["evento"])
alvo = fal[fal.evento.dt.tz_convert("UTC") >= T0]

vib = g[[c for c in VIBRATION_TAGS if c in g.columns]].max(axis=1)
PAINEIS = [
    ("T5_AVG_A", g["T5_AVG_A"], "Exaustão\nT5_AVG_A (°C)", COR["t5"],
     "temperatura de saída da turbina"),
    (PRE + "TI_0305", g[PRE + "TI_0305"], "Metal do mancal\nTI_0305 (°C)", COR["mancal"],
     "o mancal que falha (radial LNA)"),
    ("vib", vib, "Vibração\nmáx. 10 sondas (µm)", COR["vib"],
     "TV_351X … TV_355Y"),
]


def bandas(ax, serie_bool, cor, alpha):
    """Pinta faixas verticais onde a serie booleana diaria e verdadeira."""
    d = serie_bool.loc[serie_bool.index >= T0].resample("1D").mean() > 0.5
    dias, dentro, ini = d.index, False, None
    for i, v in enumerate(d.to_numpy()):
        if v and not dentro:
            ini, dentro = dias[i], True
        elif not v and dentro:
            ax.axvspan(ini, dias[i], color=cor, alpha=alpha, lw=0, zorder=1)
            dentro = False
    if dentro:
        ax.axvspan(ini, dias[-1], color=cor, alpha=alpha, lw=0, zorder=1)


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 8.2), sharex=True, facecolor="white")

    for ax, (nome, serie, rot, cor, sub) in zip(axes, PAINEIS):
        ax.set_facecolor("white")
        bandas(ax, ~op, STOP, 0.55)

        s = serie.loc[serie.index >= T0].resample("30min").median()
        ax.plot(s.index, s.to_numpy(), color=cor, lw=0.75, zorder=3)

        # o blackout de 6 h nunca cobre metade de um dia, entao banda diaria o
        # apagaria; vai como regua fina no topo do painel, que e exato e visivel
        b = (blk & op).loc[idx >= T0].resample("6h").max()
        for t in b.index[b.to_numpy().astype(bool)]:
            ax.axvspan(t, t + pd.Timedelta("6h"), ymin=0.955, ymax=1.0,
                       color=BLK, lw=0, zorder=6)

        # o metal do mancal opera em 30-80 degC; picos de sentinela ate 900 degC
        # achatariam a faixa util. Corta pelo corpo da distribuicao em operacao.
        v = serie.where(mask).dropna()
        if len(v) and nome != "T5_AVG_A":
            lo, hi = np.nanpercentile(v, [0.5, 99.5])
            folga = (hi - lo) * 0.30
            ax.set_ylim(max(0, lo - folga), hi + folga)

        for t in alvo.evento.dt.tz_convert("UTC"):
            ax.axvline(t, color=AMP, ls="--", lw=1.3, alpha=0.95, zorder=5)

        ax.set_ylabel(rot, fontsize=10, color=INK2, linespacing=1.5)
        ax.text(0.004, 0.93, sub, transform=ax.transAxes, fontsize=8.5,
                color=MUTED, va="top", style="italic")
        ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(RULE)
        ax.tick_params(labelsize=9, colors=INK2)

    # numera os 8 trips no painel de cima
    for i, t in enumerate(alvo.evento.dt.tz_convert("UTC"), 1):
        axes[0].annotate(str(i), (t, 1.045), xycoords=("data", "axes fraction"),
                         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                         color=AMP, annotation_clip=False)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    axes[-1].set_xlabel("2025 — 2026", fontsize=10, color=INK2)

    leg = [
        plt.Line2D([0], [0], color=AMP, ls="--", lw=1.4, label="trip catalogado (8)"),
        Patch(facecolor=STOP, alpha=0.55, label="máquina parada"),
        Patch(facecolor=BLK, label="blackout de 6 h pós-religamento (régua no topo)"),
        plt.Line2D([0], [0], color="white", label="  branco = janela avaliada"),
    ]
    axes[0].legend(handles=leg, loc="upper left", bbox_to_anchor=(0.0, 1.42),
                   fontsize=9, ncol=4, frameon=False)

    n_dias = mask.sum() * 2 / 60 / 24
    # `--sem-titulo` para o deck, onde o slide ja carrega o titulo -- repetir os dois
    # gasta a altura util da figura e le como descuido.
    if "--sem-titulo" not in sys.argv:
        fig.suptitle("O alvo — 8 trips do TC-330.03A em 16 meses, nas três famílias de sinal",
                     fontsize=14.5, fontweight="bold", color=INK, x=0.006, ha="left", y=1.005)
    fig.text(0.006, 0.968 if "--sem-titulo" in sys.argv else 0.962,
             f"Janela de avaliação: 01/01/2025 a 30/04/2026 · {n_dias:.0f} dias julgáveis "
             f"de 485 de calendário (máquina rodando, quente e fora do blackout).",
             fontsize=10, color=MUTED, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.945 if "--sem-titulo" in sys.argv else 0.925))
    fig.savefig("fig_alvo.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"dias julgaveis: {n_dias:.1f} de 485    trips: {len(alvo)}")
    print("-> fig_alvo.png")
